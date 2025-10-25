import os,argparse,urllib.request,h5py,sys
import numpy as np
import torch
from o2o.utils.envs import make_env,get_space_spec
from o2o.models.dsr import train_dsr,DSR
from o2o.models.ppo import PPOAgent,ActorGaussian,Critic

BASE="https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2"
NAME={"hopper-medium-v2":"hopper-medium-v2.hdf5","walker2d-medium-v2":"walker2d-medium-v2.hdf5"}
ENV={"hopper-medium-v2":"Hopper-v4","walker2d-medium-v2":"Walker2d-v4"}

def download(task,out):
    url=f"{BASE}/{NAME[task]}"
    if not os.path.exists(out):
        print("Downloading:",url);urllib.request.urlretrieve(url,out);print("Saved:",out)
    else:print("Found:",out)

def h5_to_npz(h5_path,npz_path,max_samples=None):
    with h5py.File(h5_path,'r') as f:
        obs=np.array(f['observations'],dtype=np.float32);acts=np.array(f['actions'],dtype=np.float32)
    if max_samples is not None and max_samples<len(obs):
        idx=np.random.permutation(len(obs))[:max_samples];obs,acts=obs[idx],acts[idx]
    np.savez_compressed(npz_path,states=obs,actions=acts)
    return obs.shape,acts.shape

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--task',required=True,choices=list(NAME.keys()))
    ap.add_argument('--drive_root',required=True)
    ap.add_argument('--epochs_dsr',type=int,default=20)
    ap.add_argument('--total_steps',type=int,default=300_000)
    ap.add_argument('--max_samples',type=int,default=None)
    args=ap.parse_args()

    data_dir=os.path.join(args.drive_root,'data');ckpt_dir=os.path.join(args.drive_root,'checkpts');log_dir=os.path.join(args.drive_root,'logs')
    os.makedirs(data_dir,exist_ok=True);os.makedirs(ckpt_dir,exist_ok=True);os.makedirs(log_dir,exist_ok=True)
    h5_path=os.path.join(data_dir,NAME[args.task]);npz_path=os.path.join(data_dir,args.task.replace('-v2',' ').replace('-','_').strip()+'.npz')
    dsr_out=os.path.join(ckpt_dir,args.task.replace('-v2',' ').replace('-','_').strip()+'_dsr.pt')
    env_id=ENV[args.task]

    download(args.task,h5_path)
    if not os.path.exists(npz_path):
        shapes=h5_to_npz(h5_path,npz_path,args.max_samples);print('Converted to NPZ:',npz_path,'shapes=',shapes)
    else:print('Found NPZ:',npz_path)

    device='cuda' if torch.cuda.is_available() else 'cpu';print('Device:',device)
    data=np.load(npz_path);states=data['states'].astype(np.float32);actions=data['actions'].astype(np.float32)
    print('Dataset:',states.shape,actions.shape)
    env=make_env(env_id,seed=0);spec=get_space_spec(env)
    dsr=train_dsr(states,actions,discrete=spec.discrete,action_dim=spec.action_dim,action_low=spec.action_low,action_high=spec.action_high,hidden=(256,256),activation='relu',num_negatives=1,lr=3e-4,batch_size=2048,epochs=args.epochs_dsr,device=device,action_noise=0.0,temperature=1.0)
    torch.save(dsr.state_dict(),dsr_out);print('Saved DSR:',dsr_out)

    low=np.array(spec.action_low_vec,dtype=np.float32);high=np.array(spec.action_high_vec,dtype=np.float32)
    actor=ActorGaussian(spec.state_dim,spec.action_dim,action_low=low,action_high=high,hidden=(256,256),activation='tanh')
    critic=Critic(spec.state_dim,hidden=(256,256),activation='tanh')
    agent=PPOAgent(actor=actor,critic=critic,dsr=dsr,discrete=spec.discrete,action_dim=spec.action_dim,device=device,lr_actor=3e-4,lr_critic=3e-4,clip_ratio=0.2,target_kl=0.01,vf_coeff=0.5,ent_coeff=0.0,gamma=0.99,gae_lambda=0.95,pessimism_beta=1.0,pessimism_gamma=1.0,bonus_eta=0.1,bonus_center=0.7,bonus_sigma=0.15,use_bonus=True)

    from collections import deque
    obs,_=env.reset(seed=0);ep_ret,ep_len=0.0,0;ep_returns=deque(maxlen=10)
    buf={k:[] for k in ['obs','act','rew','val','logp','done','support']}
    t=0;steps_per_epoch=4096;minibatch_size=256;train_iters=10;total_steps=args.total_steps
    while t<total_steps:
        for _ in range(steps_per_epoch):
            a,logp,_=agent.select_action(obs)
            next_obs,r,done,truncated,_=env.step(a)
            sup=agent.compute_support(obs,np.array(a));bonus=agent.compute_bonus(np.array([sup]))[0];r_total=float(r+bonus)
            v=agent.evaluate_value(obs)
            buf['obs'].append(obs);buf['act'].append(a);buf['rew'].append(r_total);buf['val'].append(v);buf['logp'].append(logp);buf['done'].append(float(done or truncated));buf['support'].append(sup)
            ep_ret+=r;ep_len+=1;t+=1;obs=next_obs
            if done or truncated:
                ep_returns.append(ep_ret);obs,_=env.reset();ep_ret,ep_len=0.0,0
            if t>=total_steps:break
        last_val=agent.evaluate_value(obs)
        rewards=np.array(buf['rew'],dtype=np.float32);values=np.array(buf['val'],dtype=np.float32);dones=np.array(buf['done'],dtype=np.float32)
        values_plus=np.concatenate([values,np.array([last_val],dtype=np.float32)])
        adv,ret=agent.compute_advantages(rewards,values_plus,dones,agent.gamma,agent.gae_lambda)
        batch={'obs':np.array(buf['obs'],dtype=np.float32),'act':np.array(buf['act'],dtype=np.float32),'adv':adv.astype(np.float32),'ret':ret.astype(np.float32),'logp':np.array(buf['logp'],dtype=np.float32),'support':np.array(buf['support'],dtype=np.float32)}
        metrics=agent.update(batch,minibatch_size=minibatch_size,train_iters=train_iters)
        buf={k:[] for k in buf}
        avg_ret=float(np.mean(ep_returns)) if len(ep_returns) else 0.0
        sup=batch['support'];sup_mean=float(np.mean(sup)) if sup.size else 0.0;p10=float(np.percentile(sup,10)) if sup.size else 0.0;p90=float(np.percentile(sup,90)) if sup.size else 0.0
        print(f"Steps {t}/{total_steps}  AvgEpRet {avg_ret:.2f}  DSR[mean/p10/p90] {sup_mean:.2f}/{p10:.2f}/{p90:.2f}  pess_reg {metrics.get('pess_reg',0.0):.4f}")
    env.close()

if __name__=='__main__':
    main()

