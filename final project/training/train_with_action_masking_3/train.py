import  os
import  ray
from ray.rllib.agents.ppo import  PPOTrainer
from ray.rllib.models import  ModelCatalog
from training.train_with_action_masking_3.bomberman_multi_env import  BombermanEnv
from ray import  tune
from training.train_with_action_masking_3.callbacks import  MyCallbacks
from training.train_with_action_masking_3.torchnet_with_masking import  ComplexTorchInputNetwork

if __name__ == '__main__':
    ray.init(
        _redis_max_memory=1024 * 1024 * 100,num_gpus=1, object_store_memory=10*2**30)
    env = BombermanEnv([f'agent_{i}' for i in range(4)])

    ModelCatalog.register_custom_model("custom_model", ComplexTorchInputNetwork)
    tune.register_env('BomberMan-v0', lambda c: BombermanEnv([f'agent_{i}' for i in range(4)]))


    def policy_mapping_fn(agent_id):
        if agent_id.startswith("agent_0"):# or np.random.rand() > 0.2:
            return "policy_01"  # Choose 01 policy for agent_01
        else:
            return "policy_02"

    def train(config, checkpoint_dir=None):
        trainer = PPOTrainer(config=config, env='BomberMan-v0')
        iter = 0

        #def update_phase(ev):
        #    ev.foreach_env(lambda e: e.set_phase(phase))

        while True:
            iter += 1
            result = trainer.train()
            if iter % 200 == 1:
                if not os.path.exists(f'./model-{iter}'):
                    trainer.get_policy('policy_01').export_model(f'./model-{iter}')
                else:
                    print("model already saved")

    train(config={
        'env': 'BomberMan-v0',
            "use_critic": True,
            'callbacks': MyCallbacks,
            "use_gae": True,
            'lambda': 0.95,
            'gamma': 0.98,
            'kl_coeff': 0.2,
            'vf_loss_coeff' : 0.5,
            'clip_rewards': False,
            'entropy_coeff': 0.0001,
            'train_batch_size': 16384,#49152,
            'sgd_minibatch_size': 256,
            'shuffle_sequences': True,
            'num_sgd_iter': 15,
            'num_workers': 4,
            #'num_cpus_per_worker': ,
            'ignore_worker_failures': True,
            'num_envs_per_worker': 4,
            #"model": {
            #    "fcnet_hiddens": [512, 512],
            #},
            "model": {
                "custom_model": "custom_model",
                "dim": 15,
                #"conv_filters": [[48, [7, 7], 2], [96, [3,3], 2], [192, [3,3], 2], [192, [1,1], 1]],
                "conv_filters": [[64, [3, 3], 1], [64, [3, 3], 1],[64, [3, 3], 1], [64, [3, 3], 1], [64, [3, 3], 1], [64, [3, 3], 1], [64, [3, 3], 1], [64, [3, 3], 1], [2, [1, 1], 1]],
                #8k run "conv_filters" : [[32, [5,5], 2], [32, [3,3], 2], [64, [3,3], 2], [128, [3,3], 2], [256, [1,1], 1]],
                "conv_activation" : "relu",
                "post_fcnet_hiddens": [256],
                "post_fcnet_activation": "relu",
                 #"fcnet_hiddens": [256,256],
                 "vf_share_layers": 'true'
                 },
            'rollout_fragment_length': 1024,
            'batch_mode': 'truncate_episodes',
            'observation_filter': 'NoFilter',
            'num_gpus': 1,
            'lr': 3e-4,
            #"lr_schedule": [[0, 0.0005], [5e6, 0.0005], [5e6+1, 0.0003]],#, [2e7, 0.0003], [2e7+1, 0.0001]],
            'log_level': 'INFO',
            'framework': 'torch',
            #'simple_optimizer': args.simple,
            'multiagent': {
                "policies": {
                    "policy_01": (None, env.observation_space, env.action_space, {}),
                    "policy_02": (None, env.observation_space, env.action_space, {}),
                },
                "policies_to_train": ["policy_01"],
                'policy_mapping_fn':
                    policy_mapping_fn,
            },
    }, )