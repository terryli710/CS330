# configs.py
import json


_dream_config_tree = {
  "environment": "vanilla",
  "instruction_agent": {
    "type": "learned",
    "policy": {
      "type": "vanilla",
      "epsilon_schedule": {
        "begin": 1.0,
        "end": 0.01,
        "total_steps": 250000
      },
      "embedder": {
        "type": "instruction",
        "obs_embedder": {
          "embed_dim": 64
        },
        "instruction_embedder": {
          "embed_dim": 64
        },
        "transition_embedder": {
          "state_embed_dim": 64,
          "action_embed_dim": 32,
          "embed_dim": 64
        },
        "trajectory_embedder": {
          "type": "ours",
          "penalty": 0.1
        },
        "attention_query_dim": 64,
        "embed_dim": 64
      },
      "test_epsilon": 0,
      "discount": 0.99
    },
    "buffer": {
      "type": "vanilla",
      "max_buffer_size": 100000
    },
    "learning_rate": 0.0001,
    "sync_target_freq": 5000,
    "min_buffer_size": 500,
    "batch_size": 32,
    "update_freq": 4,
    "max_grad_norm": 10
  },
  "exploration_agent": {
    "type": "learned",
    # "type": "random",
    "policy": {
      "type": "recurrent",
      "epsilon_schedule": {
        "begin": 1.0,
        "end": 0.01,
        "total_steps": 250000
      },
      "embedder": {
        "type": "recurrent",
        "experience_embedder": {
          "state_embed_dim": 64,
          "action_embed_dim": 16,
          "embed_dim": 64
        },
        "embed_dim": 64
      },
      "test_epsilon": 0,
      "discount": 0.99
    },
    "buffer": {
      "type": "sequential",
      "max_buffer_size": 16000,
      "sequence_length": 50
    },
    "learning_rate": 0.0001,
    "sync_target_freq": 5000,
    "min_buffer_size": 8000,
    "batch_size": 32,
    "update_freq": 4,
    "max_grad_norm": 10
  }
}

with open("./dream.json", "w") as f:
  json.dump(_dream_config_tree, f, indent=4, sort_keys=True)

_rl2_config_tree = {
  "environment": "vanilla",
  "agent": {
    # "type": "learned",
    "type": "random",
    "policy": {
      "type": "recurrent",
      "epsilon_schedule": {
        "begin": 1.0,
        "end": 0.01,
        "total_steps": 500000
      },
      "embedder": {
        "type": "recurrent",
        "experience_embedder": {
          "state_embed_dim": 64,
          "instruction_embed_dim": 64,
          "action_embed_dim": 16,
          "reward_embed_dim": 16,
          "done_embed_dim": 16,
          "embed_dim": 64
        },
        "embed_dim": 64
      },
      "test_epsilon": 0,
      "discount": 0.99
    },
    "buffer": {
      "type": "sequential",
      "max_buffer_size": 16000,
      "sequence_length": 50
    },
    "learning_rate": 0.0001,
    "sync_target_freq": 5000,
    "min_buffer_size": 800,
    "batch_size": 32,
    "update_freq": 4,
    "max_grad_norm": 10
  }
}


with open("./rl2.json", "w") as f:
  json.dump(_rl2_config_tree, f, indent=4, sort_keys=True)