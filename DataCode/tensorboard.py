import numpy as np
from tensorboardX import SummaryWriter
import os

# Load logs
rewards = np.load(
    "CS272Project/group11custom/group11models/intersection/rewardsdqn.npy"
)
lengths = np.load(
    "CS272Project/group11custom/group11models/intersection/lengthsdqn.npy"
)

log_dir = "tb_logs"
os.makedirs(log_dir, exist_ok=True)

writer = SummaryWriter(logdir=log_dir)

for step, (r, l) in enumerate(zip(rewards, lengths)):
    writer.add_scalar("reward", r, step)
    writer.add_scalar("episode_length", l, step)

writer.close()
print("TensorBoard logs created at:", log_dir)
