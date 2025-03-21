import os
import torch
from gym.wrappers import RecordVideo
from env import create_env
from agent_eval import AgentEval  # Or use your Agent class if appropriate

# Set up your args manually or load them (adjust as needed)
class Args:
    seed = 123
    disable_cuda = False
    device = torch.device('cpu')
    environment_filename = './ObstacleTower/obstacletower'
    # Other parameters (not used in demo, but required by your agent)
    atoms = 51
    V_min = -10
    V_max = 10
    model = None
    worker = 0  # Use the same worker id as used when training the intermediate model

args = Args()
if torch.cuda.is_available() and not args.disable_cuda:
    args.device = torch.device('cuda')

# Create the environment (use the same custom flag, worker id, etc. as in training)
env = create_env(
    args.environment_filename,
    custom=True,
    worker_id=1,           # For demo, choose an available worker id
    device=args.device
)

# Optional: Wrap with RecordVideo to capture a video (all episodes)
env = RecordVideo(env, video_folder='videos/', episode_trigger=lambda e: True)

# Create the agent in evaluation mode. Here we use AgentEval,
# which loads the network in eval mode.
agent = AgentEval(args, env)

# Load your intermediate checkpoint (adjust the path accordingly)
checkpoint_path = 'results/intermediate_worker0_695513_20250311_011932/model.pth'
state_dict = torch.load(checkpoint_path, map_location=args.device)
agent.online_net.load_state_dict(state_dict)
agent.online_net.eval()

# Run a demo episode:
state = env.reset()
done = False
while not done:
    # Get action from the agent's act_e_greedy (or act) method.
    action = agent.act_e_greedy(state, epsilon=0.0)  # No exploration noise for demo
    state, reward, done, info = env.step(action)
    env.render()  # This will display the frame (if you have a display) or update video recording

env.close()
