import os
import numpy as np
import torch
from torch import optim

from model import DQN, ResDQN


def create_model(args, action_space, device='cpu'):
    if args.large:
        net = ResDQN(args, action_space).to(device=device)
    else:
        net = DQN(args, action_space).to(device=device)
    return net


class Agent:
    def __init__(self, args, env):
        self.action_space = env.action_space
        self.atoms = args.atoms
        self.v_min = args.V_min
        self.v_max = args.V_max
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(
            device=args.device)  # Support (range) of z
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.batch_size = args.batch_size
        self.n = args.multi_step
        self.discount = args.discount

        self.online_net = create_model(args, self.action_space, args.device)
        for m in self.online_net.modules():
            print(m)

        if args.model and os.path.isfile(args.model):
            # Always load tensors onto CPU by default, will shift to GPU if necessary
            self.online_net.load_state_dict(torch.load(args.model, map_location='cpu'))
        self.online_net.train()

        self.target_net = create_model(args, self.action_space, args.device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.lr, eps=args.adam_eps)

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state):
        with torch.no_grad():
            return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()

    # Acts with an ε-greedy policy (used for evaluation only)
    def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
        return self.action_space.sample() if np.random.random() < epsilon else self.act(state)

    def learn(self, mem):
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

        # Calculate current state probabilities (online network noise already sampled)
        log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
        log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

        with torch.no_grad():
            # Calculate nth next state probabilities
            pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
            dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))

            # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            argmax_indices_ns = dns.sum(2).argmax(1)
            self.target_net.reset_noise()  # Sample new target net noise
            pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)

            # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
            pns_a = pns[range(self.batch_size), argmax_indices_ns]

            # Compute Tz (Bellman operator T applied to z)
            # Tz = R^n + (γ^n)z (accounting for terminal states)
            T_z = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)
            T_z = T_z.clamp(min=self.v_min, max=self.v_max)  # Clamp between supported values

            # Compute L2 projection of Tz onto fixed support z
            b = (T_z - self.v_min) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)

            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = states.new_zeros(self.batch_size, self.atoms)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(
                self.batch_size, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1),
                                  (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1),
                                  (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))

        self.online_net.zero_grad()

        (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss

        self.optimiser.step()

        mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Save model parameters on current device (don't move model between devices)
    def save(self, path):
        os.makedirs(path, exist_ok=True)  # Ensure directory exists
        torch.save(self.online_net.state_dict(), os.path.join(path, 'model.pth'))

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        with torch.no_grad():
            return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()
