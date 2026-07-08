import torch

class HFSRAMServer:
    def __init__(self, initial_model, rho=2.0, gamma=0.5):
        self.w_t = {k: v.cpu().clone() for k, v in initial_model.state_dict().items()}
        self.w_best = {k: v.cpu().clone() for k, v in initial_model.state_dict().items()}
        self.z_t = {k: torch.zeros_like(v) for k, v in initial_model.state_dict().items()}
        
        self.rho = rho
        self.gamma = gamma
        self.best_loss = float('inf')

    def aggregate_and_primal_update(self, client_meta_grads, client_v_t):
        """
        Computes the closed-form primal update w_{t+1} resolving the augmented Lagrangian.
        """
        num_sampled = len(client_meta_grads)
        w_next = {}
        
        for k in self.w_t.keys():
            # Average Hessian-free meta gradients
            g_hf_avg = sum(grads[k] for grads in client_meta_grads) / num_sampled
            # Average consensus variables from clients
            v_t_avg = sum(v[k] for v in client_v_t) / num_sampled
            
            # Closed-form w-minimization (derived in Lemma 1)
            # w_{t+1} = (\rho * v_t + \gamma * w_{best, t} - z_t - g_hf) / (\rho + \gamma)
            w_next[k] = (self.rho * v_t_avg + self.gamma * self.w_best[k] - self.z_t[k] - g_hf_avg) / (self.rho + self.gamma)
            
        return w_next

    def update_dual_multiplier(self, w_next, client_v_next):
        """
        Accumulates residual consensus error in the dual multiplier z_{t+1}.
        """
        num_sampled = len(client_v_next)
        
        for k in self.z_t.keys():
            v_next_avg = sum(v[k] for v in client_v_next) / num_sampled
            # z_{t+1} = z_t + \rho(w_{t+1} - v_{t+1})
            self.z_t[k] = self.z_t[k] + self.rho * (w_next[k] - v_next_avg)
            
    def update_w_best(self, current_loss):
        """Updates the swarm anchor if the current optimization step yields a lower global loss."""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.w_best = {k: v.clone() for k, v in self.w_t.items()}
