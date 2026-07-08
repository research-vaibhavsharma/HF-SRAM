import torch
import copy

class HFSRAMClient:
    def __init__(self, client_id, dataloader, model_template, device, lr_alpha=0.01):
        self.id = client_id
        self.dataloader = dataloader
        self.device = device
        self.lr_alpha = lr_alpha
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Local state variables for Swarm ADMM
        self.v_t = {k: torch.zeros_like(v) for k, v in model_template.state_dict().items()}
        self.V_vel = {k: torch.zeros_like(v) for k, v in model_template.state_dict().items()}
        
    def local_adaptation_and_gradient(self, w_t):
        """
        Computes the Hessian-free meta-gradient after a single step of local adaptation.
        Note: The hesitation parameter is strictly fixed at h_i = 1 for stability.
        """
        # 1. Load global weights w_t
        model = copy.deepcopy(w_t).to(self.device)
        model.train()
        
        # Sample a batch of data
        x, y = next(iter(self.dataloader))
        x, y = x.to(self.device), y.to(self.device)
        
        # 2. Local Adaptation: \phi_i(w_t) = w_t - \alpha \nabla F_i(w_t)
        outputs = model(x)
        loss = self.criterion(outputs, y)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
        
        # Apply first-order adaptation (Hessian-free)
        adapted_state_dict = copy.deepcopy(model.state_dict())
        for (name, param), grad in zip(model.named_parameters(), grads):
            adapted_state_dict[name] -= self.lr_alpha * grad
            
        model.load_state_dict(adapted_state_dict)
        
        # 3. Compute Meta-Gradient at adapted weights: g_i = \nabla F_i(\phi_i(w_t))
        outputs_meta = model(x)
        loss_meta = self.criterion(outputs_meta, y)
        meta_grads = torch.autograd.grad(loss_meta, model.parameters())
        
        # Package first-order gradients
        g_hf = {name: grad.clone().detach() for name, grad in zip(adapted_state_dict.keys(), meta_grads)}
        return g_hf

    def update_swarm_consensus(self, w_next, w_best, z_t, rho, beta=0.9, c1=2.0, c2=2.0):
        """
        Updates the evolutionary swarm velocity and auxiliary consensus variable v_{t+1}.
        """
        for k in self.v_t.keys():
            # Stochastic parameters r1, r2 ~ U(0,1)
            r1 = torch.rand_like(self.V_vel[k])
            r2 = torch.rand_like(self.V_vel[k])
            
            # Velocity update: V_{t+1} = \beta V_t + c_1 r_1 \odot (w_{t+1} - v_t) + c_2 r_2 \odot (w_{best, t} - v_t)
            self.V_vel[k] = (beta * self.V_vel[k] + 
                             c1 * r1 * (w_next[k] - self.v_t[k]) + 
                             c2 * r2 * (w_best[k] - self.v_t[k]))
            
            # Consensus update: v_{t+1} = v_t + V_{t+1} + (1/\rho) z_t
            self.v_t[k] = self.v_t[k] + self.V_vel[k] + (1.0 / rho) * z_t[k]
            
        return self.v_t
