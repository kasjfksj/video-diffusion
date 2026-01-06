import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def create_2d_spiral(n_samples=1000, noise=0.1):
    """Create a 2D spiral manifold."""
    t = np.linspace(0, 4 * np.pi, n_samples)
    x = t * np.cos(t)
    y = t * np.sin(t)
    
    # Add small noise
    x += np.random.randn(n_samples) * noise
    y += np.random.randn(n_samples) * noise
    
    return np.column_stack([x, y])

def random_projection(data, d):
    """Project data to d-dimensional space using random orthogonal matrix."""
    n_features = data.shape[1]
    # Create random orthogonal matrix
    A = np.random.randn(n_features, d)
    Q, _ = np.linalg.qr(A)
    proj_matrix = Q
    projected = data @ Q
    return projected, proj_matrix

class DiffusionModel(nn.Module):
    """Simple diffusion model with 5-layer ReLU MLP."""
    def __init__(self, input_dim, hidden_dim=256, num_layers=5):
        super().__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, input_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x, t):
        """Forward pass predicting output at timestep t."""
        return self.network(x)

class DiffusionTrainer:
    """Diffusion model trainer with three prediction objectives."""
    def __init__(self, data, num_timesteps=100, device='cpu', prediction_type='epsilon'):
        """
        prediction_type: 'epsilon' (noise prediction), 'x' (data prediction), or 'velocity' (v prediction)
        """
        self.device = device
        self.data = torch.FloatTensor(data).to(device)
        self.num_timesteps = num_timesteps
        self.data_dim = data.shape[1]
        self.prediction_type = prediction_type
        
        # Linear schedule
        betas = np.linspace(0.0001, 0.02, num_timesteps)
        alphas = 1.0 - betas
        alpha_cumprod = np.cumprod(alphas)
        self.betas = torch.FloatTensor(betas).to(device)
        self.alphas = torch.FloatTensor(alphas).to(device)
        self.alpha_cumprod = torch.FloatTensor(alpha_cumprod).to(device)
        
        # Precompute sqrt terms
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)
        
        self.model = DiffusionModel(self.data_dim, hidden_dim=256, num_layers=5).to(device)
    
    def train(self, epochs=1000, batch_size=64, lr=0.001):
        """Train the diffusion model."""
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        dataset = TensorDataset(self.data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                x_0 = batch[0][0]
                
                # Random timesteps
                t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device=self.device)
                
                # Generate noise
                noise = torch.randn_like(x_0)
                
                # Forward diffusion: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
                alpha_t = self.sqrt_alpha_cumprod[t].view(-1, 1)
                sigma_t = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1)
                x_t = alpha_t * x_0 + sigma_t * noise
                
                # Predict based on prediction type
                pred = self.model(x_t, t)
                
                if self.prediction_type == 'epsilon':
                    # Epsilon prediction: predict noise
                    target = noise
                elif self.prediction_type == 'x':
                    # x prediction: predict original data
                    target = x_0
                elif self.prediction_type == 'velocity':
                    # Velocity prediction: predict v = sqrt(alpha_cumprod_t) * noise - sqrt(1 - alpha_cumprod_t) * x_0
                    target = alpha_t * noise - sigma_t * x_0
                else:
                    raise ValueError(f"Unknown prediction type: {self.prediction_type}")
                
                loss = nn.MSELoss()(pred, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(loader):.6f}")
        
        print(f"  Training complete ({self.prediction_type} prediction)")
    
    def sample(self, n_samples=1000):
        """Generate samples using reverse diffusion."""
        self.model.eval()
        with torch.no_grad():
            x_t = torch.randn(n_samples, self.data_dim, device=self.device)
            
            for t in range(self.num_timesteps - 1, -1, -1):
                t_tensor = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
                
                pred = self.model(x_t, t_tensor)
                
                alpha_t = self.alpha_cumprod[t]
                alpha_t_prev = self.alpha_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=self.device)
                beta_t = self.betas[t]
                
                if self.prediction_type == 'epsilon':
                    # Reverse step for epsilon prediction
                    coeff = beta_t / torch.sqrt(1 - alpha_t)
                    x_0_pred = (x_t - coeff * pred) / torch.sqrt(torch.tensor(1.0, device=self.device) - beta_t)
                    
                elif self.prediction_type == 'x':
                    # Reverse step for x prediction
                    x_0_pred = pred
                    
                elif self.prediction_type == 'velocity':
                    # Reverse step for velocity prediction
                    sigma_t = torch.sqrt(1 - alpha_t)
                    x_0_pred = (x_t - sigma_t * pred) / torch.sqrt(alpha_t)
                
                # Compute mean
                mean = (torch.sqrt(alpha_t_prev) * beta_t / (1 - alpha_t)) * x_0_pred + \
                       (torch.sqrt(torch.tensor(1.0, device=self.device) - beta_t) * (1 - alpha_t_prev) / (1 - alpha_t)) * x_t
                
                x_t = mean
                
                if t > 0:
                    noise = torch.randn_like(x_t)
                    variance = (1 - alpha_t_prev) / (1 - alpha_t) * beta_t
                    x_t = x_t + torch.sqrt(variance) * noise
        
        return x_t.cpu().numpy()

class GenerativeModel:
    """Simple MLP generative model (baseline)."""
    def __init__(self, input_dim, hidden_dim=256):
        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden_dim,) * 5,
            activation='relu',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            learning_rate_init=0.001
        )
        self.input_dim = input_dim
    
    def train(self, data):
        """Train generative model."""
        latent_samples = np.random.randn(data.shape[0], 1)
        self.model.fit(latent_samples, data)
    
    def sample(self, n_samples=1000):
        """Generate samples."""
        latent_samples = np.random.randn(n_samples, 1)
        return self.model.predict(latent_samples)

def plot_results(ground_truth, d_values, projection_matrices, x_preds, 
                 diffusion_x_preds, diffusion_eps_preds, diffusion_v_preds):
    """Create comparison figure for all prediction methods."""
    fig, axes = plt.subplots(len(d_values), 5, figsize=(18, 14))
    
    for i, D in enumerate(d_values):
        proj_matrix = projection_matrices[D]
        
        # Ground truth
        ax = axes[i, 0]
        ax.scatter(ground_truth[:, 0], ground_truth[:, 1], c='orange', s=2, alpha=0.6)
        ax.set_ylabel(f'D={D}', fontsize=11, fontweight='bold')
        if i == 0:
            ax.set_title('ground-truth', fontsize=11, fontweight='bold')
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # x-prediction (MLP trained on high-dim data)
        ax = axes[i, 1]
        x_pred_2d = x_preds[D] @ proj_matrix.T
        ax.scatter(x_pred_2d[:, 0], x_pred_2d[:, 1], c='steelblue', s=2, alpha=0.6)
        if i == 0:
            ax.set_title('x-pred\n(MLP)', fontsize=11, fontweight='bold')
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Diffusion x-prediction
        ax = axes[i, 2]
        diff_x_pred_2d = diffusion_x_preds[D] @ proj_matrix.T
        ax.scatter(diff_x_pred_2d[:, 0], diff_x_pred_2d[:, 1], c='steelblue', s=2, alpha=0.6)
        if i == 0:
            ax.set_title('diffusion x-pred', fontsize=11, fontweight='bold')
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Diffusion epsilon-prediction
        ax = axes[i, 3]
        diff_eps_pred_2d = diffusion_eps_preds[D] @ proj_matrix.T
        ax.scatter(diff_eps_pred_2d[:, 0], diff_eps_pred_2d[:, 1], c='steelblue', s=2, alpha=0.6)
        if i == 0:
            ax.set_title('diffusion ε-pred', fontsize=11, fontweight='bold')
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Diffusion velocity-prediction
        ax = axes[i, 4]
        diff_v_pred_2d = diffusion_v_preds[D] @ proj_matrix.T
        ax.scatter(diff_v_pred_2d[:, 0], diff_v_pred_2d[:, 1], c='steelblue', s=2, alpha=0.6)
        if i == 0:
            ax.set_title('diffusion v-pred', fontsize=11, fontweight='bold')
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_aspect('equal')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('toy_experiment_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create ground truth 2D spiral
    n_samples = 1000
    ground_truth = create_2d_spiral(n_samples, noise=0.1)
    
    # Embedding dimensions to test
    d_values = [2, 8, 16, 512]
    
    x_preds = {}
    diffusion_x_preds = {}
    diffusion_eps_preds = {}
    diffusion_v_preds = {}
    projection_matrices = {}
    
    for D in d_values:
        print(f"\n{'='*60}")
        print(f"Processing D={D}...")
        print(f"{'='*60}")
        
        # Project to D-dimensional space
        x_high_dim, proj_matrix = random_projection(ground_truth, D)
        projection_matrices[D] = proj_matrix
        
        # x-prediction: Train generative model on high-dim data
        print(f"Training MLP for D={D}...")
        model_x = GenerativeModel(D)
        model_x.train(x_high_dim)
        x_preds[D] = model_x.sample(n_samples)
        print(f"MLP training complete")
        
        # Diffusion x-prediction
        print(f"Training Diffusion Model (x-prediction) for D={D}...")
        diffusion_x = DiffusionTrainer(x_high_dim, num_timesteps=100, device=device, prediction_type='x')
        diffusion_x.train(epochs=20000, batch_size=64, lr=0.001)
        diffusion_x_preds[D] = diffusion_x.sample(n_samples)
        
        # Diffusion epsilon-prediction
        print(f"Training Diffusion Model (epsilon-prediction) for D={D}...")
        diffusion_eps = DiffusionTrainer(x_high_dim, num_timesteps=100, device=device, prediction_type='epsilon')
        diffusion_eps.train(epochs=20000, batch_size=64, lr=0.001)
        diffusion_eps_preds[D] = diffusion_eps.sample(n_samples)
        
        # Diffusion velocity-prediction
        print(f"Training Diffusion Model (velocity-prediction) for D={D}...")
        diffusion_v = DiffusionTrainer(x_high_dim, num_timesteps=100, device=device, prediction_type='velocity')
        diffusion_v.train(epochs=20000, batch_size=64, lr=0.001)
        diffusion_v_preds[D] = diffusion_v.sample(n_samples)
    
    # Plot results
    print(f"\n{'='*60}")
    print("Plotting results...")
    print(f"{'='*60}")
    plot_results(ground_truth, d_values, projection_matrices, 
                 x_preds, diffusion_x_preds, diffusion_eps_preds, diffusion_v_preds)
    print("Figure saved as 'toy_experiment_predictions.png'")

if __name__ == "__main__":
    main()