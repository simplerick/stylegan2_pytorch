import sys
sys.path.append("../PerceptualSimilarity/")
from models import PerceptualLoss




class Projector():
    def __init__(self,
                 G, # generator
                 image_loss, # loss for comparing generated and target images
                 noise_reg_weight=1e-5, # weight for noise regularization,  None -> regularization disabled
                 dlatent_avg_samples = 10000,  # number of samples for computing avg and std
                 num_steps=1000,  # number of optimization steps
                 max_lr = 0.1, lr_rampdown_length = 0.25, lr_rampup_length = 0.05, # lr params
                 noise_factor = 0.05, noise_ramp_length  = 0.75 # dlatents noise params
                ):
        for n, v in locals().items():
            if n != 'self':
                setattr(self, n, v)


    def compute_init_approx(self):
        dlatents = self.G.sample(self.dlatent_avg_samples).detach()
        self.dlatents_avg = torch.mean(dlatents, dim = 0)
        self.dlatents_std = (torch.sum((dlatents - self.dlatents_avg) ** 2)/self.dlatent_avg_samples) ** 0.5
        min_res = self.G.const.shape[-1]
        self.noise_maps_shape = [len(self.G.layers),2,1,2,min_res*2**(i+1),min_res*2**(i+1)]


    def lr_schedule(self, step):
        t = step/self.num_steps
        x = min(1, t/self.lr_rampup_length)
        if t > (1-self.lr_rampdown_length):
            x = np.sin(0.5*np.pi*(1-t)/self.lr_rampdown_length)
        return(x)


    @staticmethod
    def show(image):
        display.clear_output(wait=True)
        plt.imshow(image)
        plt.show(image.data.cpu().permute(0,2,3,1).squeeze())


    def run(self, target_image):
        if not hasattr(self, 'dlatents_avg'):
            self.compute_init_approx()
        # initialize optimizable variables
        dlatents = self.dlatents_avg.clone().requires_grad_(True)
        noise_maps = torch.randn(self.noise_maps_shape).requires_grad_(True)
        # create opt and lr scheduler
        optimizer = torch.optim.Adam([dlatents, *noises], lr=self.max_lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.lr_schedule)
        # iterations
        for step in range(self.num_steps):
            # generate noise for dlatents
            noise_strength = self.noise_factor*max(1-(step/self.num_steps)/self.noise_ramp_length, 0)
            dlatents_noise = torch.normal(0, self.dlatents_std*noise_strength)
            # generate image
            image = self.G.deterministic_generate(dlatents + dlatents_noise, noise_maps)
            Projector.show(image)
            # compute loss
            loss = self.image_loss(image, target_image)
            if not self.noise_reg_weight is None:
                loss += self.noise_reg_weight * Noise_reg(noise_maps)
            # backward pass, opt and scheduler steps
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        return dlatents.data, noises.data
