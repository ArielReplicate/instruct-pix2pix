import subprocess

from edit_cli import *

from cog import BasePredictor, Path, Input

sys.path.append("./stable_diffusion")


class Predictor(BasePredictor):
    def setup(self):
        subprocess.run(["mkdir", "/root/.cache/torch/hub/checkpoints", "-p"])
        subprocess.run(["mv", "checkpoint_liberty_with_aug.pth", "/root/.cache/torch/hub/checkpoints/"])
        subprocess.run(["mv", "cache/torch", "/root/.cache/torch"])
        subprocess.run(["mv", "cache/huggingface", "/root/.cache/huggingface"])

        self.config = "configs/generate.yaml"
        vae_ckpt = None
        ckpt = "checkpoints/instruct-pix2pix-00-22000.ckpt"

        config = OmegaConf.load(self.config)
        self.model = load_model_from_config(config, ckpt, vae_ckpt)
        self.model.eval().cuda()
        self.model_wrap = K.external.CompVisDenoiser(self.model)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)
        self.null_token = self.model.get_learned_conditioning([""])


    def predict(
            self,
            input_image: Path = Input(description="Path to an image"),
            instruction_text: str = Input(description="Instruction text"),
            seed: int = Input(default=0, description="Random sampling seed. Sometimes, some seeds will edit the image more than others"),
            cfg_text: float = Input(default=7.5, description="Higher value leads to more drastic edits but less variety"),
            cfg_image: float = Input(default=1.5, description="Higher value means more preservation of the input image - but less drastic edits"),
            resolution: int = Input(default=512, description="Output resolution: sometimes, certain resolutions make certain types of changes better"),
    ) -> Path:
        input_image = str(input_image)
        seed = int(seed)
        cfg_text = float(cfg_text)
        cfg_image = float(cfg_image)
        resolution = int(resolution)
        output = 'output.jpg'
        steps = 100

        input_image = Image.open(input_image).convert("RGB")
        width, height = input_image.size
        factor = resolution / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

        with torch.no_grad(), autocast("cuda"), self.model.ema_scope():
            cond = {}
            cond["c_crossattn"] = [self.model.get_learned_conditioning([str(instruction_text)])]
            input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
            input_image = rearrange(input_image, "h w c -> 1 c h w").to(self.model.device)
            cond["c_concat"] = [self.model.encode_first_stage(input_image).mode()]

            uncond = {}
            uncond["c_crossattn"] = [self.null_token]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

            sigmas = self.model_wrap.get_sigmas(steps)

            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": cfg_text,
                "image_cfg_scale": cfg_image,
            }
            torch.manual_seed(seed)
            z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            z = K.sampling.sample_euler_ancestral(self.model_wrap_cfg, z, sigmas, extra_args=extra_args)
            x = self.model.decode_first_stage(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255.0 * rearrange(x, "1 c h w -> h w c")
            edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
        edited_image.save(output)

        return Path(output)

