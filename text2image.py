import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16", use_auth_token=True)
pipe = pipe.to(device)

prompt = "hr giger Two leopards on horseback seen from afar riding in the snow, snowy landscape, snow storm, fantasy, highly detailed, digital painting, artstation, concept art, illustration, art by Greg Rutkowski and Marc Simonettiet, cold colours, sharp textures, biotechnology, nikolay georgiev, tim hildebrandt, bruce pennington, donato giancola, larry elmore, masterpiece, trending on artstation, featured on pixiv, cinematic composition, sharp, details, hyper - detailed, hd, hdr, 4 k, 8 k"
with autocast("cuda"):
    image = pipe(prompt, guidance_scale=7.5).images[0]  
    
image.save("astronaut_rides_horse.png")

