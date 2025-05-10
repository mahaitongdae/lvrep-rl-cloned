import yaml
import repr_control
import os
import numpy as np
import imageio
pkg_dir = repr_control.__path__[0]
from repr_control.envs.tractor_trailer_render import Renderer
task = 'parking'
id = 4

if __name__ == "__main__":
    # Load the YAML file
    with open(f"{pkg_dir}/config/map{id}.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Print the loaded configuration
    print(config)
    
    # Create the renderer
    renderer = Renderer(save_video=False,
                        render_mode='rgb_array')
    # set constraints
    renderer.set_obstacles(np.array(config['obstacles']))
    low_state = config['tasks'][task]['init']['min'] + [0.0, 0.0, 0.0]
    renderer.set_state(np.array(low_state))
    fig = renderer.render()
    imageio.imwrite(f"{pkg_dir}/config/demo/map{id}_{task}.png", fig)