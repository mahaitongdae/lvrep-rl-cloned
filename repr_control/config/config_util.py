import yaml


def load_config_from_file(fname):
    
    # load setup from yaml file
    with open(fname, "r") as file:
        config = yaml.safe_load(file)
        
    task_config = config['tasks']
    obstacles = config['obstacles']
    trailer_config = config['trailer']
    
    return task_config, obstacles, trailer_config, config