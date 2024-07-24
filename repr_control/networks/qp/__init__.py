import yaml
import os
current_path = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_path, 'qp_default.yaml'), 'r') as f:
    qp_default_args = yaml.safe_load(f)