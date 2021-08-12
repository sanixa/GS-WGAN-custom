import os, sys
import torch
import torch.nn as nn
from models import *
from config import *

args = parse_arguments()
if not os.path.isfile(args.checkpoint):
    sys.exit("model does not exist")
print(f"converting {args.checkpoint}...")
model = GeneratorDCGAN(z_dim=args.z_dim, model_dim=args.model_dim, num_classes=10)
model = torch.load(args.checkpoint)

torch.save(model.state_dict(), args.checkpoint, _use_new_zipfile_serialization=False)
