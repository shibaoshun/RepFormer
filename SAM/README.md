# Model Checkpoints
Three model versions of the model are available with different backbone sizes. These models can be instantiated by running
```
from segment_anything import sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
```

# Getting Started
First download the checkpoint for the corresponding model type.
Additionally, masks can be generated for images:
```
python SAMAUG.py
```
