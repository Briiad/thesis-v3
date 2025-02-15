import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator

def get_retinanet_model(num_classes, pretrained_backbone=True):
    """
    Constructs a RetinaNet detection model using an EfficientNet-B0 backbone with FPN.
    
    Args:
        num_classes (int): Number of classes (including background).
        pretrained_backbone (bool): Whether to use a backbone pre-trained on ImageNet.
        
    Returns:
        model (RetinaNet): The detection model.
    """
    # Load EfficientNet-B0 from torchvision
    efficientnet = torchvision.models.efficientnet_b0(pretrained=pretrained_backbone)
    
    # Use the features (an nn.Sequential) as the backbone.
    backbone = efficientnet.features
    
    # For an nn.Sequential backbone, the keys are string indices.
    # Choose indices "3", "5", and "7" for feature extraction.
    return_layers = {
        "3": "feat0",
        "5": "feat1",
        "7": "feat2"
    }
    
    # Define the expected number of channels from each selected feature map.
    in_channels_list = [40, 112, 320]
    out_channels = 256  # FPN output channels
    
    # Wrap the backbone with FPN.
    # Note: By default, BackboneWithFPN uses LastLevelMaxPool as extra_blocks,
    # which adds an extra feature map. So the total number of outputs is 4.
    backbone_with_fpn = BackboneWithFPN(
        backbone,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels
    )
    
    # Adjust the anchor generator to have four tuples (one per feature map).
    # Here we define progressively larger anchors for each FPN level,
    # with very small anchors at the finest level for detecting small acne lesions.
    anchor_generator = AnchorGenerator(
        sizes=((4, 8), (8, 16), (16, 32), (32, 64)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4
    )
    
    # Instantiate the RetinaNet model.
    model = RetinaNet(
        backbone_with_fpn,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        score_thresh=0.01,
        nms_thresh=0.4,    
        bg_iou_thresh=0.2,
        fg_iou_thresh=0.2,     
        detections_per_img=100,
        min_size=640,           
        max_size=640, 
    )
    
    return model

if __name__ == '__main__':
    num_classes = 7
    model = get_retinanet_model(num_classes)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    model.eval()
    image = torch.randn(1, 3, 640, 640)
    output = model(image)
    print("Output shapes:", {k: v.shape for k, v in output[0].items()})
