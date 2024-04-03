import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms as T
from detr.util.misc import NestedTensor
import math
import os
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion
from detr.util.box_ops import box_cxcywh_to_xyxy
from torch import optim
from detr.models.matcher import build_matcher, HungarianMatcher
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import nms
import time
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 37 

def unnormalize_bbox(bbox, img_width, img_height):
    """
    Convert a normalized bbox [cx, cy, w, h] to pixel coordinates [xmin, ymin, xmax, ymax].

    Parameters:
    - bbox: The bounding box in [cx, cy, w, h] format, where cx and cy are the center coordinates
            of the box normalized by the width and height, and w and h are the width and height
            of the box, normalized by the image width and height.
    - img_width: The width of the image in pixels.
    - img_height: The height of the image in pixels.

    Returns:
    - A list [xmin, ymin, xmax, ymax] representing the bounding box in pixel coordinates.
    """
    cx, cy, w, h = bbox
    cx, w = cx * img_width, w * img_width
    cy, h = cy * img_height, h * img_height
    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = xmin + w
    ymax = ymin + h

    return [xmin, ymin, xmax, ymax]

def plot_image_bbox(image, bboxes, labels, class_names):
    """
    Plot an image with bounding boxes and class names.
    """
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for bbox, label in zip(bboxes, labels):
        xmin, ymin, xmax, ymax = bbox
        class_id = int(label)  
        class_name = class_names.get(class_id, "Unknown")  

        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(xmin, ymin, class_name, color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    plt.show()

def load_class_list(class_list_path):
    class_names = {}
    with open(class_list_path, 'r') as file:
        for line in file:
            class_id, _, class_name = line.strip().split('-')
            class_id = int(class_id) 
            class_names[class_id] = class_name
    return class_names

def convert_pred_boxes(pred_boxes, img_size):

    width, height = img_size
    pred_boxes = box_cxcywh_to_xyxy(pred_boxes)  
    pred_boxes = torch.stack([
        pred_boxes[:, 0] * width,
        pred_boxes[:, 1] * height,
        pred_boxes[:, 2] * width,
        pred_boxes[:, 3] * height,
    ], dim=-1)
    return pred_boxes

def visualize_predictions(img, gt_boxes, gt_labels, pred_boxes, pred_labels, class_names, confidences=None):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(img)
    ax[0].set_title('Ground Truth')
    ax[1].imshow(img)
    ax[1].set_title('Predictions')

    def ensure_cpu_numpy(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy()
        return obj

    # Ground Truth
    for box, label in zip(gt_boxes, gt_labels):
        box = [ensure_cpu_numpy(coord) for coord in box]
        label = label.item() if isinstance(label, torch.Tensor) else label
        class_name = class_names.get(int(label), "Unknown")
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='g', facecolor='none')
        ax[0].add_patch(rect)
        ax[0].text(box[0], box[1], class_name, color='white', fontsize=12, bbox=dict(facecolor='green', alpha=0.5))

    # Predictions
    for box, label, confidence in zip(pred_boxes, pred_labels, confidences):
        box = [ensure_cpu_numpy(coord) for coord in box]  # Convert each tensor within the list
        label = label.item() if isinstance(label, torch.Tensor) else label
        class_name = class_names.get(label, "Unknown")
        # Include confidence score in the annotation
        annotation = f"{class_name}: {confidence:.2f}"
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
        ax[1].add_patch(rect)
        ax[1].text(box[0], box[1], annotation, color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    plt.show()

def custom_collate_fn(batch):
    images, targets, img_names = zip(*batch)  
    images = torch.stack(images, dim=0)

    return images, targets, img_names

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos

class DinoV2SpatialBackbone(nn.Module):
    def __init__(self, dinov2_model):
        super().__init__()
        self.patch_embed = dinov2_model.patch_embed
        self.pos_drop = dinov2_model.pos_drop  
        self.blocks = dinov2_model.blocks

    def forward(self, x):
        #print(f"Input shape: {x.shape}")

        patch_embeddings = self.patch_embed(x)
        #print(f"Patch embeddings shape: {patch_embeddings.shape}")

        for i, block in enumerate(self.blocks):
            patch_embeddings = block(patch_embeddings)
            #print(f"After block {i} shape: {patch_embeddings.shape}")

        batch_size, seq_length, embedding_dim = patch_embeddings.shape
        num_patches = seq_length 
        side_length = int((num_patches) ** 0.5) 
        spatial_features = patch_embeddings.transpose(1, 2).view(batch_size, embedding_dim, side_length, side_length)
        #print(f"Spatial features shape: {spatial_features.shape}")

        flattened_features = spatial_features.flatten(2)  
        #print(f"Flattened spatial features before transpose: {flattened_features.shape}")
        flattened_features = flattened_features.transpose(1, 2) 
        #print(f"Flattened spatial features after transpose: {flattened_features.shape}")

        return flattened_features

class DinoV2WithPosEncoding(nn.Module):
    def __init__(self, dinov2_backbone, pos_embedding):
        super().__init__()
        self.dinov2_backbone = dinov2_backbone
        self.pos_embedding = pos_embedding

    def forward(self, inputs):
        features = self.dinov2_backbone(inputs) 

        batch_size, seq_length, feature_dim = features.shape
        side_length = int(seq_length ** 0.5)  
        spatial_features = features.view(batch_size, side_length, side_length, feature_dim).permute(0, 3, 1, 2) 

        dummy_mask = torch.zeros((batch_size, side_length, side_length), dtype=torch.bool, device=features.device)

        nested_tensor = NestedTensor(spatial_features, dummy_mask)

        pos_encodings = self.pos_embedding(nested_tensor)  

        combined_features = (spatial_features + pos_encodings).flatten(2).permute(0, 2, 1)  
        difference = combined_features - features

        has_changes = torch.any(difference != 0)

        
        #print(f"Are the tensors different after adding positional encoding? {'Yes' if has_changes.item() else 'No'}")

        return combined_features

class CustomDETR(nn.Module):
    def __init__(self, dinov2_pos_encoding, detr_model, num_classes=num_classes):
        super(CustomDETR, self).__init__()
        self.dinov2_pos_encoding = dinov2_pos_encoding
        self.detr_model = detr_model

        # self.adapter_conv = nn.Sequential(
        #     nn.Conv2d(384, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),

        #     nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1))
        # ).to(device)

        self.adapter_conv = nn.Sequential(
            nn.Conv2d(768, 1024, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(0.1), 
            nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Dropout(0.1)  
        ).to(device)

        self.detr_model.class_embed = nn.Linear(
            in_features=self.detr_model.class_embed.in_features, 
            out_features=num_classes + 1
        )

    def forward(self, inputs):
        #print(f"Input shape: {inputs.shape}")
        #print(f"Input batch size: {inputs.size(0)}") 
        features = self.dinov2_pos_encoding(inputs)
        #print(f"Features from DinoV2WithPosEncoding shape: {features.shape}")
        
        batch_size, seq_length, feature_dim = features.shape
        H, W = int(seq_length ** 0.5), int(seq_length ** 0.5) 
        features = features.permute(0, 2, 1).view(batch_size, feature_dim, H, W)
        #print(f"Reshaped features shape for DETR input_proj: {features.shape}")
        features = self.adapter_conv(features)
        #print(f"Features after adapter_conv shape: {features.shape}")

        features_proj = self.detr_model.input_proj(features)
        #print(f"Features after input_proj shape: {features_proj.shape}")

        features_flat = features_proj.flatten(2).permute(2, 0, 1)
        #print(f"Flattened features for encoder shape: {features_flat.shape}")

        encoder_output = self.detr_model.transformer.encoder(features_flat)
        #print(f"Encoder output shape: {encoder_output.shape}")
        
        object_queries = self.detr_model.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        #print(f"Object queries shape: {object_queries.shape}")

        decoder_output = self.detr_model.transformer.decoder(object_queries, encoder_output)
        #print(f"Decoder output shape: {decoder_output.shape}")

        if decoder_output.dim() == 4: 
            all_logits = []
            all_bbox_outputs = []
            for layer_output in decoder_output:
                layer_logits = self.detr_model.class_embed(layer_output)
                layer_bbox_output = self.detr_model.bbox_embed(layer_output).sigmoid()
                all_logits.append(layer_logits)
                all_bbox_outputs.append(layer_bbox_output)

            logits = torch.mean(torch.stack(all_logits), dim=0)
            bbox_output = torch.mean(torch.stack(all_bbox_outputs), dim=0)
        else:
            logits = self.detr_model.class_embed(decoder_output)
            bbox_output = self.detr_model.bbox_embed(decoder_output).sigmoid()

        return {'pred_logits': logits, 'pred_boxes': bbox_output}

class CustomDataset(Dataset):
    def __init__(self, root_dir, dataset_type='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        if dataset_type == 'train':
            self.image_folder = os.path.join(root_dir, 'multi/images/train')
            self.label_folder = os.path.join(root_dir, 'multi/labels/train')
        elif dataset_type == 'valid':
            self.image_folder = os.path.join(root_dir, 'multi/images/valid')
            self.label_folder = os.path.join(root_dir, 'multi/labels/valid')
        elif dataset_type == 'test':
            self.image_folder = os.path.join(root_dir, 'test/images')
            self.label_folder = os.path.join(root_dir, 'test/labels/YOLO_darknet')
        else:
            raise ValueError("Invalid dataset_type. Use 'train', 'valid', or 'test'.")

        self.images = os.listdir(self.image_folder)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_name = os.path.join(self.image_folder, self.images[idx])
        label_name = os.path.join(self.label_folder, self.images[idx].replace('.jpeg', '.txt'))

        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        labels = []
        try:
            with open(label_name, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, width, height = [float(x) for x in line.strip().split()]
                    labels.append([class_id, x_center, y_center, width, height])
        except FileNotFoundError:
            print(f"Warning: Label file not found or unreadable: {label_name}. Using default.")
            labels = [[-1, 0, 0, 0, 0]]

        labels = torch.tensor(labels, dtype=torch.float32)

        target = {}
        if labels.nelement() == 0:  
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
        else:
            target['boxes'] = labels[:, 1:]
            target['labels'] = labels[:, 0].long()

        return image, target, img_name

def unfreeze_dino_backbone_layers(model, n_layers):
    """
    Unfreezes the last n_layers of the DINO backbone in the given model.
    """
    layers = list(model.dinov2_pos_encoding.dinov2_backbone.blocks.children())[::-1] 
    for layer in layers[:n_layers]:
        for param in layer.parameters():
            param.requires_grad = True

def print_top_class_probabilities(pred_logits, top_n=5, class_names=None):
    pred_probs = F.softmax(pred_logits, dim=-1)
    top_probs, top_lbls = torch.topk(pred_probs, top_n, dim=-1)
    for i in range(top_probs.size(0)):
        print(f"Top {top_n} class probabilities for image {i+1}:")
        for j in range(top_n):
            class_id = top_lbls[i, j].item()
            class_prob = top_probs[i, j].item()
            class_name = class_names[class_id] if class_names and class_id in class_names else str(class_id)
            print(f"{class_name}: {class_prob:.4f}")
        print("")

class_list_path = 'test/class_list.txt'
dinov2_vits14 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16').to(device)
detr_model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True).to(device)

class_names = load_class_list(class_list_path)

dinov2_backbone = DinoV2SpatialBackbone(dinov2_vits14)

pos_embedding = PositionEmbeddingSine(num_pos_feats=384, 
                                      normalize=True) 

model_with_pos_encoding = DinoV2WithPosEncoding(dinov2_backbone, 
                                                pos_embedding)

custom_detr = CustomDETR(model_with_pos_encoding, 
                         detr_model).to(device)

for param in custom_detr.dinov2_pos_encoding.parameters():
    param.requires_grad = False

custom_detr.to(device)

transforms = transforms.Compose([
    transforms.Resize((490,490)),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = CustomDataset(root_dir='', 
                              dataset_type='train', 
                              transform=transforms)

valid_dataset = CustomDataset(root_dir='', 
                              dataset_type='valid', 
                              transform=transforms)

train_loader = DataLoader(train_dataset,
                           batch_size=14,
                            shuffle=True,
                            num_workers=2,
                            collate_fn=custom_collate_fn)

valid_loader = DataLoader(valid_dataset, 
                          batch_size=14, 
                          shuffle=True, 
                          num_workers=2, 
                          collate_fn=custom_collate_fn)

matcher = HungarianMatcher(cost_class=1, 
                           cost_bbox=1, 
                           cost_giou=1)

criterion = SetCriterion(num_classes=num_classes,  
                         matcher=matcher, 
                         weight_dict = {'loss_ce': 1, 'loss_bbox': 1, 'loss_giou': 1, 'cardinality_error_focal': 2},  
                         eos_coef=0.1,  
                         losses=['labels', 'boxes', 'cardinality'])  
criterion.to(device)

num_epochs = 100

def debug_train_one_batch(model, criterion, data_loader, device, num_epochs, unfreeze_epoch=20):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    for epoch in range(num_epochs):
        train_loss = 0.0
        total_train_loss = 0.0
        total_loss_ce = 0.0
        total_loss_bbox = 0.0
        total_loss_giou = 0.0
        total_loss_cardinality = 0.0

        if epoch == unfreeze_epoch:
            unfreeze_dino_backbone_layers(model, 1)
            newly_unfrozen_params = [p for p in model.dinov2_pos_encoding.dinov2_backbone.parameters() if p.requires_grad]
            base_params = [p for p in model.parameters() if not p.requires_grad or all(p.data_ptr() != q.data_ptr() for q in newly_unfrozen_params)]
            optimizer = optim.Adam([
                {'params': base_params},  
                {'params': newly_unfrozen_params, 'lr': 1e-4}  
            ], lr=1e-3)      
        
        criterion.train()

        for batch_idx, (images, targets,img_names) in enumerate(data_loader):
            images = images.to(device)

            # print(f"Image batch shape: {images.shape}")  # [batch_size, C, H, W]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]   

            # print(f"\nBatch {batch_idx+1}, Image Names: {img_names}")

            # for i, target in enumerate(targets):
            #     print(f"\nBatch {batch_idx+1}, Target {i+1} type: {type(target)}")
            #     if isinstance(target, dict):
            #         print(f"Target {i+1} keys: {list(target.keys())}")
            #         if 'labels' in target:
            #             print(f"Target {i+1} labels shape: {target['labels'].shape}")  # [num_objects]
            #             print(f"Target {i+1} first few labels: {target['labels'][:5]}")
            #         if 'boxes' in target:
            #             print(f"Target {i+1} boxes shape: {target['boxes'].shape}")  # [num_objects, 4]
            #             print(f"Target {i+1} first few boxes: {target['boxes'][:5]}")


            outputs = model(images)

            outputs["pred_logits"] = outputs["pred_logits"].permute(1, 0, 2)  
            outputs["pred_boxes"] = outputs["pred_boxes"].permute(1, 0, 2)  

            # print(f"Output keys: {list(outputs.keys())}")
            # if 'pred_logits' in outputs:
            #      print(f"Predicted logits shape: {outputs['pred_logits'].shape}")  # Expected: [batch_size, num_queries, num_classes]
            #      print(f"First predicted logits: {outputs['pred_logits'][0, :5]}")  # Print the first 5 logits for the first item in the batch

            # if 'pred_boxes' in outputs:
            #      print(f"Predicted boxes shape: {outputs['pred_boxes'].shape}")  # Expected: [batch_size, num_queries, 4]
            
            if (epoch + 1) % 50 == 0:  
                    for i in range(len(images)):
                        pred_logits_i = outputs['pred_logits'][i].detach()
                        print_top_class_probabilities(pred_logits_i, top_n=5, class_names=class_names)
  
                    pred_logits_i = outputs['pred_logits'][i].detach()
                    pred_boxes_i = outputs['pred_boxes'][i].detach()
                    
                    pred_probs_i = F.softmax(pred_logits_i, dim=-1)
                    confidences_i, pred_labels_i = pred_probs_i.max(-1)
                    
                    NO_OBJECT_CLASS_INDEX = pred_logits_i.size(1) - 1
                    CONF_THRESHOLD = 0.02 

                    valid_indices = (pred_labels_i != NO_OBJECT_CLASS_INDEX) & (confidences_i > CONF_THRESHOLD)

                    valid_pred_boxes_i = pred_boxes_i[valid_indices]
                    valid_pred_labels_i = pred_labels_i[valid_indices]
                    valid_confidences_i = confidences_i[valid_indices]

                    gt_boxes_i = [unnormalize_bbox(box, img_width=images[i].shape[-1], img_height=images[i].shape[-2]) for box in targets[i]['boxes']]
                    gt_labels_i = targets[i]['labels']
                    img_np = images[i].permute(1, 2, 0).cpu().numpy()
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_np = np.clip((img_np * std + mean) * 255.0, 0, 255).astype(np.uint8)

                    visualize_predictions(
                        img=img_np,
                        gt_boxes=gt_boxes_i,
                        gt_labels=gt_labels_i,
                        pred_boxes=valid_pred_boxes_i,
                        pred_labels=valid_pred_labels_i,
                        class_names=class_names,
                        confidences=valid_confidences_i
        )
            loss_dict = criterion(outputs, targets)
            weights = criterion.weight_dict
            total_loss = sum(loss_dict[k] * weights[k] for k in loss_dict.keys() if k in weights)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_train_loss += total_loss.item()
            total_loss_ce += loss_dict['loss_ce'].item()
            total_loss_bbox += loss_dict['loss_bbox'].item()
            total_loss_giou += loss_dict['loss_giou'].item()
            total_loss_cardinality += loss_dict['cardinality_error_focal'].item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_loss_ce = total_loss_ce / len(train_loader)
        avg_loss_bbox = total_loss_bbox / len(train_loader)
        avg_loss_giou = total_loss_giou / len(train_loader)
        avg_loss_cardinality = total_loss_cardinality / len(train_loader)

        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Average Cross-Entropy Loss: {avg_loss_ce:.4f}")
        print(f"Average Bounding Box Loss: {avg_loss_bbox:.4f}")
        print(f"Average GIoU Loss: {avg_loss_giou:.4f}")
        print(f"Average Cardinality Loss: {avg_loss_cardinality:.4f}")

        model.eval()
        total_valid_loss = 0.0
        valid_loss_ce = 0.0
        valid_loss_bbox = 0.0
        valid_loss_giou = 0.0
        valid_loss_cardinality = 0.0

        with torch.no_grad():
            for batch_idx, (images, targets, img_names) in enumerate(valid_loader):
                images = images.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                outputs = model(images)
                outputs["pred_logits"] = outputs["pred_logits"].permute(1, 0, 2)
                outputs["pred_boxes"] = outputs["pred_boxes"].permute(1, 0, 2)
                
                loss_dict = criterion(outputs, targets)
                
                total_loss = sum(loss_dict[k] * weights[k] for k in loss_dict.keys() if k in weights)
                
                total_valid_loss += total_loss.item()
                valid_loss_ce += loss_dict['loss_ce'].item()
                valid_loss_bbox += loss_dict['loss_bbox'].item()
                valid_loss_giou += loss_dict['loss_giou'].item()
                valid_loss_cardinality += loss_dict['cardinality_error_focal'].item()

        avg_valid_loss = total_valid_loss / len(valid_loader)
        avg_valid_loss_ce = valid_loss_ce / len(valid_loader)
        avg_valid_loss_bbox = valid_loss_bbox / len(valid_loader)
        avg_valid_loss_giou = valid_loss_giou / len(valid_loader)
        avg_valid_loss_cardinality = valid_loss_cardinality / len(valid_loader)
        print(f"Epoch: {epoch+1}")
        print(f"Training Loss: {avg_train_loss:.4f} (CE: {avg_loss_ce:.4f}, BBox: {avg_loss_bbox:.4f}, GIoU: {avg_loss_giou:.4f}, Cardinality: {avg_loss_cardinality:.4f})") # 
        print(f"Validation Loss: {avg_valid_loss:.4f} (CE: {avg_valid_loss_ce:.4f}, BBox: {avg_valid_loss_bbox:.4f}, GIoU: {avg_valid_loss_giou:.4f}, Cardinality: {avg_valid_loss_cardinality:.4f})")#
        print("\n" + "="*50 + "\n")        

debug_train_one_batch(custom_detr, criterion, train_loader, device, num_epochs, unfreeze_epoch=10)       

