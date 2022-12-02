import os

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as Ft
from torch import nn

from config import opt
from resnet_4ch import resnet

from PIL import Image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from object_place_dataset_plus import get_test_dataloader

from object_place_net import ObjectPlaceNet

from tqdm import tqdm

# classify if the background is a match to the foreground
class ObjectPlaceNetFBMatch(nn.Module):
    def __init__(self, backbone_pretrained=True):
        super(ObjectPlaceNetFBMatch, self).__init__()
        
        ## Backbone, only resnet
        resnet_layers = int(opt.backbone.split('resnet')[-1])

        backbone = resnet(resnet_layers,
                          backbone_pretrained,
                          os.path.join(opt.pretrained_model_path, opt.backbone+'.pth'))

        backbone_back = resnet(resnet_layers,
                          backbone_pretrained,
                          os.path.join(opt.pretrained_model_path, opt.backbone+'.pth'), without_mask=True)

        # drop pool layer and fc layer
        features = list(backbone.children())[:-2]
        backbone = nn.Sequential(*features)
        self.backbone = backbone

        features = list(backbone_back.children())[:-2]
        backbone_back = nn.Sequential(*features)
        self.backbone_back = backbone_back

        ## global predict
        self.global_feature_dim = 512 if opt.backbone in ['resnet18', 'resnet34'] else 2048
   
        self.avgpool3x3 = nn.AdaptiveAvgPool2d(3)
        self.avgpool1x1 = nn.AdaptiveAvgPool2d(1)

        self.prediction_head = nn.Linear(self.global_feature_dim*2, opt.class_num, bias=False)

    def forward(self, img_cat, img_back):
        '''  img_cat:b,4,256,256  '''
        global_feature = None
        if opt.without_mask:
            img_cat = img_cat[:,0:3] 
        feature_map = self.backbone(img_cat)  # b,512,8,8 (resnet layer4 output shape: b,c,8,8, if resnet18, c=512)
        feature_map_back = self.backbone_back(img_back)  # b,512,8,8 (resnet layer4 output shape: b,c,8,8, if resnet18, c=512)
        global_feature = self.avgpool1x1(feature_map)  # b,512,1,1
        global_feature = global_feature.flatten(1) # b,512

        global_feature_back = self.avgpool1x1(feature_map_back)  # b,512,1,1
        global_feature_back = global_feature_back.flatten(1) # b,512

        global_feature = torch.cat((global_feature, global_feature_back), 1)

        prediction = self.prediction_head(global_feature) 

        return prediction

# classify if the background is a match to the foreground
class ObjectPlaceNetFBComposite(nn.Module):
    def __init__(self, backbone_pretrained=True, device = torch.device('cuda:0'), include_rotation=False):
        super(ObjectPlaceNetFBComposite, self).__init__()
        
        self.device = device
        self.include_rotation = include_rotation

        ## Backbone, only resnet
        resnet_layers = int(opt.backbone.split('resnet')[-1])

        backbone = resnet(resnet_layers,
                          backbone_pretrained,
                          os.path.join(opt.pretrained_model_path, opt.backbone+'.pth'))

        backbone_back = resnet(resnet_layers,
                          backbone_pretrained,
                          os.path.join(opt.pretrained_model_path, opt.backbone+'.pth'), without_mask=True)

        # drop pool layer and fc layer
        features = list(backbone.children())[:-2]
        backbone = nn.Sequential(*features)
        self.backbone = backbone

        features = list(backbone_back.children())[:-2]
        backbone_back = nn.Sequential(*features)
        self.backbone_back = backbone_back

        ## global predict
        self.global_feature_dim = 512 if opt.backbone in ['resnet18', 'resnet34'] else 2048
   
        self.avgpool3x3 = nn.AdaptiveAvgPool2d(3)
        self.avgpool1x1 = nn.AdaptiveAvgPool2d(1)

        self.prediction_head = nn.Linear(self.global_feature_dim*2, 5 if self.include_rotation else 4, bias=False)
        
        #self.classify = ObjectPlaceNet(backbone_pretrained=False)

    def forward(self, img_cat, img_back):

        '''  img_cat:b,4,256,256  '''
        global_feature = None
        if opt.without_mask:
            img_cat = img_cat[:,0:3] 
        feature_map = self.backbone(img_cat)  # b,512,8,8 (resnet layer4 output shape: b,c,8,8, if resnet18, c=512)
        feature_map_back = self.backbone_back(img_back)  # b,512,8,8 (resnet layer4 output shape: b,c,8,8, if resnet18, c=512)
        global_feature = self.avgpool1x1(feature_map)  # b,512,1,1
        global_feature = global_feature.flatten(1) # b,512

        global_feature_back = self.avgpool1x1(feature_map_back)  # b,512,1,1
        global_feature_back = global_feature_back.flatten(1) # b,512

        global_feature = torch.cat((global_feature, global_feature_back), 1)

        transform_params = torch.sigmoid(self.prediction_head(global_feature))

        #print(transform_params.shape)
        #print(transform_params)

        if (self.include_rotation):            
            theta = transform_params[:,0]
            centerx = transform_params[:,1]
            centery = transform_params[:,2]
            scale = transform_params[:,3]
        else:
            theta = None
            centerx = transform_params[:,0]
            centery = transform_params[:,1]
            scale = transform_params[:,2]

        img_warp = self.affine_warp(img_cat, theta, centerx, centery, scale)
        img_comp = self.composite(img_warp, img_back)

        #prediction = self.classify(img_comp)

        return img_comp

    # rotation 0 to 1, centerx, centery 0 to 1, scale 0 to inf
    def affine_warp(self, img_cat, theta, centerx, centery, scale):
        
        if (theta is not None):
            theta = theta*2*torch.pi

        centerx = centerx*2.-1.
        centery = centery*2.-1.      

        transx = -centerx
        transy = -centery

        if (theta and theta.shape[0] == 1):
            R = torch.tensor((torch.cos(theta),-torch.sin(theta),0,torch.sin(theta),torch.cos(theta),0,0,0,1)).reshape(3,3).float()

        Ms = []

        for b in range(img_cat.shape[0]):

            if (theta is not None and theta.shape[0] == img_cat.shape[0]):
                R = torch.tensor((torch.cos(theta[b]),-torch.sin(theta[b]),0,torch.sin(theta[b]),torch.cos(theta[b]),0,0,0,1)).reshape(3,3).float()

            T = torch.tensor((1,0,transx[b],0,1,transy[b],0,0,1)).reshape(3,3).float()
            S = torch.tensor((1./scale[b],0,0,0,1./scale[b],0,0,0,1)).reshape(3,3).float()

            if (theta is not None):
                M = torch.matmul(R,torch.matmul(S,T))
            else:
                M = torch.matmul(S,T)

            M = M[0:2,:]
       
            Ms.append(M)

        Ms = torch.stack(Ms, 0)

        grid = torch.nn.functional.affine_grid(Ms, img_cat.shape, align_corners=True).to(self.device)
        img_cat_warp = torch.nn.functional.grid_sample(img_cat, grid, align_corners=True)

        return img_cat_warp

    # rotation 0 to 2pi, centerx, centery 0 to 1, scale 0 to 1
    def composite(self, img_cat, background):

        img = img_cat[:,0:3,:,:]
        mask = img_cat[:,3:4,:,:]
        mask_rgb = mask.repeat((1,3,1,1))
                
        blur = transforms.GaussianBlur(kernel_size=(5, 5), sigma=0.5)
        mask_rgb = blur(mask_rgb)

        img_comp = img*mask_rgb + (1-mask_rgb)*background

        img_comp = torch.cat([img_comp, mask], dim=1)

        return img_comp

def F1(preds, gts):
    tp = sum(list(map(lambda a, b: a == 1 and b == 1, preds, gts)))
    fp = sum(list(map(lambda a, b: a == 1 and b == 0, preds, gts)))
    fn = sum(list(map(lambda a, b: a == 0 and b == 1, preds, gts)))
    tn = sum(list(map(lambda a, b: a == 0 and b == 0, preds, gts)))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    bal_acc = (tpr + tnr) / 2
    return f1, bal_acc

def evaluate_model(device, checkpoint_path='./best-acc.pth'):
    opt.without_mask = False
    assert os.path.exists(checkpoint_path), checkpoint_path
    net = ObjectPlaceNet(backbone_pretrained=False)
    print('load pretrained weights from ', checkpoint_path)
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net = net.to(device).eval()

    use_comp = True

    modelComp = ObjectPlaceNetFBComposite(backbone_pretrained=False, device=device).to(device).eval()

    total = 0
    pred_labels = []
    gts = []

    test_loader = get_test_dataloader()

    img_transform = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.ToTensor()
    ])

    with torch.no_grad():
        for batch_index, (img_cat, label, cx, cy, scale, foreground, background) in enumerate(tqdm(test_loader)):
            img_cat, label, cx, cy, scale, foreground, background = img_cat.to(device), label.to(device), cx.to(device), cy.to(device), scale.to(device), foreground.to(device), background.to(device)
                       
            if (use_comp):
                theta = torch.tensor(0.)
                foreground_warp = modelComp.affine_warp(foreground, theta, cx, cy, scale)
                img_comp = modelComp.composite(foreground_warp, background)           
            
                #img_comp = modelComp(foreground, background)        

                #for b in range(img_cat.shape[0]):
                #    plt.figure()
                #    plt.imshow(img_cat[b,0:3,:,:].cpu().permute(1, 2, 0))

                #    plt.figure()
                #    plt.imshow(img_comp[b,0:3,:,:].cpu().permute(1, 2, 0))

                #    plt.show()

                logits = net(img_comp)
            else:
                logits = net(img_cat)

            pred_labels.extend(logits.max(1)[1].cpu().numpy())
            gts.extend(label.cpu().numpy())
            total += label.size(0)

    total_f1, total_bal_acc = F1(pred_labels, gts)
    print("Baseline model evaluate on {} images, local:f1={:.4f},bal_acc={:.4f}".format(
        total, total_f1, total_bal_acc))

    return total_f1, total_bal_acc


if __name__ == '__main__':

    opt.batch_size = 16

    device = "cuda:0"
    f1, balanced_acc = evaluate_model(device, checkpoint_path='experiments/ablation_study/resnet18_repeat3/checkpoints/best-acc.pth')

