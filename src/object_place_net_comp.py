import os

import torch
import torch.nn.functional as F
from torch import nn

from config import opt
from resnet_4ch import resnet

from PIL import Image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from object_place_dataset_plus import get_test_dataloader

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
    def __init__(self, backbone_pretrained=True, device = torch.device('cuda:0')):
        super(ObjectPlaceNetFBComposite, self).__init__()
        
        self.device = device

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

        self.prediction_head = nn.Linear(self.global_feature_dim*2, 5, bias=False)

        self.classify = ObjectPlaceNet(backbone_pretrained=False)

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

        theta = torch.tensor(0) #transform_params[:,0]
        centerx = torch.tensor(.5) #transform_params[:,1]
        centery = torch.tensor(.5) #transform_params[:,2]
        scale = torch.tensor(.5) #transform_params[:,3]

        img_warp = model.affine_warp(img, theta, centerx, centery, scale)

        img_comp = model.composite(img_warp, background)

        prediction = self.classify(img_comp)

        return prediction, img_comp

    # rotation 0 to 1, centerx, centery 0 to 1, scale 0 to 1
    def affine_warp(self, img_cat, theta, centerx, centery, scale):
        
        theta = theta*2*torch.pi

        centerx = centerx*2.-1.
        centery = centery*2.-1.      

        transx = -centerx
        transy = -centery

        rot = torch.tensor((torch.cos(theta),-torch.sin(theta),0,torch.sin(theta),torch.cos(theta),0,0,0,1)).reshape(3,3)
        trans = torch.tensor((1,0,transx,0,1,transy,0,0,1)).reshape(3,3)
        scale = torch.tensor((1./scale,0,0,0,1./scale,0,0,0,1)).reshape(3,3)

        mat = torch.matmul(rot,torch.matmul(scale,trans))
        mat = mat[0:2,:]
       
        grid = torch.nn.functional.affine_grid(mat.unsqueeze(0), img_cat.shape).to(self.device)
        img_cat_warp = torch.nn.functional.grid_sample(img_cat, grid)

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

if __name__ == '__main__':
    with torch.no_grad():
        device = torch.device('cuda:0')
    
        image_size = 256

        img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        img = Image.open("/mnt/e/Research/Images/new_OPA/foreground/airplane/157394.jpg")
        img = img_transform(img).unsqueeze(0).to(device)
    
        mask = Image.open("/mnt/e/Research/Images/new_OPA/foreground/airplane/mask_157394.jpg").convert('L') 
        mask = img_transform(mask).unsqueeze(0).to(device)  

        background = Image.open("/mnt/e/Research/Images/new_OPA/background/airplane/15016.jpg")
        background = img_transform(background).unsqueeze(0).to(device)  

        img = torch.cat([img, mask], dim=1)
    
        model = ObjectPlaceNetFBComposite(backbone_pretrained=False, device=device).to(device)

        checkpoint_path='experiments/ablation_study/resnet18_repeat3/checkpoints/best-acc.pth'
        print('load pretrained weights from ', checkpoint_path)
        model.classify.load_state_dict(torch.load(checkpoint_path, map_location=device))

        #theta = torch.tensor(.5)
        #centerx = torch.tensor(.75)
        #centery = torch.tensor(.75)
        #scale = torch.tensor(0.5)

        #img_warp = model.affine_warp(img, theta, centerx, centery, scale)
        #img_comp = model.composite(img_warp, background)

        prediction, img_comp = model(img, background)

        prob = F.softmax(prediction)

        print(prob)

        plt.figure()
        plt.imshow(img_comp[0,0:3,:,:].cpu().permute(1, 2, 0))
        plt.show()

#def F1(preds, gts):
#    tp = sum(list(map(lambda a, b: a == 1 and b == 1, preds, gts)))
#    fp = sum(list(map(lambda a, b: a == 1 and b == 0, preds, gts)))
#    fn = sum(list(map(lambda a, b: a == 0 and b == 1, preds, gts)))
#    tn = sum(list(map(lambda a, b: a == 0 and b == 0, preds, gts)))
#    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
#    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
#    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
#    bal_acc = (tpr + tnr) / 2
#    return f1, bal_acc


#def evaluate_model(device, checkpoint_path='./best-acc.pth'):
#    opt.without_mask = False
#    assert os.path.exists(checkpoint_path), checkpoint_path
#    net = ObjectPlaceNet(backbone_pretrained=False)
#    print('load pretrained weights from ', checkpoint_path)
#    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
#    net = net.to(device).eval()

#    total = 0
#    pred_labels = []
#    gts = []

#    test_loader = get_test_dataloader()

#    with torch.no_grad():
#        for batch_index, (img_cat, label, target_box) in enumerate(tqdm(test_loader)):
#            img_cat, label, target_box = img_cat.to(
#                device), label.to(device), target_box.to(device)

#            logits = net(img_cat)

#            pred_labels.extend(logits.max(1)[1].cpu().numpy())
#            gts.extend(label.cpu().numpy())
#            total += label.size(0)

#    total_f1, total_bal_acc = F1(pred_labels, gts)
#    print("Baseline model evaluate on {} images, local:f1={:.4f},bal_acc={:.4f}".format(
#        total, total_f1, total_bal_acc))

#    return total_f1, total_bal_acc


#if __name__ == '__main__':
#    device = "cuda:0"
#    f1, balanced_acc = evaluate_model(device, checkpoint_path='experiments/ablation_study/resnet18_repeat3/checkpoints/best-acc.pth')

