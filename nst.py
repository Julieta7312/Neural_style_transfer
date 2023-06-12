from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import torchvision


    
device = torch.device( 'mps'  if torch.backends.mps.is_available() else \
                       'cuda' if torch.cuda.is_available() else 'cpu' )

img_pipe = transforms.Compose([ transforms.Resize(size=(512, 512) if device.type=='mps' or device.type=='cuda' else 128), \
                                transforms.ToTensor() ])
def load_img(img_dir):
    img = img_pipe(Image.open(img_dir)).unsqueeze(0) # batch 1 dim is required as an input. 
    return img.to(device, torch.float)

def load_img_from_np(np_img_arr):
    img = img_pipe(Image.fromarray(np_img_arr)).unsqueeze(0) # batch 1 dim is required as an input. 
    return img.to(device, torch.float)

unload_img = transforms.ToPILImage()  # Function to reconvert into PIL image

def imshow(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0) # create deepcopy of a tensor & remove the batch dimension
    image = unload_img(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

def gram(input):
    
    '''
        b = first dim of the input = batch size = 1
        c = number of feature maps (channels)
        (h,w) = dim of a feature f. maps (N = h*w)
    '''
    
    b, c, h, w = input.size()
    features = input.view( b*c, h*w ) # resize F_XL into \hat{F_XL}
    G = torch.mm(features, features.t()).div( b*c*h*w )  # gram product
    return G # 'normalize' the values of the gram matrix with the number of element in each feature maps.

class ContentLoss(nn.Module):
    
    '''
        'detach' the 'target' from the dynamic tree to the gradient. 
        Now, 'target' is static, so criterion.forward() won't throw an error.
    '''
    
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.gram_target = gram(target).detach()
        
    def forward(self, input):
        self.loss = F.mse_loss(gram(input), self.gram_target)
        return input

class Normalization(nn.Module):
    
    '''
        b = batch size, c = number of channels, h = height, w = width.    
        .view(-1,1,1) the mean and std to make them [c x 1 x 1] so that
        they can directly work with image Tensor of shape [b x c x h x w].
    '''
    
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

def check_layer(i, layer):
    if   isinstance(layer, nn.Conv2d):
        return i+1, f"conv_{i}", layer
    elif isinstance(layer, nn.ReLU):
        layer = nn.ReLU(inplace=False)
        return i, f"relu_{i}", layer
    elif isinstance(layer, nn.MaxPool2d):
        return i, f"pool_{i}", layer
    elif isinstance(layer, nn.BatchNorm2d):
        return i, f"bn_{i}", layer
    else:
        raise RuntimeError("Unrecognized layer: {layer.__class__.__name__}")
def get_style_model_and_losses(cnn, \
                               norm_mean, \
                               norm_std, \
                               style_img, \
                               content_img, \
                               content_layers=['conv_4'], \
                               style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    
    '''
        assuming that cnn is a nn.Sequential, make a new nn.Sequential
        to put in modules that are supposed to be activated sequentially.

        The nn.ReLU(inplace=True) doesn't play very nicely with the ContentLoss
        and StyleLoss we insert below. So we replace with out-of-place ones here.
    '''
    
    normalization = Normalization(norm_mean, norm_std).to(device)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)
    
    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        
        i, name, layer = check_layer(i, layer)
        model.add_module(name, layer)

        if name in content_layers:
            content_loss = ContentLoss( model(content_img).detach() )
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            style_loss = StyleLoss( model(style_img).detach() )
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)
    
    for i in range( len(model)-1, -1, -1 ):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    
    return model[:i+1], style_losses, content_losses # keep the layers until the last Content OR Style Loss.

model = None
def run_style_transfer(cnn, \
                       norm_mean, \
                       norm_std, \
                       content_img, \
                       style_img, input_img, \
                       num_steps=500, \
                       style_weight=1000000, \
                       content_weight=1, \
                       opt='LBFGS'):
    """Run the style transfer."""    
    global model
    
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     norm_mean, 
                                                                     norm_std, 
                                                                     style_img, 
                                                                     content_img)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    model.requires_grad_(False)
    
    # input is a .parameters() that requires a gradient
    optimizer = optim.LBFGS([input_img]) if opt=='LBFGS' else ( \
                optim.Adam([input_img], lr=0.1) if opt=='Adam' else \
                optim.SGD([input_img], lr=1) )
    

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 20 == 0:
                imshow(input_img, title=f'iter {run[0]}')
                print(f"run {run[0]}")
                print(f"StyleLoss: {style_score.item():.4f}, ContentLoss: {content_score.item():.4f} \n")

            return style_score + content_score

        optimizer.step(closure) # feature update!
    
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img



if __name__ == "__main__":

    import pandas as pd
    import numpy as np
    import os
    
    
    # Stylize YOLO detected object. Start

    DATASET_DIR = './to_yolo/dataset'
    TOTAL_RES_DIR = f'{DATASET_DIR}/preds/yolo_total_res.csv'
    OUTPUT_DIR = f'{DATASET_DIR}/output'
    STYLE_IMG_DIR_ROSA = "./sources/Rosa.jpg"
    STYLE_IMG_DIR_MIC = "./sources/mic.jpg"
    STYLE_IMG_DIR_GUI = "./sources/guitar.jpg"
    STYLE_IMG_DIR_HUM = "./sources/human.jpg"
    STYLE_IMG_DIR_ARM = "./sources/armchair.jpg"        

    cnn = models.vgg19(weights='DEFAULT').features.to(device).eval()
    cnn_norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    tot_res = pd.read_csv(TOTAL_RES_DIR, index_col=0)
    style_img = load_img(STYLE_IMG_DIR_ROSA)
    arng = lambda row, mi, mx: np.arange(round(row[mi]), round(row[mx])+1, 1)

    row_num = 0
    for idx, row in tot_res.iterrows():
        
        if not os.path.exists(f"{OUTPUT_DIR}/{row['name']}_{idx}_{row_num}.jpg"):
            xs, ys = np.meshgrid(arng(row, 'xmin', 'xmax'), arng(row, 'ymin', 'ymax'))
            img = np.array(Image.open(f'{DATASET_DIR}/images/{idx}.jpg'))
            
            content_img = load_img_from_np(img[ys, xs])
            # imshow(content_img, title='Content Image')
            
            load_params = STYLE_IMG_DIR_ROSA if row['name'] == 'Rosa' else (\
                          STYLE_IMG_DIR_MIC if  row['name'] == 'mic' else (\
                          STYLE_IMG_DIR_GUI if  row['name'] == 'guitar' else (\
                          STYLE_IMG_DIR_HUM if  row['name'] == 'human' else STYLE_IMG_DIR_ARM ) ) )
                    
            style_img = load_img( load_params )
            assert style_img.size() == content_img.size(), "style_img.size() == content_img.size() should be True"
            input_img = content_img.clone() # torch.randn(content_img.data.size(), device=device)
            
            output = run_style_transfer(cnn, \
                                        cnn_norm_mean, \
                                        cnn_norm_std, \
                                        content_img, \
                                        style_img, \
                                        input_img, \
                                        num_steps=300, \
                                        content_weight=25)
            
            # create deepcopy of a tensor & remove the fake batch dimension
            output_resized = unload_img( output.cpu().clone().squeeze(0) ).resize(ys.T.shape)
            img_w_output = img.copy()
            img_w_output[ys, xs] = np.array(output_resized)
            img_w_output = load_img_from_np(img_w_output)
            
            imshow(img_w_output, title='Image With Output')
            torchvision.utils.save_image(img_w_output.squeeze(0), fp=f"{OUTPUT_DIR}/{row['name']}_{idx}_{row_num}.jpg")
        
        row_num += 1
    # Stylize YOLO detected object. End


    # # Explore the extracted features. Start 

    # ''' Load Images '''
    # style_img = load_img("./sources/goldenTexture.jpg")
    # content_img = load_img("./to_yolo/dataset/798.jpg")
    # assert style_img.size() == content_img.size(), "style_img.size() == content_img.size() should be True"

    # plt.ion()
    # imshow(style_img, title='Style Image')
    # imshow(content_img, title='Content Image')

    # cnn = models.vgg19(weights='DEFAULT').features.to(device).eval()
    # cnn_norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    # cnn_norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # input_img = content_img.clone() # input_img = torch.randn(content_img.data.size(), device=device) # random noise init img
    # imshow(input_img, title='Input Image')
    # # for opt in ['LBFGS', 'Adam', 'SGD']:
    # for opt in ['SGD']:
    #     output = run_style_transfer(cnn, \
    #                                 cnn_norm_mean, \
    #                                 cnn_norm_std, \
    #                                 content_img, \
    #                                 style_img, \
    #                                 input_img, \
    #                                 num_steps=500, \
    #                                 content_weight=30, \
    #                                 opt=opt)
    #     imshow(output, title='Output Image') # plt.figure()
    #     torchvision.utils.save_image(output.squeeze(0), fp=f'./to_yolo/dataset/opt_test/result_{opt}.jpg')

    # # feature extraction
    # LOSS_DIR = './to_yolo/dataset/layers'
    # for mdl_param in ( (5, 2), (9, 3), (12, 4), (17, 5), (20, 6) ):
    #     for cs_img in [content_img, style_img]:
    #         style_loss = model[:mdl_param[0]](cs_img)
    #         for i, conv_img in enumerate(style_loss.squeeze(0)):
    #             add = '' if cs_img is content_img else 'texture'
    #             torchvision.utils.save_image(conv_img, fp=f"{LOSS_DIR}/style_loss_{mdl_param[1]}/style_{add}{mdl_param[1]}_{i}.jpg")
    
    # # Explore the extracted features. End
    
    
    # # plt.ioff()
    # # plt.show()

    