import torch
import torch.optim as optim

import myUtils
from edge_detector.model_torch import res_skip
from myUtils import psnr_by_list, ssim_by_list, draw_by_list
from utils.io import load_ckpt
from utils.io import save_ckpt
from torchvision.utils import make_grid
from torchvision.utils import save_image
from modules.RFRNet import RFRNet, VGG16FeatureExtractor
import os
import time


class RFRNetModel():
    def __init__(self, config=None):
        self.G = None
        self.lossNet = None
        self.iter = None
        self.optm_G = None
        self.device = None
        self.real_A = None
        self.real_B = None
        self.fake_B = None
        self.comp_B = None
        self.l1_loss_val = 0.0

        self.edge_model = self.get_edge_model().cuda()
        self.config = config
    
    def initialize_model(self, path=None, train=True):
        self.G = RFRNet()
        self.optm_G = optim.Adam(self.G.parameters(), lr = 2e-4)
        if train:
            self.lossNet = VGG16FeatureExtractor()
        try:
            start_iter = load_ckpt(path, [('generator', self.G)], [('optimizer_G', self.optm_G)])
            if train:
                self.optm_G = optim.Adam(self.G.parameters(), lr = 2e-4)
                print('Model Initialized, iter: ', start_iter)
                self.iter = start_iter
        except:
            print('No trained model, from start')
            self.iter = 0
        
    def cuda(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Model moved to cuda")
            self.G.cuda()
            if self.lossNet is not None:
                self.lossNet.cuda()
        else:
            self.device = torch.device("cpu")
        
    def train(self, train_loader, save_path, finetune = False, iters=450000, val_loader=None,val_from_train_loader=None,config=None):
        epoch = 0
    #    writer = SummaryWriter(log_dir="log_info")
        self.G.train(finetune = finetune)
        if finetune:
            self.optm_G = optim.Adam(filter(lambda p:p.requires_grad, self.G.parameters()), lr = 5e-5)
        print("Starting training from iteration:{:d}".format(self.iter))
        s_time = time.time()
        psnr_list = []
        ssim_list = []
        psnr_path = '{:s}/psnr_img.png'.format(save_path)
        ssim_path = '{:s}/ssim_img.png'.format(save_path)
        while self.iter<iters:
            self.G.train()
            for i, (imgs, structures, masks, labels, tags) in enumerate(train_loader):
                gt_images, masks = self.__cuda__(imgs, masks)
                # 設置cuda

                # gt_images, structures, masks, labels, tags = self.__cuda__(*items)
                masked_images = gt_images * masks
                self.forward(masked_images, masks, gt_images)
                self.update_parameters()
                self.iter += 1
                epoch += 1
                if self.iter % 50 == 0:
                    e_time = time.time()
                    int_time = e_time - s_time
                    print("Iteration:%d, l1_loss:%.4f, time_taken:%.2f" %(self.iter, self.l1_loss_val/50, int_time))
                    s_time = time.time()
                    self.l1_loss_val = 0.0
                    # 計算剩餘時間，單位為小時
                    remain_time = (iters - self.iter) * int_time / 3600
                    print("Remaining time:%.2f hours" %remain_time)

                if self.iter % 20000 == 0:
                    if not os.path.exists('{:s}'.format(self.config.model_path)):
                        os.makedirs('{:s}'.format(self.config.model_path))
                    save_ckpt('{:s}/g_{:d}.pth'.format(self.config.model_path, self.iter ), [('generator', self.G)], [('optimizer_G', self.optm_G)], self.iter)

            if epoch % 4 != 0:
                continue
            # 測試
            print("Testing...")
            self.G.eval()
            result_save_path = save_path
            if not os.path.exists('{:s}'.format(result_save_path)):
                os.makedirs('{:s}'.format(result_save_path))
            count = 0
            avg_psnr = 0
            avg_ssim = 0
            self.G.eval()
            with torch.no_grad():
                for i, (imgs, structures, masks, labels, tags) in enumerate(val_loader):
                    gt_images, masks = self.__cuda__(imgs, masks)
                    masked_images = gt_images * masks
                    masks = torch.cat([masks] * 3, dim=1)
                    fake_B, mask = self.G(masked_images, masks)
                    comp_B = fake_B * (1 - masks) + gt_images * masks
                    if not os.path.exists(self.config.val_img_save_path_compare):
                        os.makedirs(self.config.val_img_save_path_compare)


                    imgs = gt_images.detach().cpu().numpy()
                    comp_imgs = comp_B.detach().cpu().numpy()
                    # 計算psnr,ssim
                    this_psnr = psnr_by_list(imgs, comp_imgs)
                    this_ssim = ssim_by_list(imgs, comp_imgs)

                    avg_psnr += this_psnr
                    avg_ssim += this_ssim

                    val_grid = self.get_grid(imgs, structures, masks, masked_images, comp_imgs)

                    save_image(val_grid, '{:s}/{:d}.png'.format(self.config.val_img_save_path_compare, count))

                avg_psnr /= len(val_loader)
                avg_ssim /= len(val_loader)
                print("Iteration:%d, avg_PSNR:%.4f, avg_SSIM:%.4f" %(self.iter, avg_psnr, avg_ssim))
                psnr_list.append(avg_psnr)
                ssim_list.append(avg_ssim)
                draw_by_list(psnr_list, "PSNR", self.config.psnr_img_save_path,show_max=True)
                draw_by_list(ssim_list, "SSIM", self.config.ssim_img_save_path,show_max=True)
                print("Testing done")

                for i, (imgs, structures, masks, labels, tags) in enumerate(val_from_train_loader):
                    gt_images, masks = self.__cuda__(imgs, masks)
                    masked_images = gt_images * masks
                    masks = torch.cat([masks] * 3, dim=1)
                    fake_B, mask = self.G(masked_images, masks)
                    comp_B = fake_B * (1 - masks) + gt_images * masks
                    if not os.path.exists(self.config.val_from_train_img_save_path_compare):
                        os.makedirs(self.config.val_from_train_img_save_path_compare)

                    imgs = gt_images.detach().cpu().numpy()
                    comp_imgs = comp_B.detach().cpu().numpy()

                    val_grid = self.get_grid(imgs, structures, masks, masked_images, comp_imgs)

                    save_image(val_grid, '{:s}/{:d}.png'.format(self.config.val_from_train_img_save_path_compare, count))





        if not os.path.exists('{:s}'.format(self.config.model_path)):
            os.makedirs('{:s}'.format(self.config.model_path))
            save_ckpt('{:s}/g_{:s}.pth'.format(self.config.model_path, "final"), [('generator', self.G)], [('optimizer_G', self.optm_G)], self.iter)

    def get_grid(self, imgs, structures, masks, img_masked, comp_imgs):

        comp_imgs_structures = self.get_edge(comp_imgs)

        # 都转成rgb格式
        imgs_rgb = myUtils.gray2rgb(imgs)
        structures_rgb = myUtils.gray2rgb(structures)
        masks_rgb = myUtils.gray2rgb(masks)
        img_masked_rgb = myUtils.gray2rgb(img_masked)
        comp_imgs_rgb = myUtils.gray2rgb(comp_imgs)
        comp_imgs_structures_rgb = myUtils.gray2rgb(comp_imgs_structures, mode='RED')
        mask_red = myUtils.gray2rgb(masks, mode='RED')
        # 从【0,1】放缩到【-1,1】
        mask_red = (mask_red - 0.5) / 0.5

        # 在img的mask区域填充为红色
        img_masked_red = torch.where(masks.byte() == False, mask_red, imgs)  # 将 mask 区域的像素值设为红色 (1, 0, 0)

        # 拼接structures和comp_imgs_structures的mask区域
        comp_imgs_structures_rgb_x = comp_imgs_structures_rgb * (1 - masks_rgb) + structures_rgb * masks_rgb

        grid_list = [imgs_rgb, structures_rgb, masks_rgb, img_masked_red, comp_imgs_rgb, comp_imgs_structures_rgb_x]

        return myUtils.make_val_grid_list(grid_list)

    def get_edge(self, img):
        """
        获取边缘
        :param img: 图片
        :return: 边缘
        """
        # with torch.no_grad():

        # 将-1到1的图片放缩到0-255
        img = (img + 1) * 127.5

        edge = self.edge_model(img)

        # 截取255-0
        edge = torch.clamp(edge, 0, 255)

        # 放缩到-1至1
        edge = (edge - 127.5) / 127.5

        return edge

    def get_edge_model(self):
        """
        获取边缘检测
        :return: 模型
        """
        # 获取边缘检测
        edge_detect = res_skip()

        edge_detect.load_state_dict(torch.load(self.config.edge_model_path))

        myUtils.set_requires_grad(edge_detect, False)

        edge_detect.cuda()
        edge_detect.eval()

        return edge_detect

    def test(self, test_loader, result_save_path):
        self.G.eval()
        # for para in self.G.parameters():
        #     para.requires_grad = False
        if not os.path.exists('{:s}'.format(result_save_path)):
            os.makedirs('{:s}'.format(result_save_path))
        count = 0
        with torch.no_grad():
            for items in test_loader:
                gt_images, masks = self.__cuda__(*items)
                masked_images = gt_images * masks
                masks = torch.cat([masks]*3, dim = 1)
                fake_B, mask = self.G(masked_images, masks)
                comp_B = fake_B * (1 - masks) + gt_images * masks
                if not os.path.exists('{:s}/results'.format(result_save_path)):
                    os.makedirs('{:s}/results'.format(result_save_path))
                for k in range(comp_B.size(0)):
                    count += 1
                    grid = make_grid(comp_B[k:k+1])
                    file_path = '{:s}/results/img_{:d}.png'.format(result_save_path, count)
                    save_image(grid, file_path)

                    grid = make_grid(masked_images[k:k+1] +1 - masks[k:k+1] )
                    file_path = '{:s}/results/masked_img_{:d}.png'.format(result_save_path, count)
                    save_image(grid, file_path)
    
    def forward(self, masked_image, mask, gt_image):
        self.real_A = masked_image
        self.real_B = gt_image
        self.mask = mask
        fake_B, _ = self.G(masked_image, mask)
        self.fake_B = fake_B
        self.comp_B = self.fake_B * (1 - mask) + self.real_B * mask
    
    def update_parameters(self):
        self.update_G()
        self.update_D()
    
    def update_G(self):
        self.optm_G.zero_grad()
        loss_G = self.get_g_loss()
        loss_G.backward()
        self.optm_G.step()
    
    def update_D(self):
        return
    
    def get_g_loss(self):
        real_B = self.real_B
        fake_B = self.fake_B
        comp_B = self.comp_B
        
        real_B_feats = self.lossNet(real_B)
        fake_B_feats = self.lossNet(fake_B)
        comp_B_feats = self.lossNet(comp_B)
        
        tv_loss = self.TV_loss(comp_B * (1 - self.mask))
        style_loss = self.style_loss(real_B_feats, fake_B_feats) + self.style_loss(real_B_feats, comp_B_feats)
        preceptual_loss = self.preceptual_loss(real_B_feats, fake_B_feats) + self.preceptual_loss(real_B_feats, comp_B_feats)
        valid_loss = self.l1_loss(real_B, fake_B, self.mask)
        hole_loss = self.l1_loss(real_B, fake_B, (1 - self.mask))
        
        loss_G = (  tv_loss * 0.1
                  + style_loss * 120
                  + preceptual_loss * 0.05
                  + valid_loss * 1
                  + hole_loss * 6)
        
        self.l1_loss_val += valid_loss.detach() + hole_loss.detach()
        return loss_G
    
    def l1_loss(self, f1, f2, mask = 1):
        return torch.mean(torch.abs(f1 - f2)*mask)
    
    def style_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            _, c, w, h = A_feat.size()
            A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
            B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
            A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
            B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
            loss_value += torch.mean(torch.abs(A_style - B_style)/(c * w * h))
        return loss_value
    
    def TV_loss(self, x):
        h_x = x.size(2)
        w_x = x.size(3)
        h_tv = torch.mean(torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]))
        w_tv = torch.mean(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]))
        return h_tv + w_tv
    
    def preceptual_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            loss_value += torch.mean(torch.abs(A_feat - B_feat))
        return loss_value
            
    def __cuda__(self, *args):
        return (item.to(self.device) for item in args)
            