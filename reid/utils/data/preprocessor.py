from __future__ import absolute_import
import os.path as osp
from torch.utils.data import Dataset
from PIL import Image


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid, index
    
    def get_indexes(self, indexes):
        all_data = [self._get_single_item(index) for index in indexes]
        images, pids = [val[0] for val in all_data], [val[2] for val in all_data]
        return images, pids
    
class ModernPreprocessor(Dataset):
    """
    一个现代化的、清晰的Preprocessor，继承自torch.utils.data.Dataset。
    """
    def __init__(self, dataset, root=None, transform=None):
        super(ModernPreprocessor, self).__init__()
        self.dataset = dataset      # 这是一个列表，每个元素是 (图像路径, pid, camid, ...)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        PyTorch DataLoader会调用这个方法来获取单个样本。
        """
        # 1. 从数据列表中获取信息
        #    我们假设列表中的元组至少有3个元素
        data_tuple = self.dataset[index]
        img_path = data_tuple[0]
        pid = data_tuple[1]
        
        # 兼容处理3元组和4元组的情况
        camid = data_tuple[2] # 在多域模式下，这其实是domain_id
        
        # 2. 构建完整图像路径
        fpath = img_path
        if self.root is not None:
            fpath = osp.join(self.root, img_path)
        
        # 3. 读取并转换图像
        try:
            img = Image.open(fpath).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {fpath}")
            # 返回一个占位符或抛出异常
            # 这里我们简单地返回一个黑色图片，并打印错误
            img = Image.new('RGB', (128, 256), color = 'black')
            print(e)
            
        # 4. 应用图像变换 (包括 ToTensor() 和 Normalize())
        if self.transform is not None:
            img = self.transform(img)
            
        # 5. 返回下游代码期望的所有信息
        #    根据你evaluators.py中的解包方式 `images, fnames, pids, _, camids = data`
        #    我们需要返回一个5元组
        return img, img_path, pid, index, camid

