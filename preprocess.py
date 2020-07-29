from pathlib import Path
import SimpleITK as sitk
import numpy as np
import progressbar
import re

dict_path = {
    'BRATS2015': '/home/jianjunming/data/open_datasets/BRATS2015/training/LGG',
    'BRATS2018': '/home/jianjunming/data/open_datasets/BraTS2018/MICCAI_BraTS_2018_Data_Training/LGG'
}


class PreProc():
    def __init__(self, dataset):
        self.dataset = dataset
        self.path_data_raw = Path(dict_path[dataset])

    def preproc(self):
        t1, t2, t1c, flair, mask = [], [], [], [], []

        if self.dataset == 'BRATS15':
            # only 54 cases LGG in training cohort was used
            bar = progressbar.ProgressBar(max_value=54)
            for ind, path_pat in enumerate(self.path_data_raw.iterdir()):
                for path_mod in path_pat.iterdir():
                    array = sitk.GetArrayFromImage(sitk.ReadImage(str(path_mod)))
                    if path_mod.name.startswith('VSD.Brain.XX.O.MR_T1.'):
                        t1.append(array)
                    elif path_mod.name.startswith('VSD.Brain.XX.O.MR_T2.'):
                        t2.append(array)
                    elif path_mod.name.startswith('VSD.Brain.XX.O.MR_T1c.'):
                        t1c.append(array)
                    elif path_mod.name.startswith('VSD.Brain.XX.O.MR_Flair.'):
                        flair.append(array)
                    elif re.match('VSD.Brain_\dmore', path_mod.name):
                        mask.append(array)
                    else:
                        raise Exception('Unknown file ', path_mod)
                bar.update(ind+1)
                # print(ind)
            print('\t')
            print('Saving files... ')
            if not Path('data/'+ self.dataset).exists():
                Path('data/' + self.dataset).mkdir()
            np.savez('data/' + self.dataset + '/T1.npz', t1)
            np.savez('data/' + self.dataset + '/T2.npz', t2)
            np.savez('data/' + self.dataset + '/T1c.npz', t1c)
            np.savez('data/' + self.dataset + '/Flair.npz', flair)
            np.savez('data/' + self.dataset + '/Mask.npz', mask)

        elif self.dataset == 'BRATS18':
            # only 75 cases LGG in training cohort was used
            bar = progressbar.ProgressBar(max_value=75)
            ind = 0

            for path_pat in self.path_data_raw.iterdir():
                for path_mod in path_pat.iterdir():
                    array = sitk.GetArrayFromImage(sitk.ReadImage(str(path_mod)))
                    if path_mod.name.endswith('t1.nii.gz'):
                        t1.append(array)
                    elif path_mod.name.endswith('t2.nii.gz'):
                        t2.append(array)
                    elif path_mod.name.endswith('t1ce.nii.gz'):
                        t1c.append(array)
                    elif path_mod.name.endswith('flair.nii.gz'):
                        flair.append(array)
                    elif path_mod.name.endswith('seg.nii.gz'):
                        mask.append(array)
                    else:
                        raise Exception('Unknown file ', path_mod)
                ind += 1
                bar.update(ind)
            print('\t')
            print('Saving files... ')
            if not Path('data/'+ self.dataset).exists():
                Path('data/' + self.dataset).mkdir()
            np.savez('data/' + self.dataset + '/T1.npz', t1)
            np.savez('data/' + self.dataset + '/T2.npz', t2)
            np.savez('data/' + self.dataset + '/T1c.npz', t1c)
            np.savez('data/' + self.dataset + '/Flair.npz', flair)
            np.savez('data/' + self.dataset + '/Mask.npz', mask)

        else:
            raise Exception('Unknown dataset ', self.dataset)





if __name__ == '__main__':
    data = PreProc(dataset='BRATS18')
    data.preproc()

