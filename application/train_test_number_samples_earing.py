from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

data_dir = './data'

transform = transforms.Compose([
    transforms.Resize(178),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

male_idx = 20      # 'Male'
smiling_idx = 31   # 'smiling' (0-based index in CelebA attrs)

def count_gender_smiling(split):
    dataset = datasets.CelebA(root=data_dir,
                              split=split,
                              transform=transform,
                              download=False)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    fm = fn = mm = mn = 0  # female_smile, female_nosmile, male_smile, male_nosmile

    for _, attrs in loader:
        gender = attrs[:, male_idx]      # 0 = female, 1 = male
        smiling = attrs[:, smiling_idx]  # 0 = not smiling, 1 = smiling

        female_mask = (gender == 0)
        male_mask   = (gender == 1)
        smile_mask  = (smiling == 1)
        nosmile_mask = (smiling == 0)

        fm += (female_mask & smile_mask).sum().item()
        fn += (female_mask & nosmile_mask).sum().item()
        mm += (male_mask & smile_mask).sum().item()
        mn += (male_mask & nosmile_mask).sum().item()

    print(f"=== {split.upper()} SPLIT ===")
    print(f"Smiling Females:     {fm}")
    print(f"Non-Smiling Females: {fn}")
    print(f"Smiling Males:       {mm}")
    print(f"Non-Smiling Males:   {mn}")

# Call for train and test
count_gender_smiling('train')
count_gender_smiling('test')
