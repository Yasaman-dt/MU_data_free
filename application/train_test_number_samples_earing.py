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
earing_idx = 34   # 'earing' (0-based index in CelebA attrs)

def count_gender_earing(split):
    dataset = datasets.CelebA(root=data_dir,
                              split=split,
                              transform=transform,
                              download=False)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    fm = fn = mm = mn = 0  # female_smile, female_nosmile, male_smile, male_nosmile

    for _, attrs in loader:
        gender = attrs[:, male_idx]      # 0 = female, 1 = male
        earing = attrs[:, earing_idx]  # 0 = not earing, 1 = earing

        female_mask = (gender == 0)
        male_mask   = (gender == 1)
        smile_mask  = (earing == 1)
        nosmile_mask = (earing == 0)

        fm += (female_mask & smile_mask).sum().item()
        fn += (female_mask & nosmile_mask).sum().item()
        mm += (male_mask & smile_mask).sum().item()
        mn += (male_mask & nosmile_mask).sum().item()

    print(f"=== {split.upper()} SPLIT ===")
    print(f"Males earing:       {mm}")
    print(f"Females earing:     {fm}")
    print(f"Males Non-earing:   {mn}")
    print(f"Females Non-earing: {fn}")

# Call for train and test
count_gender_earing('train')
count_gender_earing('test')
