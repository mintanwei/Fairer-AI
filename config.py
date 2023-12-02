from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform_train = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize([512, 648]),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomVerticalFlip(p=0.5),
                                      transforms.ColorJitter(brightness=(1, 1.3)),
                                      transforms.ToTensor(),
                                      normalize
                                      ])

transform_val = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize([512, 648]),
                                    transforms.ToTensor(),
                                    normalize
                                    ])

OculoScope_dir = "/home/user4/workplace/data/OculoScope"
MixNAF_dir = "/home/user4/workplace/data/MixNAF"
