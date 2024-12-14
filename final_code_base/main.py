from model import DynamicImageDataset

dataset = DynamicImageDataset("./dataset")
print(dataset[0][0].shape)
