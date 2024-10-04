import torch
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt


# Funzione per creare coppie di immagini e etichettarli con "maggiore", "minore" o "uguale"
def create_image_pairs(dataset):
    image_pairs = []
    labels = []

    for i in range(len(dataset)):
        img_A, label_A = dataset[i]

        # Seleziona un'altra immagine casuale
        idx_B = random.randint(0, len(dataset) - 1)
        img_B, label_B = dataset[idx_B]
        # Determina la relazione tra le etichette
        if label_A > label_B:
            relation_label = 0  # A > B
        elif label_A < label_B:
            relation_label = 1  # A < B
        else:
            relation_label = 2  # A = B

        # Aggiungi la coppia e la relazione
        image_pairs.append((img_A, img_B))
        labels.append(relation_label)

    return image_pairs, labels

# Dataset personalizzato per gestire le coppie di immagini
class MNISTPairDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.image_pairs, self.labels = create_image_pairs(dataset)

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img_A, img_B = self.image_pairs[idx]
        label = self.labels[idx]

        # Concatenare le immagini lungo il canale (depth)
        concatenated_image = torch.cat((img_A, img_B), dim=0)

        return concatenated_image, label

# Funzione per visualizzare una coppia di immagini e la loro etichetta di relazione
def show_image_pair(img_A, img_B, relation_label):
    relation_labels = {0: 'A > B', 1: 'A < B', 2: 'A = B'}

    # Converte il tensore in numpy array per visualizzazione
    img_A = img_A.squeeze().numpy()  # Rimuove la dimensione del canale
    img_B = img_B.squeeze().numpy()

    # Crea una figura con due immagini
    fig, axes = plt.subplots(1, 2, figsize=(5, 5))
    axes[0].imshow(img_A, cmap='gray')
    axes[0].set_title('Image A')
    axes[1].imshow(img_B, cmap='gray')
    axes[1].set_title('Image B')

    # Mostra il titolo della relazione
    plt.suptitle(f'Relation: {relation_labels[relation_label]}')
    plt.show()

def show_dataset(dataset, num_images=5):
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    for i in range(5):
        concatenated_image = images[i]
        label = labels[i].item()

        img_A = concatenated_image[0, :, :]
        img_B = concatenated_image[1, :, :]

        show_image_pair(img_A, img_B, label)

if __name__ == '__main__':
    # Trasformo le immagini in tensori e normalizzo i valori secondo le statiche del dataset MNIST
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Applico data augmentation sul dataset di training
    transform_augmented = transforms.Compose([
        transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_augmented)

    train_pair_dataset = MNISTPairDataset(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_pair_dataset, batch_size=16, shuffle=True)

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    test_pair_dataset = MNISTPairDataset(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_pair_dataset, batch_size=16, shuffle=False)
    
    print("Size train: ", len(train_loader), " Size test: ", len(test_loader))

    see_img = True

    if see_img:
        show_dataset(train_loader)
    

