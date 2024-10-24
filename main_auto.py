import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm
import time
import os
import json

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

    # Crea una figura con due immagini, riducendo la dimensione per occupare meno spazio
    fig, axes = plt.subplots(1, 2, figsize=(4, 4))  # Dimensioni ridotte

    # Mostra le immagini
    axes[0].imshow(img_A, cmap='gray')
    axes[0].set_title('Image A')
    axes[0].axis('off')  # Rimuove gli assi

    axes[1].imshow(img_B, cmap='gray')
    axes[1].set_title('Image B')
    axes[1].axis('off')  # Rimuove gli assi

    # Riduce lo spazio tra i subplots
    plt.tight_layout(pad=0.5)  # Riduce il padding tra le immagini

    # Mostra il titolo della relazione
    plt.suptitle(f'Relation: {relation_labels[relation_label]}', y=0.85)  # Posiziona il titolo più vicino
    plt.show()

# Funzione per visualizzare più coppie di immagini in un'unica figura
def show_dataset(dataset, num_images=5):
    dataiter = iter(dataset)
    images, labels = next(dataiter)

    # Mappa delle etichette di relazione
    relation_labels = {0: 'A > B', 1: 'A < B', 2: 'A = B'}

    # Numero di righe: ogni riga contiene 1 coppia di immagini (2 colonne)
    num_rows = num_images  # 1 coppia per riga

    # Crea una griglia di subplots: num_rows righe, 3 colonne (Image A, Relation, Image B)
    fig, axes = plt.subplots(num_rows, 3, figsize=(6, num_rows * 2))
    
    for i in range(num_images):
        concatenated_image = images[i]
        label = labels[i].item()

        img_A = concatenated_image[0, :, :]
        img_B = concatenated_image[1, :, :]

        # Mostra l'immagine A nel subplot
        axes[i, 0].imshow(img_A, cmap='gray')
        axes[i, 0].set_title('Image A')
        axes[i, 0].axis('off')

        # Mostra l'etichetta di relazione nel subplot centrale
        axes[i, 1].text(0.5, 0.5, f'Relation: {relation_labels[label]}', 
                        fontsize=12, ha='center', va='center')
        axes[i, 1].axis('off')

        # Mostra l'immagine B nel subplot
        axes[i, 2].imshow(img_B, cmap='gray')
        axes[i, 2].set_title('Image B')
        axes[i, 2].axis('off')  # Nascondi gli assi

    # Applica il layout compatto per ridurre lo spazio tra i subplot
    plt.tight_layout(pad=0.5)
    plt.show()

# Definisci la funzione per aggiungere rumore gaussiano
def add_gaussian_noise(tensor, mean=0.0, std=0.05):
    return tensor + std * torch.randn_like(tensor) + mean

# Definisci la funzione per invertire i colori
def invert_colors(tensor):
    return 1 - tensor

class CustomLeNet5(nn.Module):
    def __init__(self):
        super(CustomLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(2, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(train_data_loader, model):
    print('Training')
    train_itr = 0
    correct_train = 0
    total_train = 0
    running_loss = 0.0
    train_loss_list = []
    train_accuracy_list = []
    
    # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
    for i, data in enumerate(prog_bar):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        running_loss += loss_value

        train_loss_list.append(loss_value)
        train_accuracy_list.append(100 * correct_train / total_train)

        train_itr += 1
    
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return running_loss / len(train_data_loader), 100 * correct_train / total_train, train_loss_list, train_accuracy_list

def validate(valid_data_loader, model):
    print('Validating')
    val_itr = 0
    correct = 0
    total = 0
    running_loss = 0.0
    val_loss_list = []
    val_accuracy_list = []
    
    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    
    for i, data in enumerate(prog_bar):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        loss_value = loss.item()
        running_loss += loss_value
        val_loss_list.append(loss_value)
        val_accuracy_list.append(100 * correct / total)
        val_itr += 1
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return loss_value, 100 * correct / total, val_loss_list, val_accuracy_list

def show_incorrect_predictions(model, dataloader, num_images=5):
    model.eval()
    all_images = []
    all_labels = []
    all_preds = []

    # Itera su tutti i batch nel dataloader
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, 1)
            
            all_images.append(images.cpu())
            all_labels.append(labels.cpu())
            all_preds.append(predicted.cpu())

    # Concatena tutti i batch raccolti
    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    # Trova gli indici delle previsioni errate
    incorrect_predictions = (all_preds != all_labels).nonzero(as_tuple=True)[0]

    print(f"{len(incorrect_predictions)} previsioni errate trovate.")

    if len(incorrect_predictions) == 0:
        print("Nessuna previsione errata trovata.")
        return

    # Limita il numero di immagini errate da mostrare
    num_images = min(num_images, len(incorrect_predictions))

    fig, axes = plt.subplots(num_images, 3, figsize=(6, num_images * 2))
    relation_labels = {0: 'A > B', 1: 'A < B', 2: 'A = B'}
    
    for idx, i in enumerate(incorrect_predictions[:num_images]):
        img_A = all_images[i][0, :, :]
        img_B = all_images[i][1, :, :]
        pred_label = all_preds[i].item()
        true_label = all_labels[i].item()

        # Mostra l'immagine A
        axes[idx, 0].imshow(img_A, cmap='gray')
        axes[idx, 0].set_title('Image A')
        axes[idx, 0].axis('off')

        # Mostra l'etichetta di relazione prevista e reale
        axes[idx, 1].text(0.5, 0.5, f'Pred: {relation_labels[pred_label]}\nTrue: {relation_labels[true_label]}', 
                          fontsize=12, ha='center', va='center')
        axes[idx, 1].axis('off')

        # Mostra l'immagine B
        axes[idx, 2].imshow(img_B, cmap='gray')
        axes[idx, 2].set_title('Image B')
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.show()

# Funzione per calcolare e stampare la matrice di confusione normalizzata
def plot_confusion_matrix(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    # Disabilita i gradienti per velocizzare il calcolo
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Crea la matrice di confusione
    cm = confusion_matrix(all_labels, all_preds)
    
    # Normalizza la matrice di confusione
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalizza per ogni riga (classe vera)
    
    # Visualizza la matrice di confusione normalizzata in percentuale
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_normalized * 100, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=['A > B', 'A < B', 'A = B'], 
                yticklabels=['A > B', 'A < B', 'A = B'])
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (in %)')
    plt.show()

def classification_metrics(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    # Disabilita i gradienti per velocizzare il calcolo
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Genera il report di classificazione
    report = classification_report(all_labels, all_preds, target_names=['A > B', 'A < B', 'A = B'])
    print(report)

# Funzione per eseguire un esperimento con set specifici di iperparametri
def run_experiment(learning_rate, momentum, epochs, batch_size, optimizer):
    # Imposta l'ottimizzatore con gli iperparametri dinamici
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    total_train_loss = []
    total_train_acc = []
    total_val_loss = []
    total_val_acc = []

    # Ciclo di training e validazione
    for epoch in range(epochs):
        print(f"\nEPOCH {epoch+1} of {epochs} with LR={learning_rate} Momentum={momentum}")

        # Training e validazione
        train_loss, train_acc, train_loss_list, train_acc_list = train(train_loader, model)
        val_loss, val_acc, val_loss_list, val_acc_list = validate(validation_loader, model)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        total_train_loss.extend(train_loss_list)
        total_train_acc.extend(train_acc_list)
        total_val_loss.extend(val_loss_list)
        total_val_acc.extend(val_acc_list)

        print(f"Epoch #{epoch+1} train loss: {train_loss:.3f} train accuracy: {train_acc:.2f}")
        print(f"Epoch #{epoch+1} validation loss: {val_loss:.3f} validation accuracy: {val_acc:.2f}")
    
    _, test_acc, _, _ = validate(test_loader, model)
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    # Salva il modello con nome basato sugli iperparametri
    model_filename = f'model_lr_{learning_rate}_momentum_{momentum}_epochs_{epochs}.pth'
    torch.save(model.state_dict(), os.path.join('saved_models', model_filename))

    # Salva i risultati dell'esperimento
    experiment_log = {
        'learning_rate': learning_rate,
        'momentum': momentum,
        'epochs': epochs,
        'batch_size': batch_size,
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'final_validation_accuracy': test_acc
    }

    with open(f'experiment_log_lr_{learning_rate}_momentum_{momentum}_epochs_{epochs}.json', 'w') as f:
        json.dump(experiment_log, f)

    print(f'Model saved as {model_filename}')
    print(f'Experiment log saved as experiment_log_lr_{learning_rate}_momentum_{momentum}_epochs_{epochs}.json')



if __name__ == '__main__': 
    PRINT_IMG = True
    AUGMENT_DATASET = False
    TRANSFORM_DATASET = False

    # Ciclo per provare diversi set di iperparametri
    learning_rates = [0.001, 0.005, 0.01, 0.05]
    momentums = [0.85, 0.9, 0.95, 1]
    batch_sizes = [16, 32, 64, 128]
    epochs = 20 # Ridotto per velocizzare il testing, puoi aumentare

    # Trasformo le immagini in tensori e normalizzo i valori secondo le statiche del dataset MNIST
    transform_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    # Applico data augmentation sul dataset di training
    transform_augmented = transforms.Compose([
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.RandomChoice([
            transforms.Lambda(lambda x: add_gaussian_noise(x)),
            transforms.Lambda(lambda x: invert_colors(x)),
            transforms.Lambda(lambda x: x),
        ]),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if not TRANSFORM_DATASET and not AUGMENT_DATASET:
        train_dataset = datasets.MNIST(root='./data', train=True, download=True,
                                transform=transform_norm)

        train_pair_dataset = MNISTPairDataset(train_dataset)

        print(f"Dimensione del dataset originale: {len(train_pair_dataset)}")

    if TRANSFORM_DATASET:
        train_dataset_transformed = datasets.MNIST(root='./data', train=True, download=True,
                                                    transform=transform_augmented)
        
        train_pair_dataset = MNISTPairDataset(train_dataset_transformed)
        print(f"Dimensione del dataset trasformato: {len(train_pair_dataset)}")

    if AUGMENT_DATASET:
        train_dataset_augmented = datasets.MNIST(root='./data', train=True, download=True,
                                                    transform=transform_augmented)
        train_pair_dataset_augmented = MNISTPairDataset(train_dataset_augmented)
        train_pair_dataset = torch.utils.data.ConcatDataset([train_pair_dataset, train_pair_dataset_augmented])

        print(f"Dimensione del dataset aumentato: {len(train_pair_dataset_augmented)}")
        print(f"Dimensione del dataset combinato: {len(train_pair_dataset)}")

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_norm)

    test_pair_dataset = MNISTPairDataset(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_pair_dataset, batch_size=BATCH_SIZE, shuffle=False)

    total_test_size = len(test_dataset)

    # Percentuale di suddivisione per il validation set 70% train, 30% validation   
    validation_split = 0.3
    validation_size = int(total_test_size * validation_split)
    test_size = total_test_size - validation_size

    # Suddividi il dataset di test
    test_subset, validation_subset = torch.utils.data.random_split(test_pair_dataset, [test_size, validation_size])

    # Crea i DataLoader per il test set e il validation set
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(validation_subset, batch_size=BATCH_SIZE, shuffle=False)

    print("Size train_loader: ", len(train_loader), " Size train_dataset: ", len(train_pair_dataset))
    print("Size validation_loader: ", len(validation_loader), " Size validation_dataset: ", len(validation_subset))
    print("Size test_loader: ", len(test_loader), " Size test_dataset: ", len(test_subset))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = CustomLeNet5()
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    for lr in learning_rates:
        for momentum in momentums:
            for batch_size in batch_sizes:
                print(f"Starting experiment with LR={lr}, Momentum={momentum}, Batch Size={batch_size}")
                train_loader = torch.utils.data.DataLoader(
                                train_pair_dataset,
                                batch_size=batch_size,
                                shuffle=True)
                # Ricarica il modello prima di ogni nuovo esperimento
                model = CustomLeNet5().to(device)
                
                # Esegui l'esperimento con gli iperparametri specificati
                run_experiment(lr, momentum, epochs, batch_size)
    
    '''
    if PRINT_IMG:
        show_dataset(train_loader, 10)

    model.train()
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    total_val_loss = []
    total_val_acc = []
    total_train_loss = []
    total_train_acc = []
    for epoch in range(EPOCHS):
            print(f"\nEPOCH {epoch+1} of {EPOCHS}")

            # start timer and carry out training and validation
            start = time.time()
            train_loss, train_acc, train_loss_list, train_acc_list = train(train_loader, model)
            val_loss, val_acc, val_loss_list, val_acc_list = validate(test_loader, model)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            total_train_loss.extend(train_loss_list)
            total_train_acc.extend(train_acc_list)
            total_val_loss.extend(val_loss_list)
            total_val_acc.extend(val_acc_list)
            print(f"Epoch #{epoch+1} train loss: {train_loss:.3f} train accuracy: {train_acc:.2f}")   
            print(f"Epoch #{epoch+1} validation loss: {val_loss:.3f} validation accuracy: {val_acc:.2f}")   
            end = time.time()
            print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch+1}")

    # Visualizza alcune previsioni errate
    show_incorrect_predictions(model, validation_loader, num_images=5)

    # Calcola e visualizza la matrice di confusione
    plot_confusion_matrix(model, test_loader, device)

    # Stampa il report delle metriche
    classification_metrics(model, test_loader, device)

    # Plot della loss per epoca
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.show()

    # Plot dell'accuratezza per epoca
    plt.figure(figsize=(8, 6))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy per Epoch')
    plt.legend()
    plt.show()
    '''
