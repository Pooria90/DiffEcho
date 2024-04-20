import torch
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from random import sample
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import yaml


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train classifier")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration file",
        default="config_training.yaml",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to the data directory 4 classes",
        default="data/train",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        help="Path to the validation data that has 4 classes",
        default="data/valid",
    )
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.001)
    return parser.parse_args()


# Custom dataset class
class CustomDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, included_folders=None):
        super().__init__(root, transform)
        if included_folders is not None:
            self.samples = [
                s
                for s in self.samples
                if os.path.basename(os.path.dirname(s[0])) in included_folders
            ]
            self.samples = [
                (
                    s[0],
                    "ed" if "ed" in os.path.basename(os.path.dirname(s[0])) else "es",
                )
                for s in self.samples
            ]
            self.class_to_idx = {"ed": 0, "es": 1}
            self.samples = [(s[0], self.class_to_idx[s[1]]) for s in self.samples]
            self.imgs = self.samples


# Dataset preparation functions
def balance_and_select_samples(dataset, num_samples_per_class=800):
    # Track indices for each class
    indices_ed = []
    indices_es = []

    # Iterate over dataset to separate indices by class
    for idx, (_, label) in enumerate(dataset.samples):
        if label == dataset.class_to_idx["ed"]:  # Check if the label is 'ed'
            indices_ed.append(idx)
        elif label == dataset.class_to_idx["es"]:  # Check if the label is 'es'
            indices_es.append(idx)

    # Randomly select up to num_samples_per_class from each class
    selected_indices = sample(
        indices_ed, min(len(indices_ed), num_samples_per_class)
    ) + sample(indices_es, min(len(indices_es), num_samples_per_class))

    # Return a Subset of the original dataset based on the selected indices
    return Subset(dataset, selected_indices)


def load_and_prepare_datasets(
    data_dir, new_data_dir, val_dir, transform, included_folders
):
    dataset = CustomDataset(
        root=data_dir, transform=transform, included_folders=included_folders
    )
    if new_data_dir:
        new_dataset = CustomDataset(
            root=new_data_dir, transform=transform, included_folders=included_folders
        )
        balanced_new_dataset = balance_and_select_samples(new_dataset)
        dataset = ConcatDataset([dataset, balanced_new_dataset])
    val_dataset = CustomDataset(
        root=val_dir, transform=transform, included_folders=included_folders
    )
    return dataset, val_dataset


def adjust_trainable_parameters(model, target_trainable_params):
    """
    Adjust a model's layers to achieve a target number of trainable parameters.
    It freezes all parameters initially, then unfreezes from the end until the target is reached.

    Parameters:
    - model: The PyTorch model to adjust.
    - target_trainable_params: Target number of trainable parameters.
    """
    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    def get_trainable_params():
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def recursive_unfreeze(module):
        nonlocal target_trainable_params
        # Reverse iterate through the children of a module
        for child in reversed(list(module.children())):
            if get_trainable_params() >= target_trainable_params:
                return  # Early exit if target is reached

            # Recurse if child has further children
            if list(child.children()):
                recursive_unfreeze(child)
            if get_trainable_params() >= target_trainable_params:
                return  # Check target again after recursion

            # Unfreeze current child's parameters if it doesn't meet the target
            for param in child.parameters():
                param.requires_grad = True
                if get_trainable_params() >= target_trainable_params:
                    return  # Stop unfreezing once target is reached

    recursive_unfreeze(model)

    # Log the final count of trainable parameters
    final_trainable_params = get_trainable_params()
    print(
        f"Adjusted model to have approximately {final_trainable_params} trainable parameters."
    )


# Model preparation function
def initialize_model(model_name, num_classes=2):
    num_trainable_params = 2e6
    if model_name == "resnet18":
        if os.path.isfile("resnet18.pth"):
            model = models.resnet18()
            model.load_state_dict(torch.load("resnet18.pth"))
        else:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        torch.save(model.state_dict(), "resnet18.pth")
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnet50":
        if os.path.isfile("resnet50.pth"):
            model = models.resnet50()
            model.load_state_dict(torch.load("resnet50.pth"))
        else:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        torch.save(model.state_dict(), "resnet50.pth")
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "vgg16":
        if os.path.isfile("vgg16.pth"):
            model = models.vgg16()
            model.load_state_dict(torch.load("vgg16.pth"))
        else:
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        torch.save(model.state_dict(), "vgg16.pth")
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif model_name == "vgg19":
        if os.path.isfile("vgg19.pth"):
            model = models.vgg19()
            model.load_state_dict(torch.load("vgg19.pth"))
        else:
            model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        torch.save(model.state_dict(), "vgg19.pth")
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif model_name == "efficientnet-b0":
        if os.path.isfile("efficientnet-b0.pth"):
            model = models.efficientnet_b0()
            model.load_state_dict(torch.load("efficientnet-b0.pth"))
        else:
            model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT
            )
        torch.save(model.state_dict(), "efficientnet-b0.pth")
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    num_trainable_params_before = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(
        f"Number of trainable parameters before adjustment: {num_trainable_params_before}"
    )
    adjust_trainable_parameters(model, num_trainable_params)
    num_trainable_params_after = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(
        f"Number of trainable parameters after adjustment: {num_trainable_params_after}"
    )
    return model


# Training and evaluation functions
def train_model(
    model_name,
    data_name,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=30,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_model_path = f"{model_name}_{data_name}_best.pth"
    best_model_metrics = f"{model_name}_{data_name}_best_metrics.txt"
    best_accuracy = 0.0

    best_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
        # compute training accuracy

        train_metrics = evaluate_model(model, train_loader, device)
        train_accuracy = train_metrics["accuracy"]
        train_precision = train_metrics["precision"]
        train_recall = train_metrics["recall"]
        train_f1 = train_metrics["f1"]
        print(f"Accuracy on training dataset: {train_accuracy}")
        print(f"Training Precision: {train_precision}")
        print(f"Training Recall: {train_recall}")
        print(f"Training F1 Score: {train_f1}")

        # compute validation accuracy
        val_metrics = evaluate_model(model, val_loader, device)
        val_accuracy = val_metrics["accuracy"]
        val_precision = val_metrics["precision"]
        val_recall = val_metrics["recall"]
        val_f1 = val_metrics["f1"]

        print(f"Accuracy on validation dataset: {val_accuracy}")
        print(f"Validation Precision: {val_precision}")
        print(f"Validation Recall: {val_recall}")
        print(f"Validation F1 Score: {val_f1}")
        # Save the best model based on validation accuracy
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_metrics = val_metrics
            torch.save(model.state_dict(), best_model_path)

            # Save metrics in a text file
            with open(best_model_metrics, "w") as f:
                f.write(f"Best Model Validation Metrics:\n")
                f.write(f"Accuracy: {best_metrics['accuracy']}\n")
                f.write(f"Precision: {best_metrics['precision']}\n")
                f.write(f"Recall: {best_metrics['recall']}\n")
                f.write(f"F1 Score: {best_metrics['f1']}\n")
                # trainig metrics
                f.write(f"Training Accuracy: {train_accuracy}\n")
                f.write(f"Training Precision: {train_precision}\n")
                f.write(f"Training Recall: {train_recall}\n")
                f.write(f"Training F1 Score: {train_f1}\n")
                # number of epochs
                f.write(f"Number of epochs: {epoch+1}\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Data: {data_name}\n")
                # number of trainable parameters
                f.write(
                    f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n"
                )
                # number of training samples
                f.write(f"Number of training samples: {len(train_loader.dataset)}\n")
                # number of validation samples
                f.write(f"Number of validation samples: {len(val_loader.dataset)}\n")


def evaluate_model(model, loader, device):
    """
    Evaluate the model on the given data loader and calculate accuracy, F1 score, recall, and precision.

    Parameters:
    - model: The PyTorch model to evaluate.
    - loader: DataLoader for the dataset to evaluate against.
    - device: The device (CPU or CUDA) the model is running on.

    Returns:
    - A dictionary containing the evaluated metrics: accuracy, f1, recall, and precision.
    """
    model.eval()  # Set model to evaluation mode
    true_labels = []
    predictions = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(preds.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="macro")
    recall = recall_score(true_labels, predictions, average="macro")
    precision = precision_score(true_labels, predictions, average="macro")

    metrics = {"accuracy": accuracy, "f1": f1, "recall": recall, "precision": precision}

    return metrics


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def main():
    print("Training classifier")
    args = parse_arguments()
    config = load_config(args.config)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    model_names = config["model_names"]
    data_names = config["data_names"]
    name_to_path = config["name_to_path"]
    included_folders = ["class_ch2_ed", "class_ch2_es", "class_ch4_ed", "class_ch4_es"]

    for model_name in model_names:
        for data_name in data_names:
            print(f"Training {model_name} on {data_name} data")
            # check if the model has been saved before and also the metrics then pass
            if os.path.isfile(f"{model_name}_{data_name}_best.pth") and os.path.isfile(
                f"{model_name}_{data_name}_best_metrics.txt"
            ):
                continue
            new_data_dir = name_to_path[data_name]
            train_dataset, val_dataset = load_and_prepare_datasets(
                args.data_dir, new_data_dir, args.val_dir, transform, included_folders
            )

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            # number of class labels in data
            print(
                f"Number of class labels in training data: {len(set([label for _, label in val_dataset.samples]))}"
            )

            model = initialize_model(model_name)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
            train_model(
                model_name,
                data_name,
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
            )


if __name__ == "__main__":
    main()
