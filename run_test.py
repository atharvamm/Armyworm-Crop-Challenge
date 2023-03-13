# Importing required modules
from dataset import *
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from network import *
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import torchvision
import gc

#### Define config dictionary
config = {
    "epochs" : 1,
    "batch_size" : 4,
    "init_lr" : 2e-5,
    "weight_decay" : 1e-10,
    "lr_mode" : "max",
    "factor" : 0.5,
    "patience" : 10,
    "shuffle" : True,
    "logs_string" : "Logs for Makerere Challenge\n\n\n\n",
    "clean_logs" : True,
    "threshold_mode" : "rel",
    "threshold_scheduler" : 1e-3,
    "load_checkpoint" : "submits/checkpoint_95.06_03_12_2023_16_53_55.pth",
}

# Validation Iteration
def val(model,dataloader,criterion):

    model.eval()

    correct = 0
    total_loss = 0

    for i,data in enumerate(dataloader):
        images,labels = data
        images,labels = images.to(DEVICE),labels.to(DEVICE)
        with torch.inference_mode():
            outputs = model(images)
            cur_loss = criterion(outputs,labels).cpu().detach().numpy()
        correct += int((torch.argmax(outputs,axis = 1) == labels).cpu().detach().numpy().sum())
        total_loss += cur_loss

    acc = correct / (config["batch_size"]*len(dataloader))
    epoch_loss = total_loss/len(dataloader)
    return epoch_loss,acc

# Test Function
def test(model,dataloader):
    
    model.eval()
    predictions = [[],[]]
    for i,data in enumerate(dataloader):
        image_ids,images = data
        
        images = images.to(DEVICE)

        outputs = model(images)
        predicted = torch.argmax(outputs,axis = 1)
        predictions[0].extend(image_ids)
        predictions[1].extend(predicted.tolist())
    
    return predictions

# Write prediction to file
def write_op(predictions = [[],[]],dt = None):
    cols = ["Image_id","Label"]
    data = {cols[0] : predictions[0],cols[1] : predictions[1]}
    pred_df = pd.DataFrame(data = data,columns=cols)
    pred_df.to_csv("submissions/Subimssion_{}.csv".format(date_time),index=False)

if __name__ == "__main__":

    # Data Directory
    train_fname,test_fname = "files/Train.csv","files/Test.csv"
    train_data,test_data = pd.read_csv(train_fname),pd.read_csv(test_fname)
    train_examples,train_labels = train_data['Image_id'],train_data['Label']

    # Loading Data
    X_train,X_val,y_train,y_val = train_test_split(train_examples,train_labels,test_size=0.2,shuffle=True,random_state=42)
    X_train = X_train.values.tolist()
    X_val = X_val.values.tolist()
    y_train = y_train.values.tolist()
    y_val = y_val.values.tolist()

    # Validation Image Transforms
    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
    ])


    val_dataset = ImageDataset(X_val,y_val,val_transforms)
    test_data = test_data.values.tolist()
    test_dataset = TestImageDataset(test_data,val_transforms)

    # Dataloaders
    val_dataloader = DataLoader(val_dataset,batch_size=config["batch_size"],shuffle=config["shuffle"])
    test_dataloader = DataLoader(test_dataset,batch_size=config["batch_size"],shuffle=False)

    # Print Details
    print("Device: ",DEVICE)
    print("Batch Size:",config["batch_size"])
    print("Validation Batches:",len(val_dataloader))
    print("Test Batches:",len(test_dataloader))

    # Initialize Model
    model = Network()
    checkpoint = torch.load(config["load_checkpoint"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)


    # Initialize optimizer,lr scheduler,loss function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params= model.parameters(),lr = config["init_lr"],weight_decay=config["weight_decay"])
    # optimizer.to(DEVICE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode = config["lr_mode"],factor=config["factor"],patience=config["patience"],threshold_mode=config["threshold_mode"],threshold=config["threshold_scheduler"])

    from datetime import datetime
    now = datetime.now() # current date and time
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    write_log_op = "logs/logs_"+date_time+".txt"
    # Log Data
    with open(write_log_op,'w') as f:
        f.write(config["logs_string"])
    best_acc = 0.01
    log_cols = ["epoch", "train_loss", "train_acc","val_loss","val_acc","learning_rate"]
    logs = pd.DataFrame(columns=log_cols)

    # Iterating over epochs
    gc.collect()
    torch.cuda.empty_cache()
    
    cur_lr = float(optimizer.param_groups[0]["lr"])


    val_loss,val_acc = val(model=model,dataloader=val_dataloader,criterion=loss_fn)
    
    # Testing Model
    predictions = test(model=model,dataloader=test_dataloader)
    print("Writing Output to File")
    write_op(predictions=predictions,dt = date_time)


# Use argument parser and shell scripts.

# Useful shell commands

# rm -rf checkpoints/*
# rm -rf logs/*
# nohup python main.py > output.txt 2>&1 &