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
    "epochs" : 50,
    "batch_size" : 4,
    "init_lr" : 1e-5,
    "momentum" : 0.9,
    "weight_decay" : 1e-3,
    "lr_mode" : "max",
    "factor" : 0.25,
    "patience" : 3,
    "shuffle" : True,
    "logs_string" : "Logs for Makerere Challenge\n\n\n\n"
}

# Cur Time

# Train Iteration
def train(model,dataloader,optimizer,criterion):
    model.train()

    correct = 0
    total_loss = 0

    for i,data in enumerate(dataloader):
        optimizer.zero_grad()

        images,labels = data
        images,labels = images.to(DEVICE),labels.to(DEVICE)

        outputs = model(images)
        correct += int((torch.argmax(outputs,axis = 1) == labels).cpu().detach().numpy().sum())
        cur_loss = criterion(outputs, labels)
        total_loss += float(cur_loss.cpu().detach().numpy().item())

        cur_loss.backward()
        optimizer.step()
    
    acc = correct/(config["batch_size"]*len(dataloader))
    epoch_loss = total_loss/(len(dataloader))

    return epoch_loss,acc

# Validation Iteration
def val(model,dataloader,criterion):

    model.eval()

    correct = 0
    total_loss = 0

    for i,data in enumerate(dataloader):
        images,labels = data
        images,labels = images.to(DEVICE),labels.to(DEVICE)

        outputs = model(images)
        cur_loss = criterion(outputs,labels).cpu().detach().numpy()
        correct = int((torch.argmax(outputs,axis = 1) == labels).cpu().detach().numpy().sum())
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

# Write Logs
def write_logs(logf = "logs.txt",vals = [1,2,3,4,5,6]):
    with open(logf ,'a') as f:
        append_string = "Epoch:{}, Train L:{:.4f}, Train A:{:.2f}, Val L:{:.4f}, Val A:{:.2f}, LR:{}\n".format(*vals)
        f.write(append_string)

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

    # Training Image Transforms
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
    ])

    # Validation Image Transforms
    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
    ])


    train_dataset = ImageDataset(X_train,y_train,train_transforms)
    val_dataset = ImageDataset(X_val,y_val,val_transforms)
    test_data = test_data.values.tolist()
    test_dataset = TestImageDataset(test_data,val_transforms)

    # Dataloaders
    train_dataloader = DataLoader(train_dataset,batch_size=config["batch_size"],shuffle=config["shuffle"])
    val_dataloader = DataLoader(val_dataset,batch_size=config["batch_size"],shuffle=config["shuffle"])
    test_dataloader = DataLoader(test_dataset,batch_size=config["batch_size"],shuffle=False)

    # Print Details
    print("Device: ",DEVICE)
    print("Batch Size:",config["batch_size"])
    print("Train Batches:",len(train_dataloader))
    print("Validation Batches:",len(val_dataloader))
    print("Test Batches:",len(test_dataloader))

    # Initialize Model
    model = Network()
    model.to(DEVICE)
    # Initialize Model Weights
    @torch.no_grad()
    def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)

        elif type(m) == torch.nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight)

    # Initialize optimizer,lr scheduler,loss function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params= model.parameters(),lr = config["init_lr"],momentum= config["momentum"],weight_decay=config["weight_decay"])
    # optimizer.to(DEVICE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode = config["lr_mode"],factor=config["factor"],patience=config["patience"])

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
    for epoch in range(config["epochs"]):
        gc.collect()
        torch.cuda.empty_cache()
        
        cur_lr = float(optimizer.param_groups[0]["lr"])

        
        train_loss,train_acc = train(model=model,dataloader=train_dataloader,optimizer=optimizer,criterion=loss_fn)


        val_loss,val_acc = val(model=model,dataloader=val_dataloader,criterion=loss_fn)

        # Write Log to text
        # Log parameter values
        log_vals = [epoch+1,train_loss,100*train_acc,val_loss,100*val_acc,cur_lr] # Replace with current learning rate, you can get value from either optimizer or scheduler
        logs.loc[len(logs.index)] = log_vals 
        write_logs(logf=write_log_op,vals=log_vals)

        
        if val_acc > best_acc:
            torch.save(
                {
                    "model_state_dict" : model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'scheduler_state_dict' : scheduler.state_dict(),
                    'val_acc' : val_acc, 
                    'epoch' : epoch,
                },
                "checkpoints/checkpoint_{}_{}.pth".format(val_acc,date_time)
            )
            best_acc = val_acc

        scheduler.step(val_acc)
    logs.to_csv("logs/logs_record_" + date_time + ".csv",index = False)
    
    # Testing Model
    predictions = test(model=model,dataloader=test_dataloader)
    print("Writing Output to File")
    write_op(predictions=predictions,dt = date_time)


# Use argument parser and shell scripts.

# Useful shell commands

# rm -rf checkpoints/*
# rm -rf logs/*
# nohup python main.py > output.txt 2>&1 &