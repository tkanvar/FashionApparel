import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from sklearn.preprocessing import OneHotEncoder
import os

def readTestData():
    test_img_folder = "../test"

    test_data = pd.read_csv('../test.csv')
    test_data = test_data.to_numpy()

    test_img_path = ["../test/" + str(x) + ".png" for x in test_data[:, 0]]
    print ("# of test data = ", len(test_img_path))
    test_img = []
    for x in test_img_path:
        test_img.append(rgb2gray(mpimg.imread(x)))

    print ("Collected Test")

    test_data = torch.from_numpy(test_data)

    return test_data, test_img


def readData(batch_size):
    folder = "train"
    train_img_folder = "../" + folder

    train_data = pd.read_csv('../'+folder+'.csv')
    train_data = train_data.to_numpy()

    num_rows_valid = int(train_data.shape[0] * 20 / 100)

    valid_data = train_data[:num_rows_valid,:]
    train_data = train_data[num_rows_valid:,:]

    valid_img_path = ["../"+folder+"/" + str(x) + ".png" for x in valid_data[:, 0]]
    print ("# of valid data = ", len(valid_img_path))
    #valid_img = [rgb2gray(mpimg.imread(x)) for x in valid_img_path]
    valid_img = []
    for x in valid_img_path:
        valid_img.append(rgb2gray(mpimg.imread(x)))

    print ("Collected Valid")

    train_img_path = ["../"+folder+"/" + str(x) + ".png" for x in train_data[:, 0]]
    print ("# of train data = ", len(train_img_path))
    llen = len(train_img_path)
    train_img1 = [rgb2gray(mpimg.imread(train_img_path[x])) for x in range(int(llen/4))]
    print ("Trained = ", len(train_img1))
    print ("Range = 0, ", llen/4)
    train_img2 = [rgb2gray(mpimg.imread(train_img_path[x])) for x in range(int(llen/4), int(llen*2/4))]
    print ("Trained = ", len(train_img2))
    print ("Range = ", llen/4, ", ", llen*2/4)
    train_img3 = [rgb2gray(mpimg.imread(train_img_path[x])) for x in range(int(llen*2/4), int(llen*3/4))]
    print ("Trained = ", len(train_img3))
    print ("Range = ", llen*2/4, ", ", llen*3/4)
    #train_img4 = [rgb2gray(mpimg.imread(train_img_path[x])) for x in range(int(llen*3/4), llen)]
    train_img4 = []
    for x in range(int(llen*3/4), llen):
        train_img4.append(rgb2gray(mpimg.imread(train_img_path[x])))
    print ("Trained = ", len(train_img4))
    print ("Range = ", llen*3/4, ", ", llen*4/4)

    train_img = train_img1 + train_img2
    train_img = train_img + train_img3
    train_img = train_img + train_img4

    print ("Trained = ", len(train_img))

    #for i in valid_img:
    #    plt.imshow(i, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    #    plt.show()

    print ("Collected Train")

    valid_data = torch.from_numpy(valid_data)
    train_data = torch.from_numpy(train_data)

    # Divide in batches
    num_rows = train_data.size(0)
    num_batches = int(num_rows / batch_size) + 1

    train_data_list = []
    train_img_list = []
    for i in range(num_batches):
        train_data_list.append(train_data[i*batch_size:(i+1)*batch_size, :])
        train_img_list.append(train_img[i*batch_size:(i+1)*batch_size])

    #print ("train data list = ", len(train_data_list[48]))
    #print ("train img list = ", len(train_img_list[48]))

    return train_data_list, valid_data, train_img_list, valid_img, num_batches

class CNN(torch.nn.Module):
    #Our batch shape for input x is (1, 28, 28)
    def __init__(self):
        super(CNN, self).__init__()

        self.showImg = False

        self.conv1 = torch.nn.Conv2d(1, 28, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.drop_out = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(28 * 14 * 14, 100)
        self.fc2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 28 * 14 * 14)
        x = self.drop_out(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return(x)

def createLossAndOptimizer(net, learning_rate=0.001):
    #Loss function
    loss = torch.nn.CrossEntropyLoss()
    
    #Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    return(loss, optimizer)

def trainNet(net, batch_size, n_epochs, learning_rate, threshold):
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("threshold=", threshold)
    print("=" * 30)

    print ("Collect Data")

    ######### TRAINING
    modelFile = "model.txt"
    if os.path.exists(modelFile):
        net.load_state_dict(torch.load(modelFile))
        net.eval()
    else:
        #Get training data
        train_data, valid_data, train_img, valid_img, num_batches = readData(batch_size)

        #Create our loss and optimizer functions
        loss, optimizer = createLossAndOptimizer(net, learning_rate)

        print ("Training")
        #Loop for n_epochs
        for epoch in range(n_epochs):
            running_loss = 0.0
            total_train_loss = 0
            #net.showImg = True

            for i, data in enumerate(train_data):
                #Get inputs
                inputs = train_img[i]
                labels = data[:,1]

                inputs = torch.FloatTensor(inputs)
                if inputs.size(0) == 0: continue
                #print ("inputs.shape = ", inputs.shape)
                inputs = inputs.reshape(inputs.size(0), 1, inputs.size(1), inputs.size(2))

                #Wrap them in a Variable object
                inputs, labels = Variable(inputs), Variable(labels)

                #Set the parameter gradients to zero
                optimizer.zero_grad()

                #Forward pass, backward pass, optimize
                outputs = net(inputs)
                loss_size = loss(outputs, labels)
                loss_size.backward()
                optimizer.step()

                #Print statistics
                running_loss += loss_size.item()
                total_train_loss += loss_size.item()

                #Print every 10th batch of an epoch
                print_every = 10
                if (i + 1) % (print_every) == 0:
                    #print ("==============TRAIN DATA=================")
                    #print ("outputs = ", outputs)
                    #print ("softmax outputs = ", [torch.nn.functional.softmax(x) for x in outputs])
                    #print ("labels = ", [x.item() for x in labels])
                    #print ("")
                    #print ("")

                    print("Epoch {}, {:d}% \t train_loss: {:.2f}".format(
                          epoch+1, int(100 * (i+1) / num_batches), running_loss / print_every))
                    #Reset running loss and time
                    running_loss = 0.0

            #At the end of the epoch, do a pass on the validation set
            val_out = []
            val_lab = []
            total_val_loss = 0
            acc = 0.0
            for i, data in enumerate(valid_data):
                inputs = valid_img[i]
                labels = data[1]
                labels = labels.reshape(1)

                inputs = torch.FloatTensor(inputs)
                inputs = inputs.reshape(1, 1, inputs.size(0), inputs.size(1))

                #plt.imshow(inputs[0][0], cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
                #plt.show()

                #Wrap tensors in Variables
                inputs, labels = Variable(inputs), Variable(labels)

                #Forward pass
                val_outputs = net(inputs)
                val_loss_size = loss(val_outputs, labels)
                total_val_loss += val_loss_size.item()

                val_out.append(val_outputs)
                val_lab.append(labels.item())
            
                maxout = torch.max(val_outputs, 1)[1].item()
                acc += labels.item() != maxout

            #print ("==============VALID DATA====================")
            #out = [torch.nn.functional.softmax(x) for x in val_out]
            #out = [round(torch.max(x).item(), 2) for x in out]
            #maxout = [torch.max(x, 1)[1].item() for x in val_out]
            #print ("softmax outputs = ", out)
            #print ("max indx = ", maxout)
            #print ("labels = ", val_lab)
            #print ("")
            #print ("")

            #print("Validation loss = {:.2f}".format(total_val_loss / valid_data.size(0)))
            print ("Accuracy = {:.2f}%".format(100 - (acc * 100 / valid_data.size(0))))

            if total_val_loss <= threshold:
                print ("Model is trained")
                break

        torch.save(net.state_dict(), modelFile)

    print("Training finished")

    ############### TESTING
    test_data, test_img = readTestData()
    f = open("SubmitTest.txt", "w")
    f.write("id,label\n")

    for i, data in enumerate(test_data):
            inputs = test_img[i]

            inputs = torch.FloatTensor(inputs)
            inputs = inputs.reshape(1, 1, inputs.size(0), inputs.size(1))
            inputs = Variable(inputs)

            test_outputs = net(inputs)

            f.write(str(data[0].item())+","+str(torch.max(test_outputs, 1)[1].item()) + "\n")

    f.close()
            
cnn = CNN()
trainNet(cnn, batch_size=1000, n_epochs=300, learning_rate=0.001, threshold=1e-1)
