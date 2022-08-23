from flask import Flask, render_template, request, session
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, session
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import PIL
import os
import torch
import torch.nn as nn
import torchvision 
import torchvision.transforms as transforms

# Model Architecture
class Convnet(nn.Module):
    def __init__(self):
        super(Convnet,self).__init__()
        
        #convolutional layer 1
            self.layer1 = nn.Sequential(
                nn.Conv2d(3,16,kernel_size=3, padding=0, stride=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            #convolutional layer 2
            self.layer2 = nn.Sequential(
                nn.Conv2d(16,32, kernel_size=3, padding=0, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2)
                )
            #convolutional layer 3
            self.layer3 = nn.Sequential(
                nn.Conv2d(32,64, kernel_size=3, padding=0, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2)
                )
            
            #Linear layer 1
            self.fc1 = nn.Linear(3*3*64,10)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(10,2)
            self.relu = nn.ReLU()
            
            #forward propagation
        def forward(self,x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = out.view(out.size(0),-1)
            out = self.relu(self.fc1(out))
            out = self.fc2(out)
            return out






app = Flask(__name__)
@app.route('/')
def image_classifier():
	return render_template('index.html')



@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        image = PIL.Image.open(file_path)
       
        transform = transforms.Compose([
                                  
                                  transforms.RandomResizedCrop(255),
                                  transforms.RandomHorizontalFlip(255), #flip image
                                  transforms.ToTensor(), #convert to tensor format
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #normalize tensors
                                      ])
        image = transform(image).unsqueeze(0)
        model = torch.load('model.pth', map_location={'cuda:0': 'cpu'})
        model.eval()
        pred = torch.argmax(model(image))
        print(pred)
        predict=pred.item()
        if predict==0:
            return('The image is predicted to be a CAT')
        elif predict==1:
            return('The image is predicted to be a DOG')
        elif predict==2:
            return('The image cannot be identified')
        
    return render_template('index.html', predict)


    
if __name__ == '__main__':
	app.run(debug=True)
