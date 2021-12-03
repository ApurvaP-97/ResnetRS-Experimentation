# Project Structure
Code files : Configure.py, DataLoader.py, ImageUtils.py, Model.py, Network.py, main.py </br>

# (1) Running the project
## Training : For the training mode to be active. The training configs are passed directly to model.train(train_data,test_data,epochs,lr,wt_decay,momentum) in main.py
python main.py </br>

# (2) For loading the saved model 
## path_to_ckpt is the path to checkpoint file : 'Apurva Purushotama-91.19 acc.ckpt'
model = MyModel() </br>
model.network.load_state_dict(torch.load(path_to_ckpt)) </br>

# (3) Testing : To check the test accuracy on the public test dataset. test_data will be available upon executing the main.py file
## Comment model.train() line in main.py file
model.training = False </br>
model.evaluate(test_data) </br>

# (4) Prediction : To get predictions for the private dataset
## Load the saved model - step (2) mentioned above.
## Load private dataset into private_data - private_data_path : root path to private dataset
private_data = load_testing_images(private_data_path, preprocess_config)</br>
model.pred_prob(private_data)</br>
