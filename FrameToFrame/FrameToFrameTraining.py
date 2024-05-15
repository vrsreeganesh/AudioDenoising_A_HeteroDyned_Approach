'''
====================================================================================================
# Aim: 
    We train a neural network that takes in the noisy-frame and produces its estimate of the clean-frame.
# Note:

====================================================================================================
'''
# LIBRARIES/PACKAGES #####################################################################
import torch; import scipy; import scipy.io; import matplotlib.pyplot as plt; import math 
import torch; import torch.nn as nn; import sys; import torchaudio; 
import os; import numpy as np; import datetime; import time; import random;
# MODEL ###############################################################
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.enc00 = nn.Sequential(nn.Linear(320, 256), nn.ReLU(True))
        self.enc01 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(True))
        self.enc02 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(True));
        
        self.dec02 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(True));
        self.dec01 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(True))
        self.dec00 = nn.Sequential(nn.Linear(256, 320), nn.ReLU(True))

        self.linearlayer = nn.Sequential(nn.Linear(320, 320))
        

    def forward(self, x):
        x00 = self.enc00(x);
        x01 = self.enc01(x00);
        x = self.enc02(x01);

        x = self.dec02(x);
        x = self.dec01(x+x01);
        x = self.dec00(x+x00)

        x = self.linearlayer(x)

        return x

# FUNCTIONS ###############################################################
def FetchData(inputfilelist, outputfilelist):
    '''
    So this function receives two lists.
    The first list contains the files that stores the clean frames. The ideal output.
    The second list stores the files that contain the dirty frames. The input.
    '''
    # choosing the files to pick from list of names
    indicestopick = np.random.randint(0, len(inputfilelist)); # randomly choosing some file indices
    inputfiletitle = inputfilelist[indicestopick]; outputfiletitle = outputfilelist[indicestopick];
    # loading the mat files
    inputmat = scipy.io.loadmat(inputfiletitle); outputmat = scipy.io.loadmat(outputfiletitle);
    # taking the required matrices
    inputdata = inputmat['speechnoisemixframes']; outputdata = outputmat['cleanspeechframes'];
    # converting to numpy
    inputdata = torch.from_numpy(inputdata); outputdata = torch.from_numpy(outputdata); # converting the files to torch
    # transposing so that the channels come first
    inputdata = torch.transpose(inputdata, 0, 1); outputdata = torch.transpose(outputdata, 0, 1); # tranposing the data
    # just taking the first 257 cause symmetry of fft
    # inputdata = inputdata[:, 0:257]; outputdata = outputdata[:, 0:257]; 
    return inputdata, outputdata
def get_files_with_prefix(directory, prefix):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.startswith(prefix)]
    return files
def match_files(clean_files, noise_files):
    # Create a dictionary to store matches
    matches = {}

    # Match files based on the common parts of their names
    for clean_file in clean_files:
        clean_common_part = clean_file.split('_', 1)[1]  # Extract the part after the prefix
        for noise_file in noise_files:
            noise_common_part = noise_file.split('_', 1)[1]  # Extract the part after the prefix
            if clean_common_part == noise_common_part:
                matches[clean_file] = noise_file
                break  # Move to the next clean file

    return matches
def butter_lowpass(cutoff, fs, order=5):
    return scipy.signal.butter(order, cutoff, fs=fs, btype='low', analog=False)
def butter_lowpass_filter(data, cutoff, fs, order, axis):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = scipy.signal.lfilter(b, a, data, axis = axis)
    return y
def butter_highpass(cutoff, fs, order=5):
    return scipy.signal.butter(order, cutoff, fs=fs, btype='highpass', analog=False)
def butter_highpass_filter(data, cutoff, fs, order, axis):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = scipy.signal.lfilter(b, a, data, axis = axis)
    return y
def frequencyshiftleftby4000(currentinput_highpass):
    # high-pass shifting
    shiftingsignal = np.sin(2*np.pi*(4000/16000)*currentinput_highpass.shape[1]*np.arange(0, currentinput_highpass.shape[1])/currentinput_highpass.shape[1]);
    shiftingsignal = np.tile(shiftingsignal[None,:], (currentinput_highpass.shape[0],1))
    currentinput_highpass_shifted = np.multiply(currentinput_highpass, shiftingsignal);
    currentinput_highpass_shifted_filtered = butter_lowpass_filter(currentinput_highpass_shifted, 4000, 16000, 30, axis = 1);

    return currentinput_highpass_shifted_filtered
def frequencyshiftrightby4000(currentinput_highpass):
    # high-pass shifting
    shiftingsignal = np.sin(-2*np.pi*(4000/16000)*currentinput_highpass.shape[1]*np.arange(0, currentinput_highpass.shape[1])/currentinput_highpass.shape[1]);
    shiftingsignal = np.tile(shiftingsignal[None,:], (currentinput_highpass.shape[0],1))
    currentinput_highpass_shifted = np.multiply(currentinput_highpass, shiftingsignal);
    currentinput_highpass_shifted_filtered = butter_highpass_filter(currentinput_highpass_shifted, 4000, 16000, 30, axis = 1);

    return currentinput_highpass_shifted_filtered
#########################################################################################################
def train():

    # sending to gpu =================================================
    if torch.backends.mps.is_available(): device = torch.device('mps')
    elif torch.cuda.is_available(): device = torch.device('cuda')
    else: device = torch.device('cpu')
    
    # initializing autoencoders =======================================================
    autoencoder = Autoencoder().to(device); autoencoder_high = Autoencoder().to(device)

    # just running the code through the network
    batchsize = 16; # pretty good
    criterion = torch.nn.MSELoss()

    # optimizers ==========================================================
    optimizer_low = torch.optim.SGD(autoencoder.parameters(),lr = 5); 
    optimizer_high = torch.optim.SGD(autoencoder_high.parameters(),lr = 5);
    
    # saving the model ============================================================================================================================================
    currenttimestring = datetime.datetime.now(); currenttimestring = currenttimestring.strftime("%m_%d_%H_%M_%S");
    modelname_low = 'model_low_'+sys.argv[0][:-3]+'.pth';
    modelname_high = 'model_high_'+sys.argv[0][:-3]+'.pth';

    # getting train and test list ==================================================================================================================================
    directory = '/Users/vrsreeganesh/Desktop/BUClasses/Thesis/Data/Dataset06_RawAudioFrames/'; clean_files = get_files_with_prefix(directory, 'cleanspeechframes_'); 
    noise_files = get_files_with_prefix(directory, 'speechnoisemixframes_'); file_matches = match_files(clean_files, noise_files);
    
    # Extract the paths of the matched files ====================================================
    outputfilelist = [os.path.join(directory, clean_file) for clean_file in file_matches.keys()]
    inputfilelist = [os.path.join(directory, noise_file) for noise_file in file_matches.values()]

    # Splitting into train and test set =================================================================================================
    N = len(outputfilelist); numbers = list(range(N)); random.shuffle(numbers); trainpercent = 0.8; n_first_list = int(trainpercent * N); 
    first_list = numbers[:n_first_list]; second_list = numbers[n_first_list:]
    train_outputfilelist = [outputfilelist[x] for x in first_list]; train_inputfilelist = [inputfilelist[x] for x in first_list]
    test_outputfilelist = [outputfilelist[x] for x in second_list]; test_inputfilelist = [inputfilelist[x] for x in second_list]

    # parameters for training ===============================================================
    losslist = []; testlosslist = []; baselineloss = []; losslowlist = []; losshighlist = [];

    
    # Training ##############################################################################
    for i in range(math.floor(1e5)):
        # fetching data
        inputdata, outputdata = FetchData(train_inputfilelist, train_outputfilelist)
        for j in range(20):

            # getting data ========================================================================================================
            autoencoder.train(); autoencoder_high.train()
            rowstopick = torch.randint(0, inputdata.shape[0], size = (batchsize,))
            currentinput = inputdata[rowstopick,:].float().to(device); currentoutput = outputdata[rowstopick,:].float().to(device); 

            # low-pass filtering ====================================================================
            currentinput_lowpass = butter_lowpass_filter(currentinput.cpu().detach().numpy(), 4000, 16000, 30, axis = 1);
            currentoutput_lowpass = butter_lowpass_filter(currentoutput.cpu().detach().numpy(), 4000, 16000, 30, axis = 1);

            # high-pass filtering ===================================================================
            currentinput_highpass = butter_highpass_filter(currentinput.cpu().detach().numpy(), 4000, 16000,30, axis = 1);
            currentoutput_highpass = butter_highpass_filter(currentoutput.cpu().detach().numpy(), 4000, 16000,30, axis = 1);

            # shifting the frequencies of the highpass signal =======================================
            currentinput_highpass_shifted_filtered = frequencyshiftleftby4000(currentinput_highpass);
            currentoutput_highpass_shifted_filtered = frequencyshiftleftby4000(currentoutput_highpass);

            # prepping the inputs to feed to the neural net =========================================
            currentinput_lowpass = torch.from_numpy(currentinput_lowpass).float().to(device);
            currentinput_highpass = torch.from_numpy(currentinput_highpass_shifted_filtered).float().to(device);
            currentoutput_lowpass = torch.from_numpy(currentoutput_lowpass).float().to(device);
            currentoutput_highpass = torch.from_numpy(currentoutput_highpass_shifted_filtered).float().to(device);

            # running the signals through the neural net. ===========================================
            netoutput_low = autoencoder(currentinput_lowpass);
            netoutput_high = autoencoder_high(currentinput_highpass);            

            # calculating loss ======================================================================
            loss_low = criterion(netoutput_low, currentoutput_lowpass); 
            loss_high = criterion(netoutput_high, currentoutput_highpass); 

            # optimizing ============================================================================
            optimizer_low.zero_grad(); optimizer_high.zero_grad(); 
            loss_low.backward(); loss_high.backward(); 
            optimizer_low.step(); optimizer_high.step();

        ####################### testing 
        if i%10 == 0:

            # adding to loss-list ==================================================================
            print("i = ", i);
            losslowlist.append(loss_low.item()); losshighlist.append(loss_high.item());

            # fetching input-output data ===========================================================
            autoencoder.eval(); autoencoder_high.eval();
            inputdata, outputdata = FetchData(test_inputfilelist, test_outputfilelist)
            rowstopick = torch.randint(0, inputdata.shape[0], size = (batchsize,))
            currentinput = inputdata[rowstopick,:].float().to(device); currentoutput = outputdata[rowstopick,:].float().to(device); 

            # low-pass filtering ===================================================================
            currentinput_lowpass = butter_lowpass_filter(currentinput.cpu().detach().numpy(), 4000, 16000, 30, axis = 1);
            currentoutput_lowpass = butter_lowpass_filter(currentoutput.cpu().detach().numpy(), 4000, 16000, 30, axis = 1);

            # high-pass filtering ==================================================================
            currentinput_highpass = butter_highpass_filter(currentinput.cpu().detach().numpy(), 4000, 16000,30, axis = 1);
            currentoutput_highpass = butter_highpass_filter(currentoutput.cpu().detach().numpy(), 4000, 16000,30, axis = 1);

            # shifting the frequencies of the highpass signal ======================================
            currentinput_highpass_shifted_filtered = frequencyshiftleftby4000(currentinput_highpass);
            currentoutput_highpass_shifted_filtered = frequencyshiftleftby4000(currentoutput_highpass);

            # prepping the inputs to feed to the neural net ========================================
            currentinput_lowpass = torch.from_numpy(currentinput_lowpass).float().to(device);
            currentinput_highpass = torch.from_numpy(currentinput_highpass_shifted_filtered).float().to(device);
            currentoutput_lowpass = torch.from_numpy(currentoutput_lowpass).float().to(device);
            currentoutput_highpass = torch.from_numpy(currentoutput_highpass_shifted_filtered).float().to(device);

            # running the signals through the neural net. ==========================================
            netoutput_low = autoencoder(currentinput_lowpass); netoutput_high = autoencoder_high(currentinput_highpass);    

            # reconstructing the signal ============================================================
            reconstructedsignal = netoutput_low.cpu().detach() + frequencyshiftrightby4000(netoutput_high.cpu().detach());
            reconstructedsignal = np.roll(reconstructedsignal, -10, axis = 1) # this is done because there is a circulsr shifting when filtering is done. This is done to off-set it. 

            # finding the test-loss ================================================================
            reconstructedsignal = torch.from_numpy(reconstructedsignal).float().to(device);
            loss = criterion(reconstructedsignal, currentoutput);

            # adding loss to list ==================================================================
            testlosslist.append(loss.item());

            # saving the model =====================================================================
            torch.save(autoencoder, modelname_low)
            torch.save(autoencoder_high, modelname_high)

            # Plotting =============================================================================
            # Plotting =============================================================================
            # Plotting Losses =============================================================================
            plt.figure(1);
            plt.clf()  # or plt.cla() to clear only the current axes
            plt.plot(losslowlist,linewidth=0.5,  color = "red",    label = 'losslist-low')
            plt.plot(losshighlist,linewidth=0.5, color = "blue",   label = 'losslist-high')
            plt.plot(testlosslist,linewidth=0.5, color = "orange", label = 'test loss list')
            # plt.plot(baselineloss,linewidth=0.5, linestyle=':', label = 'baseline loss')
            plt.legend(); plt.draw(); plt.pause(0.001)

            # Plotting Outputs =============================================================================
            plt.figure(2);
            progressidealoutput = currentoutput[0,:].cpu().detach().numpy();
            progressnetoutput = reconstructedsignal[0,:].cpu().detach().numpy();
            plt.clf()  # or plt.cla() to clear only the current axes
            # plt.plot(progressinput, linewidth = 0.5, color = 'red', label = 'progressinput');
            plt.plot(progressidealoutput, linewidth = 0.5, color = 'blue', label = 'progress ideal output');
            plt.plot(progressnetoutput, linewidth = 1.5,color = 'orange', label = 'progressnetoutput');
            plt.legend(); plt.draw(); plt.pause(0.01)

            # bruhnum = 400;
            # bruh = i%bruhnum;
            # if bruh<=bruhnum/2:
            #     # plotting
            #     plt.clf()  # or plt.cla() to clear only the current axes
            #     plt.plot(losslowlist,linewidth=0.5,  color = "red",    label = 'losslist-low')
            #     plt.plot(losshighlist,linewidth=0.5, color = "blue",   label = 'losslist-high')
            #     plt.plot(testlosslist,linewidth=0.5, color = "orange", label = 'test loss list')
            #     # plt.plot(baselineloss,linewidth=0.5, linestyle=':', label = 'baseline loss')
            #     plt.legend(); plt.draw(); plt.pause(0.001)
            # else:
            #     # Plotting Input-Prediction-Ideal Output data 
            #     # progressinput = currentinput[0,:].cpu().detach().numpy();
            #     progressidealoutput = currentoutput[0,:].cpu().detach().numpy();
            #     progressnetoutput = reconstructedsignal[0,:].cpu().detach().numpy();
            #     plt.clf()  # or plt.cla() to clear only the current axes
            #     # plt.plot(progressinput, linewidth = 0.5, color = 'red', label = 'progressinput');
            #     plt.plot(progressidealoutput, linewidth = 0.5, color = 'blue', label = 'progress ideal output');
            #     plt.plot(progressnetoutput, linewidth = 1.5,color = 'orange', label = 'progressnetoutput');
            #     plt.legend(); plt.draw(); plt.pause(0.01)

def test():

    # sending the data to the GPU
    if torch.backends.mps.is_available(): device = 'mps'
    else: device = 'cpu'

    # loading the two models
    autoencoder = torch.load('/Users/vrsreeganesh/Desktop/BUClasses/Thesis/Data/Dataset06_RawAudioFrames/Models/model_low_P17_00C_FrequencyShifting.py_04_14_18_46_14.pth')
    autoencoder_high = torch.load('/Users/vrsreeganesh/Desktop/BUClasses/Thesis/Data/Dataset06_RawAudioFrames/Models/model_high_P17_00C_FrequencyShifting.py_04_14_18_46_14.pth')
    autoencoder = autoencoder.to(device); autoencoder_high = autoencoder_high.to(device)

    # loading data
    inputdatatitle = '/Users/vrsreeganesh/Desktop/BUClasses/Thesis/Data/Dataset06_RawAudioFrames/test.mat'
    inputmat = scipy.io.loadmat(inputdatatitle); currentinput = inputmat['speechnoisemixframes']
    currentinput = torch.from_numpy(currentinput); currentinput = torch.transpose(currentinput, 0, 1).float(); 
    # currentinput = currentinput.to(device);

    # low-pass filtering
    currentinput_lowpass = butter_lowpass_filter(currentinput, 4000, 16000, 30, axis = 1);

    # high-pass filtering
    currentinput_highpass = butter_highpass_filter(currentinput, 4000, 16000,30, axis = 1);

    # shifting the frequencies of the highpass signal
    currentinput_highpass_shifted_filtered = frequencyshiftleftby4000(currentinput_highpass);

    # prepping the inputs to feed to the neural net
    currentinput_lowpass = torch.from_numpy(currentinput_lowpass).float().to(device);
    currentinput_highpass = torch.from_numpy(currentinput_highpass_shifted_filtered).float().to(device);

    # running the signals through the neural net. 
    netoutput_low = autoencoder(currentinput_lowpass);
    netoutput_high = autoencoder_high(currentinput_highpass);    

    # reconstructing the signal
    reconstructedsignal = netoutput_low.cpu().detach() + frequencyshiftrightby4000(netoutput_high.cpu().detach());
    reconstructedsignal = np.roll(reconstructedsignal, -10, axis = 1)

    # just changing names
    outputdata = reconstructedsignal;

    # # converting and saving results.
    # outputdata = outputdata.numpy();
    scipy.io.savemat('/Users/vrsreeganesh/Desktop/BUClasses/Thesis/Data/Dataset06_RawAudioFrames/testresults.mat', {"netoutput":outputdata});
if __name__ == "__main__":
    arg1 = sys.argv[1]  # First argument (index 0 is the script name)
    

    if arg1 is not None:
        if arg1=='train': 
            totrainornottotrain = 1;
            print("beginning training")
        else: 
            totrainornottotrain = 0;
            print("beginning testing")
    else:
        print("beginning training");
        totrainornottotrain = 1;  
    
    if totrainornottotrain:
        train();
    else:
        test();


