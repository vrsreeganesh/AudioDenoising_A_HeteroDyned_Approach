'''
====================================================================================================
# P34_00: this method takes in noisy-frame and ss-frame to produce 512-bin magnitude and phase
    P34_00A: we'll do the frequency shifting thing and use the same 512 bin networks for mimicking those too. 
====================================================================================================
'''
# ==================================================================================================
import torch; import scipy; import scipy.io; import matplotlib.pyplot as plt; import math 
import torch; import torch.nn as nn; import sys; import torchaudio; 
import pdb;
import os; import numpy as np; import datetime; import time; import random;
# ==================================================================================================
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.enc_SS = nn.Sequential(
            nn.Linear(320, 257), nn.ReLU(True),
            nn.Linear(257, 257), nn.ReLU(True),
        )

        self.enc_Raw = nn.Sequential(
            nn.Linear(320, 257), nn.ReLU(True),
            nn.Linear(257, 257), nn.ReLU(True),
        )

        self.dec = nn.Sequential(
            nn.Linear(257, 257), nn.ReLU(True),
            nn.Linear(257, 257), nn.ReLU(True),
        )

    def forward(self, x_Raw, x_SS):
        
        # passing through first branch
        x_Raw = self.enc_Raw(x_Raw);

        # passing through SS
        x_SS = self.enc_SS(x_SS);

        # pass through common branch
        x = self.dec(x_Raw + x_SS);
        
        return x
def FetchData(inputfilelist, numfilestointegrate = 5):
    # choosing the files to pick from list of names 
    inputfiletitle = np.random.choice(inputfilelist, numfilestointegrate)

    # loading the mat files
    speechnoisemixdata_list = []; ssdata_list = []; cleanspeechdata_list = [];
    for x in inputfiletitle:
        inputmat = scipy.io.loadmat(x);
        speechnoisemixdata = inputmat["speechnoisemix_frames"]; speechnoisemixdata = torch.from_numpy(speechnoisemixdata); speechnoisemixdata = torch.transpose(speechnoisemixdata, 0, 1); speechnoisemixdata_list.append(speechnoisemixdata);
        ssdata = inputmat["SS_frames"]; ssdata = torch.from_numpy(ssdata); ssdata = torch.transpose(ssdata, 0, 1); ssdata_list.append(ssdata);
        cleanspeechdata = inputmat["cleanspeechframes"]; cleanspeechdata = torch.from_numpy(cleanspeechdata); cleanspeechdata = torch.transpose(cleanspeechdata, 0, 1); cleanspeechdata_list.append(cleanspeechdata);
    
    # concatenating them together to produce a larger tensor
    speechnoisemixdata = torch.cat(speechnoisemixdata_list, dim = 0);
    ssdata = torch.cat(ssdata_list, dim = 0);
    cleanspeechdata = torch.cat(cleanspeechdata_list, dim = 0);

    import pdb; pdb.set_trace

    
    # returning stuff
    return speechnoisemixdata, ssdata, cleanspeechdata
def NormaliseToUnitEnergy(currentinput):
    
    var00 = torch.sqrt(torch.sum(currentinput**2, dim = 1)).unsqueeze(1)
    var00[var00==0] = 1; # to make sure we're not dividing anything by zero. 
    currentinput = torch.div(currentinput,
                             torch.tile(var00, (1, currentinput.shape[1])));
    return currentinput
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
# def frequencyshiftleftby4000(currentinput_highpass):
#     # high-pass shifting
#     shiftingsignal = np.sin(2*np.pi*(4000/16000)*currentinput_highpass.shape[1]*np.arange(0, currentinput_highpass.shape[1])/currentinput_highpass.shape[1]);
#     shiftingsignal = np.tile(shiftingsignal[None,:], (currentinput_highpass.shape[0],1))
#     currentinput_highpass_shifted = np.multiply(currentinput_highpass, shiftingsignal);
#     currentinput_highpass_shifted_filtered = butter_lowpass_filter(currentinput_highpass_shifted, 4000, 16000, 30, axis = 1);

#     return currentinput_highpass_shifted_filtered
# def frequencyshiftrightby4000(currentinput_highpass):
#     # high-pass shifting
#     shiftingsignal = np.sin(-2*np.pi*(4000/16000)*currentinput_highpass.shape[1]*np.arange(0, currentinput_highpass.shape[1])/currentinput_highpass.shape[1]);
#     shiftingsignal = np.tile(shiftingsignal[None,:], (currentinput_highpass.shape[0],1))
#     currentinput_highpass_shifted = np.multiply(currentinput_highpass, shiftingsignal);
#     currentinput_highpass_shifted_filtered = butter_highpass_filter(currentinput_highpass_shifted, 4000, 16000, 30, axis = 1);

#     return currentinput_highpass_shifted_filtered
def frequencyshiftleftby4000(currentinput_highpass, order = 11):
    # first high-pass filtering at half
    currentinput_highpass = butter_highpass_filter(currentinput_highpass, 4000, 16000, order, axis = 1);

    # high-pass shifting
    fs = 16000
    timearray = np.arange(0, currentinput_highpass.shape[1])/fs;
    shiftingsignal = 2 * np.cos(2*np.pi*4000* timearray); # multiplication with scalar is intentional. 
    shiftingsignal = np.tile(shiftingsignal[None,:], (currentinput_highpass.shape[0],1))

    # multilpying with shifting signal 
    currentinput_highpass_shifted = np.multiply(currentinput_highpass, shiftingsignal);
    currentinput_highpass_shifted_filtered = butter_lowpass_filter(currentinput_highpass_shifted, 4000, fs, order, axis = 1);

    return currentinput_highpass_shifted_filtered
def frequencyshiftrightby4000(currentinput_highpass, order = 11):
    # low-pass filtering before shifting
    currentinput_highpass = butter_lowpass_filter(currentinput_highpass, 4000, 16000, order, axis = 1);

    # high-pass shifting
    fs = 16000
    timearray = np.arange(0, currentinput_highpass.shape[1])/fs;
    shiftingsignal = 2 * np.cos(2*np.pi*4000* timearray); # multiplication with scalar is intentional. 
    shiftingsignal = np.tile(shiftingsignal[None,:], (currentinput_highpass.shape[0],1))
    
    # multiplying: this will create two sidebands. 
    currentinput_highpass_shifted = np.multiply(currentinput_highpass, shiftingsignal);

    # filtering to remove lower sideband. 
    currentinput_highpass_shifted_filtered = butter_highpass_filter(currentinput_highpass_shifted, 4000, 16000, order, axis = 1);

    return currentinput_highpass_shifted_filtered
#########################################################################################################



###########################################################################################################################################################################################################
###########################################################################################################################################################################################################
###########################################################################################################################################################################################################
###########################################################################################################################################################################################################
###########################################################################################################################################################################################################

# MDOEL PATHS ###########################################################################################################################################################
modelname = '/Users/vrsreeganesh/Desktop/BUClasses/Thesis/Code/PythonFiles/DenoisingArtifacts/Models/model_'  +  os.path.basename(sys.argv[0])[:-3]  +  '.pth'; 
modelname_phase = '/Users/vrsreeganesh/Desktop/BUClasses/Thesis/Code/PythonFiles/DenoisingArtifacts/Models/model_phase_'  +  os.path.basename(sys.argv[0])[:-3]  +  '.pth'; 
modelname_magnitude_frame2 = '/Users/vrsreeganesh/Desktop/BUClasses/Thesis/Code/PythonFiles/DenoisingArtifacts/Models/model_magnitude_frame2_'  +  os.path.basename(sys.argv[0])[:-3]  +  '.pth'; 
modelname_phase_frame2 = '/Users/vrsreeganesh/Desktop/BUClasses/Thesis/Code/PythonFiles/DenoisingArtifacts/Models/model_phase_frame2'  +  os.path.basename(sys.argv[0])[:-3]  +  '.pth'; 

# tensor amplifications
tensoramplification = 1;
###########################################################################################################################################################################################################
###########################################################################################################################################################################################################
###########################################################################################################################################################################################################
###########################################################################################################################################################################################################
###########################################################################################################################################################################################################











# TRAIN ####################################################################################################################################################################################################
# TRAIN ####################################################################################################################################################################################################
# TRAIN ####################################################################################################################################################################################################
# TRAIN ####################################################################################################################################################################################################
# TRAIN ####################################################################################################################################################################################################
# TRAIN ####################################################################################################################################################################################################
# TRAIN ####################################################################################################################################################################################################
# TRAIN ####################################################################################################################################################################################################
# TRAIN ####################################################################################################################################################################################################
# TRAIN ####################################################################################################################################################################################################
def train():

    # sending to gpu #################################################
    if torch.backends.mps.is_available(): device = torch.device('mps')
    elif torch.cuda.is_available(): device = torch.device('cuda')
    else: device = torch.device('cpu')
    device = torch.device('cpu')
    
    # initializing autoencoders #######################################################
    autoencoder = Autoencoder().to(device);
    phaseencoder = Autoencoder().to(device);
    magnitudeencoder_frame2 = Autoencoder().to(device);
    phaseencoder_frame2 = Autoencoder().to(device);

    # just running the code through the network
    batchsize = 128; # pretty good
    criterion = torch.nn.MSELoss()

    # optimizers ###############################################################
    optimizer = torch.optim.SGD(autoencoder.parameters(), lr = 1);    
    optimizer_phase = torch.optim.SGD(phaseencoder.parameters(), lr = 1);
    optimizer_magnitude_frame2 = torch.optim.SGD(magnitudeencoder_frame2.parameters(), lr = 1);    
    optimizer_phase_frame2 = torch.optim.SGD(phaseencoder_frame2.parameters(), lr = 1);    

    # getting train and test list ############################################################
    directory = '/Users/vrsreeganesh/Desktop/BUClasses/Thesis/Data/Dataset16_RawFrame_SSFrame_CleanSpeechFrame'
    datafiles = get_files_with_prefix(directory, 'datamatrix_'); 
    
    # Extract the paths of the matched files #####################################################
    datafilelist = [os.path.join(directory, x) for x in datafiles];

    # Splitting into train and test set #################################################################################################
    N = len(datafilelist); numbers = list(range(N)); random.shuffle(numbers); trainpercent = 0.8; n_first_list = int(trainpercent * N);  first_list = numbers[:n_first_list]; second_list = numbers[n_first_list:] ;
    trainfilelist = [datafilelist[x] for x in first_list]; testfilelist = [datafilelist[x] for x in second_list]

    # parameters for training ==============================
    losslist = []; testlosslist = []; baselinelosslist = [];
    trainloss_frame2 = []; testloss_frame2 = [];
    baselinelosslist_frame2 = [];
    
    # Training #####################################################################################################################
    # Training #####################################################################################################################
    # Training #####################################################################################################################
    # Training #####################################################################################################################
    # Training #####################################################################################################################
    # Training #####################################################################################################################
    # Training #####################################################################################################################
    # Training #####################################################################################################################
    # Training #####################################################################################################################
    # Training #####################################################################################################################
    i = -1
    while i>-5:
        i+= 1;
        speechnoisemixdata, ssdata, outputdata = FetchData(trainfilelist, 20);

        for j in range(20):

            # putting things to train=================
            autoencoder.train(); phaseencoder.train();

            # getting data ==========================================================================================================================
            rowstopick = torch.randint(0, speechnoisemixdata.shape[0], size = (batchsize,))
            currentinput = speechnoisemixdata[rowstopick,:].float().to(device); ssinput = ssdata[rowstopick,:].float().to(device); currentoutput = outputdata[rowstopick,:].float().to(device);
            
            # building ideal output for frame 2======================================
            idealoutput_frame2 = torch.from_numpy(frequencyshiftleftby4000(currentoutput, order = 11)).float();
            currentinput_shifted = torch.from_numpy(frequencyshiftleftby4000(currentinput, order = 11)).float();
            ssinput_shifted = torch.from_numpy(frequencyshiftleftby4000(ssinput, order = 11)).float();

            # running the data through the neural net ==================================
            netoutput = autoencoder(currentinput, ssinput);
            phaseoutput = phaseencoder(currentinput, ssinput);
            netoutput_magnitude_frame2 = magnitudeencoder_frame2(currentinput_shifted, ssinput_shifted);
            netoutput_phase_frame2 = phaseencoder_frame2(currentinput_shifted, ssinput_shifted);

            # flipping to make it symmetric ============================================================================================================
            netoutput = torch.cat((netoutput, torch.flip(netoutput[:, 1:-1], dims = [1])  ), dim = 1)
            phaseoutput = torch.cat((phaseoutput, -1*torch.flip(phaseoutput[:, 1:-1], dims = [1])  ), dim = 1)
            netoutput_magnitude_frame2 = torch.cat((netoutput_magnitude_frame2, torch.flip(netoutput_magnitude_frame2[:, 1:-1], dims = [1])  ), dim = 1)
            netoutput_phase_frame2 = torch.cat((netoutput_phase_frame2, -1*torch.flip(netoutput_phase_frame2[:, 1:-1], dims = [1])  ), dim = 1)

            # combining with the phase of the input =============================================
            netoutput = netoutput * torch.exp(1j*phaseoutput);
            netoutput_frame2 = netoutput_magnitude_frame2 * torch.exp(1j*netoutput_phase_frame2);

            # taking the ifft =========================================================================
            netoutput = torch.real(torch.fft.ifftn(netoutput, axis = 1, norm = "ortho"));
            netoutput_frame2 = torch.real(torch.fft.ifftn(netoutput_frame2, axis = 1, norm = "ortho"));

            # trimming since we're working with 512-bit fft
            netoutput = netoutput[:, 0:320];
            netoutput_frame2 = netoutput_frame2[:, 0:320];

            # finding the loss ===========================================
            loss = criterion(netoutput, currentoutput); 
            loss_frame2 = criterion(netoutput_frame2, idealoutput_frame2);

            # backpropagating=================================
            optimizer.zero_grad(); optimizer_phase.zero_grad(); optimizer_magnitude_frame2.zero_grad(); optimizer_phase_frame2.zero_grad();
            loss.backward(); loss_frame2.backward();
            optimizer.step(); optimizer_phase.step(); optimizer_magnitude_frame2.step(); optimizer_phase_frame2.step();


        # EVALUATING ################################################################## 
        # EVALUATING ################################################################## 
        # EVALUATING ################################################################## 
        # EVALUATING ################################################################## 
        # EVALUATING ################################################################## 
        if i%10 == 0:

            # adding to loss-list 
            print("i = ", i)
            losslist.append(loss.item());           
            trainloss_frame2.append(loss_frame2.item()); 

            # fetching input-output data  ################################################
            autoencoder.eval(); phaseencoder.eval();
            speechnoisemixdata, ssdata, outputdata = FetchData(testfilelist, 20);
            rowstopick = torch.randint(0, speechnoisemixdata.shape[0], size = (batchsize,))
            currentinput = speechnoisemixdata[rowstopick,:].float().to(device); ssinput = ssdata[rowstopick,:].float().to(device); currentoutput = outputdata[rowstopick,:].float().to(device);

            # building ideal output for frame 2======================================
            idealoutput_frame2 = torch.from_numpy(frequencyshiftleftby4000(currentoutput, order = 11)).float();
            currentinput_shifted = torch.from_numpy(frequencyshiftleftby4000(currentinput, order = 11)).float();
            ssinput_shifted = torch.from_numpy(frequencyshiftleftby4000(ssinput, order = 11)).float();

            # running the data through the neural net ##################################
            netoutput = autoencoder(currentinput, ssinput);
            phaseoutput = phaseencoder(currentinput, ssinput);
            netoutput_magnitude_frame2 = magnitudeencoder_frame2(currentinput_shifted, ssinput_shifted);
            netoutput_phase_frame2 = phaseencoder_frame2(currentinput_shifted, ssinput_shifted);

            # flipping to make it symmetric ============================================================================================================
            netoutput = torch.cat((netoutput, torch.flip(netoutput[:, 1:-1], dims = [1])  ), dim = 1)
            phaseoutput = torch.cat((phaseoutput, -1*torch.flip(phaseoutput[:, 1:-1], dims = [1])  ), dim = 1)
            netoutput_magnitude_frame2 = torch.cat((netoutput_magnitude_frame2, torch.flip(netoutput_magnitude_frame2[:, 1:-1], dims = [1])  ), dim = 1)
            netoutput_phase_frame2 = torch.cat((netoutput_phase_frame2, -1*torch.flip(netoutput_phase_frame2[:, 1:-1], dims = [1])  ), dim = 1)

            # combining with the phase of the input =============================================
            netoutput = netoutput * torch.exp(1j*phaseoutput);
            netoutput_frame2 = netoutput_magnitude_frame2 * torch.exp(1j*netoutput_phase_frame2);

            # taking the ifft =========================================================================
            netoutput = torch.real(torch.fft.ifftn(netoutput, axis = 1, norm = "ortho"));
            netoutput_frame2 = torch.real(torch.fft.ifftn(netoutput_frame2, axis = 1, norm = "ortho"));

            # trimming since we're working with 512-bit fft
            netoutput = netoutput[:, 0:320];
            netoutput_frame2 = netoutput_frame2[:, 0:320];

            # finding the loss ===========================================
            loss = criterion(netoutput, currentoutput); 
            loss_frame2 = criterion(netoutput_frame2, idealoutput_frame2);

            # adding loss to list ##############################
            testlosslist.append(loss.item());
            testloss_frame2.append(loss_frame2.item());
            baselineloss = criterion(currentinput, currentoutput); baselinelosslist.append(baselineloss.item())
            baselineloss_frame = criterion(currentinput_shifted, idealoutput_frame2); baselinelosslist_frame2.append(baselineloss_frame.item())

            # saving the model occassionally - allows us to pause code occasionally ======
            torch.save(autoencoder, modelname); torch.save(phaseencoder, modelname_phase);
            torch.save(magnitudeencoder_frame2, modelname_magnitude_frame2);
            torch.save(phaseencoder_frame2, modelname_phase_frame2);
            
            # Plotting #############################################################################################################        

            # plotting the losses
            plt.figure(1);
            plt.clf();
            plt.subplot(1,2,1); plt.plot(losslist, linewidth = 0.5, color = 'red', label = 'frame 1 train loss');
            plt.subplot(1,2,1); plt.plot(baselinelosslist, linewidth = 0.5, color = 'green', label = 'frame 1 baseline loss list');
            plt.subplot(1,2,1); plt.plot(testlosslist,  ':', linewidth = 0.5, color = 'orange', label = 'frame 1 test loss');plt.legend();
            plt.subplot(1,2,2); plt.plot(trainloss_frame2, linewidth = 0.5, color = 'red', label = 'frame 2 train loss');
            plt.subplot(1,2,2); plt.plot(testloss_frame2, ':', linewidth = 0.5, color = 'orange', label = 'frame 2 test loss');
            plt.subplot(1,2,2); plt.plot(baselinelosslist_frame2, linewidth = 0.5, color = 'green', label = 'frame 2 baselineloss');
            plt.legend(); plt.draw(); plt.pause(0.001);

            # plotting the results
            plt.figure(2);
            plt.clf();
            var00 = netoutput[0,:].cpu().detach().numpy();
            var01 = currentoutput[0,:];
            var02 = netoutput_frame2[0,:].cpu().detach().numpy();
            var03 = idealoutput_frame2[0,:];
            plt.subplot(2,1,1); plt.plot(var00, linewidth = 0.5, color = 'orange', label = 'frame 1 netoutput');
            plt.subplot(2,1,1); plt.plot(var01, ':', linewidth = 0.5, color = 'blue', label = 'frame 1 ideal output');plt.legend();
            plt.subplot(2,1,2); plt.plot(var02, linewidth = 0.5, color = 'orange', label = 'frame 2 netoutput');
            plt.subplot(2,1,2); plt.plot(var03, ':', linewidth = 0.5, color = 'blue', label = 'frame 2 ideal output');
            plt.legend(); plt.draw(); plt.pause(0.001);


# TESTING ###########################################################################################################################################################################
# TESTING ###########################################################################################################################################################################
# TESTING ###########################################################################################################################################################################
# TESTING ###########################################################################################################################################################################
# TESTING ###########################################################################################################################################################################
# TESTING ###########################################################################################################################################################################
# TESTING ###########################################################################################################################################################################
# TESTING ###########################################################################################################################################################################
# TESTING ###########################################################################################################################################################################
# TESTING ###########################################################################################################################################################################
def test():
    # sending the data to the GPU ////////////////////////
    if torch.backends.mps.is_available(): device = 'mps'
    else: device = 'cpu'
    device = 'cpu'

    # loading the two models ######################
    autoencoder = torch.load(modelname).to(device);
    phaseencoder = torch.load(modelname_phase).to(device);
    magnitudeencoder_frame2 = torch.load(modelname_magnitude_frame2).to(device);
    phaseencoder_frame2 = torch.load(modelname_phase_frame2).to(device);

    # loading data ###########################################################################################
    inputdatatitle = '/Users/vrsreeganesh/Desktop/BUClasses/Thesis/Code/PythonFiles/DenoisingArtifacts/Models/test.mat'
    inputmat = scipy.io.loadmat(inputdatatitle); 
    currentinput = inputmat['speechnoisemixframes']; currentinput = torch.from_numpy(currentinput).float(); currentinput = torch.transpose(currentinput, 0, 1);
    ssinput = inputmat["SS_frames"]; ssinput = torch.from_numpy(ssinput).float(); ssinput = torch.transpose(ssinput, 0, 1);

    # running the data through the neural net ##################################
    currentinput_shifted = torch.from_numpy(frequencyshiftleftby4000(currentinput, order = 11)).float();
    ssinput_shifted = torch.from_numpy(frequencyshiftleftby4000(ssinput, order = 11)).float();

    netoutput = autoencoder(currentinput, ssinput);
    phaseoutput = phaseencoder(currentinput, ssinput);
    netoutput_magnitude_frame2 = magnitudeencoder_frame2(currentinput_shifted, ssinput_shifted);
    netoutput_phase_frame2 = phaseencoder_frame2(currentinput_shifted, ssinput_shifted);

    # flipping to make it symmetric ============================================================================================================
    netoutput = torch.cat((netoutput, torch.flip(netoutput[:, 1:-1], dims = [1])  ), dim = 1)
    phaseoutput = torch.cat((phaseoutput, -1*torch.flip(phaseoutput[:, 1:-1], dims = [1])  ), dim = 1)
    netoutput_magnitude_frame2 = torch.cat((netoutput_magnitude_frame2, torch.flip(netoutput_magnitude_frame2[:, 1:-1], dims = [1])  ), dim = 1)
    netoutput_phase_frame2 = torch.cat((netoutput_phase_frame2, -1*torch.flip(netoutput_phase_frame2[:, 1:-1], dims = [1])  ), dim = 1)

    # combining with the phase of the input =============================================
    netoutput = netoutput * torch.exp(1j*phaseoutput);
    netoutput_frame2 = netoutput_magnitude_frame2 * torch.exp(1j*netoutput_phase_frame2);

    # taking the ifft =========================================================================
    netoutput = torch.real(torch.fft.ifftn(netoutput, axis = 1, norm = "ortho"));
    netoutput_frame2 = torch.real(torch.fft.ifftn(netoutput_frame2, axis = 1, norm = "ortho"));

    # trimming since we're working with 512-bit fft ===============================
    netoutput = netoutput[:, 0:320]; netoutput_frame2 = netoutput_frame2[:, 0:320];

    # shifting the output and adding them together ===================================================
    netoutput = netoutput + torch.from_numpy(frequencyshiftrightby4000(netoutput_frame2.detach().numpy(), order = 11));

    # converting and saving results.########################################################################################################
    scipy.io.savemat('/Users/vrsreeganesh/Desktop/BUClasses/Thesis/Code/PythonFiles/DenoisingArtifacts/Models/testresults.mat', {"netoutput":netoutput.cpu().detach().numpy()});

# MAIN ################################################################################
# MAIN ################################################################################
# MAIN ################################################################################
# MAIN ################################################################################
# MAIN ################################################################################
# MAIN ################################################################################
# MAIN ################################################################################
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


