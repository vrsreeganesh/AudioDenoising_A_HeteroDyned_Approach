%{
===========================================================================
# Running Inference on FrametoDFT metohd
Note:
    This script performs the following
        1. Generates noisy-clip at specified SNR
        2. Spectral-subtracts noisy-audio
        3. Saves the frame as matfiles
        4. Calls test method in python script
        5. Reads the results of python script
        6. Plays, saves and plots
===========================================================================
%}

%% Basic Setup
clc; clear; close all;
snrinput = -10;
% rng("default");

%% Reading clean speech audio
cleanspeech_cointoss = randi(3);
cleanspeech_cointoss = 2;
if cleanspeech_cointoss == 1
    [cleanspeech, samplingfrequency] = audioread("/Users/vrsreeganesh/Desktop/BUClasses/Thesis/Code/Matlab/Data/p234_002.wav"); 
elseif cleanspeech_cointoss == 2
    [cleanspeech, samplingfrequency] = audioread("/Users/vrsreeganesh/Desktop/BUClasses/Thesis/Data/archive/data/TRAIN/DR1/FCJF0/SA1.WAV.wav");
elseif cleanspeech_cointoss == 3
    [cleanspeech, samplingfrequency] = audioread("/Users/vrsreeganesh/Desktop/BUClasses/Thesis/Data/archive/data/TRAIN/DR1/MJEB1/SA1.WAV.wav");
end
cleanspeech = normaliseenergy(cleanspeech);             % function removes the DC component and then normalises it so that energy is unity
audioduration = length(cleanspeech)/samplingfrequency;  % calculating the total duration of the audio

%% Adding some zeros 
indicestoaddat = [1,18518, 21093, 23273, 34434, 36050, 45125, length(cleanspeech)];
temp00 = [];

%% adding a few seconds in front
numsecondstoadd = 0;
cleanspeech = [zeros(numsecondstoadd*samplingfrequency,1); cleanspeech]; % Adding a few seconds in front so that there is adequate DSI sequences to estimate noise from

%% Loading different noise-s
% [noisesignal, samplingfrequency1] = audioread("/Users/vrsreeganesh/Desktop/BUClasses/Thesis/Code/Matlab/Data/cafe_noise.m4a");
[noisesignal, samplingfrequency1] = audioread("/Users/vrsreeganesh/Desktop/BUClasses/Thesis/Code/Matlab/Data/engine_noise.wav");
% [noisesignal, samplingfrequency1] = audioread("/Users/vrsreeganesh/Desktop/BUClasses/Thesis/Code/Matlab/Data/rain_noise.wav");
noisesignal = noisesignal(:,1);

%% decimating the noise signal: 
% Note: This is cause the speech was taken from TIMIT database, which is
% 16Khz while the noise data was taken from various sources from the
% internet. The usual sampling rate is around 48KHz. Hence the decimation.
if samplingfrequency1 ~= samplingfrequency
    fprintf("> The sampling frequency of noise and input signal is different \n");
    decimationfactor = samplingfrequency1/samplingfrequency;
    noisesignal = decimate(noisesignal, decimationfactor);
end

%% Designing window 2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2
windowlength = 20e-3;                                               % time duration of window in seconds
windownumsamples = windowlength*samplingfrequency;                  % number of samples in window
shiftlength = 10e-3;                                                % time duration after which next window starts
shiftnumsamples = shiftlength*samplingfrequency;                    % number of samples between adjacent shifts
overlaplength = windownumsamples - shiftnumsamples;                 % number of overlapping samples
fftlength = 512;                                                    % length of the fft
win = transpose(sqrt(hann(windownumsamples, "periodic"))); 
win = win/norm(win);                                                % done to meet this particular condition (refer thesis)

%% Creating Mixture 3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3
% fixing up the lengths of speechsignal so that the numframes are right. 
var85 = ((length(cleanspeech) - windownumsamples)/shiftnumsamples) + 1;
var85 = ceil(var85);
var87 = (var85-1)*shiftnumsamples + windownumsamples;
var88 = var87 - length(cleanspeech);
cleanspeech = [cleanspeech; zeros([var88,1])];

% making/reading noise and normalising it. 
% snrinput = -21;                     % snr = 20log(rms(signal)/rms(noise))
noiseweight = 1/(exp(snrinput/10)); % the value by which the noise signal must be weighed before adding to signal
% noiseweight = 0; % This is to be used when we want zero noise in our speech-noise mixture. 

%% Preprocessing noise: trimming + normalising 
noisesignal = noisesignal(:,1);                             % just taking one channel
noisesignal = noisesignal(1:size(cleanspeech,1), 1);        % trimming to make length same as that of speech
noisesignal = noiseweight*normaliseenergy(noisesignal);     % normalizing and multiplying with noise-weight, obtained from input SNR

% adding noise and taking stft of mixture
speechnoisemix = cleanspeech + noisesignal;

% taking mySTFT
stftmatrix = mystft(speechnoisemix, win, shiftnumsamples, fftlength);          % stft of mixture
stftmatrix_cleanspeech = mystft(cleanspeech, win, shiftnumsamples, fftlength); % stft of cleanspeech
phaseofstft = exp(1i.*angle(stftmatrix));                                      % phase of mixture, we'll need this for reconstruction

%% Making Sequences From Speech Mixture 4-4-4-4-4-4-4-4-4-4-4-4-4-4-4-4-4-4

% parameters for sequencing
M = 9; % the number of frames in a sequence (obtained from thesis)
S = 5; % the number of frames between adjacent sequences (obtained from thesis)

% Here, we calculate the starting frame number for each sequence
startingpoints_forsequences = 1:S:size(stftmatrix,2)-M+1; % the last point is included btw

% This object helps us maintain its index when we distribute it to
% different sequences, since we share the same frame with more than one
% seqeunce. 
maintaining_frame_identity = zeros([1,M,length(startingpoints_forsequences)]); 

% Initializing matrix to store sequences
sequences = zeros([size(stftmatrix,1), M, length(startingpoints_forsequences)]); % this object contains each sequence as a slice of the tensor. That is, i-th seqeunce is sequences(:,:,i)
sequences_cleanspeech = zeros([size(stftmatrix,1), M, length(startingpoints_forsequences)]);

% Here, we pick frames from the stftmatrix and allocate them to the 3D
% matrix that holds the sequences. The maintaining_frame_identity object
% lets us remember its original index in the stft matrix. Its importance
% comes in when we query the frames down the lane. 
for i = 1:length(startingpoints_forsequences)
    sequences(:,:,i) = stftmatrix(:, startingpoints_forsequences(i):startingpoints_forsequences(i)+M-1);   % copying all the frames that belong to the i-th sequence to the matrix at the appropriate depth
    maintaining_frame_identity(:,:,i) = startingpoints_forsequences(i):startingpoints_forsequences(i)+M-1; % copying the indices (in the stft) of the frames to this object. This will be of use later. 
    sequences_cleanspeech(:,:,i) = stftmatrix_cleanspeech(:, startingpoints_forsequences(i):startingpoints_forsequences(i)+M-1); % the sequence structure but for clean speech. 
end

%% Setting up some matrices 5-5-5-5-5-5-5-5-5-5-5-5-5-5-5-5-5-5-5-5-5-5-5-5
% frequency values corresponding to each index. Used for subsetting later. 
frequencyvalues = transpose(linspace(0,(fftlength-1)*samplingfrequency/fftlength, fftlength));
% The starting frequency positions of the overlapping frequency bands (refer thesis)
overlappingfrequencybandstart = transpose(0:500:(samplingfrequency/2)-1000);
% The starting frequency positions of the non-overlapping frequency bands (refer thesis) 
nonoverlappingfrequencybandstart = transpose(0:1000:(samplingfrequency/2)-1000);
% alphavaluesacrosssequences = [X, number of sequences]
alpha_column_per_sequence = [];
% initializing alphavalues for frames in a sequence structure = [-, M, -]
alphavalues1 = zeros([length(overlappingfrequencybandstart), size(sequences,2), size(sequences,3)]);

%% Noise estimation related parameters 6-6-6-6-6-6-6-6-6-6-6-6-6-6-6-6-6-6
% calculating the start-time and end-time of the frames
framestarttimes = 0:shiftlength:audioduration - windowlength; % starting time of each frame
frameendtimes = framestarttimes + windowlength;               % end time of each frame.
% calculating the start-time and end-time of the sequences
sequencelength = M*windowlength;                                    % the duration of each seqeunce in seconds
sequenceshiftlength = S*windowlength;                               % the time-shift in start-points of two adjacent sequences
sequencestarttimes = 0:sequenceshiftlength:audioduration-sequencelength; % array containing start time of each sequence
sequenceendtimes = sequencestarttimes+sequencelength;               % array containing end time of each sequence

%% Some testing 7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7
% Loading Sidhanth's codebook
codebookname = '/Users/vrsreeganesh/Desktop/BUClasses/Thesis/FromProfessor/CODEBOOK.mat'; 
load(codebookname) % This contains a number of objects that he uses, not just the codebook

% % Calculating the VID score of the entire Mixture to verify if code is
% % working well. 
% scoreofcurrentsequence = FetchMyVIDScores(stftmatrix, codebookname, win, shiftnumsamples, fftlength, samplingfrequency);
% 
% % Plotting VID scores, speech noise mixture and so on. 
% figure(140);
% subplot(3,1,1); plot(scoreofcurrentsequence); xlim([0, length(scoreofcurrentsequence)-1]);
% subplot(3,1,2); plot(speechnoisemix); xlim([1, length(speechnoisemix)]);
% subplot(3,1,3); imagesc(abs(stftmatrix));

%% Calculating Subtraction Parameters and Estimating Noise 8-8-8-8-8-8-8-8

% noise-estimate related matrices
sequencescores = [];                    % stores the scores of sequences in the incoming audio
previousDSIestimates = [];              % History since the estimate is a linear combination of past estimates
DSIweights = [0.5, 0.25, 0.125, 0.125]; % the weights for the linear combination of noise estimates

% this will contain the noise-estimate to be used to subtract from each
% frame.
absnoisestft = [];

% Some matrices
scoreofcurrentsequencelist = []; % the vector that contains score of sequences for every score
everysequencesmoothed = []; % matrix that contains the smoothed sequences as columns

for i = 1:size(sequences,3) % going through each sequence
    % This object contains the alpha values for the bands with the starting
    % frequencies given by the object a few lines above. 
    alphavalues = [];
    current_sequenceSTFT = sequences(:,:,i); % slicing out the current sequences
    currentsequencesmoothed = mean(abs(current_sequenceSTFT).^2,2);
    everysequencesmoothed = [everysequencesmoothed, currentsequencesmoothed]; % energy smoothing, equation 6.7, Sidhanth's thesis
    % Scoring the current sequence 
    scoreofcurrentsequence = FetchMyVIDScores(current_sequenceSTFT, codebookname, win, shiftnumsamples, fftlength, samplingfrequency);
    scoreofcurrentsequencelist = [scoreofcurrentsequencelist, mean(scoreofcurrentsequence)]; % appending the score of current sequence to larger list
    % Choosing top DSI frames
    [~, sortedindices] = sort(scoreofcurrentsequencelist, 'descend'); % sorting the **VID SCORES** of every sequence, so that we can find the top speech-less frames
    topxpercentofDSIframes = 0.15; % the percentage we're looking for (thesis used 15)
    sequenceindicescontaininghighestscore = sortedindices(1:ceil( topxpercentofDSIframes*length(sortedindices) )); % this contains the indices that are in the top x-percent of the sequences-scores
    

    % Obtaining DSI estimate
    recentDSIIndices = sort(sequenceindicescontaininghighestscore, 'descend'); % sorting the **INDICES**. This is done so that the most recent smoothed DSI-sequence, is weighed the most 
    var00 = everysequencesmoothed(:, recentDSIIndices);
    var00weights = 2.^((0:-1:-size(var00,2)+1 ))/2;
    current_noisesequence = var00*transpose(fliplr(var00weights));      % note that the weights are flipped so that the temporally closest smoothed-sequences are weighed the highest
    current_noisesequence = current_noisesequence/sum(var00weights);    % this is done to normalise the gain of the weighed sum


    % current_noisesequence = ULTIMATENOISEESTIMATE;

    
    % Adding the noise-estimates to the matrix we'll be subtracting from
    % the stft of the input signal. 
    absnoisestft = [absnoisestft, repmat(current_noisesequence, [1, S])];
    

    % finding out the alpha values for the non-overlapping frequency bands
    for j = 1:length(nonoverlappingfrequencybandstart)
        % The indices gives us the frequency bins that are within the band that
        % we're currently considering. 
        frequencyindicesofinterest = (frequencyvalues >= nonoverlappingfrequencybandstart(j)).*(frequencyvalues < (nonoverlappingfrequencybandstart(j)+1000));
        % frequencyvalues(frequencyindicesofinterest==1, :) % just to see if the subsetting is happening the way it should be

        % subsetting the frequency bins corresponding to current frequency
        % band. And then calculating the MNR
        band0 = current_sequenceSTFT(frequencyindicesofinterest==1,:); % subsetting frequency bins
        band0 = abs(band0).^2; % taking the absolute value and squaring them
        band0 = sum(mean(band0,2)); % finding the energy
        
        % doing the same for the noise-estimate
        band1 = current_noisesequence(frequencyindicesofinterest==1,:); % subsetting frequency bins in the noise-band
        band1 = sum(band1); % Calculating noise-estimate
        
        % Calculating the mixture-to-noise ratio
        mnr = 10*log10(band0/band1); 
        alpha = 5 - (4/25)*(mnr+5); % verified
        alpha(alpha>5) = 5; 
        alpha(alpha<1) = 1; 
    
        % building the alpha values and band-start values
        alphavalues = [alphavalues; alpha];
    end
    
    % finding alpha values for overlapping frequency bands by averaging the
    % alpha values of adjacent non-overlapping frequency bands.
    alphavalues_foroverlappingfrequencybands_forthissequence = [];
    for j = 1:length(overlappingfrequencybandstart)
        frequencyindicesofinterest = (nonoverlappingfrequencybandstart >= overlappingfrequencybandstart(j)-500).*(nonoverlappingfrequencybandstart < (overlappingfrequencybandstart(j)+1000));
        % nonoverlappingfrequencybandstart(frequencyindicesofinterest==1)
        % frequencyvalues(frequencyindicesofinterest==1, :) % just to see if the subsetting is happening the way it should be
        bruh2 = mean(alphavalues(frequencyindicesofinterest==1));
        alphavalues_foroverlappingfrequencybands_forthissequence = [alphavalues_foroverlappingfrequencybands_forthissequence; bruh2];
    end
    
    % building the object we need
    alpha_column_per_sequence = [alpha_column_per_sequence, alphavalues_foroverlappingfrequencybands_forthissequence];    
    alphavalues1(:,:,i) = repmat(alphavalues_foroverlappingfrequencybands_forthissequence, [1, size(sequences,2)]);
    

end

% Updating the final noise estimate
absnoisestft = [absnoisestft, repmat(current_noisesequence, [1, (M-S)])];
% absnoisestft = repmat(ULTIMATENOISEESTIMATE, [1, size(absnoisestft)]);

%% Combining alpha values acrosss sequences 9-9-9-9-9-9-9-9-9-9-9-9-9-9-9-9
%{
    The alpha values for frames that belong to two sequences will have
    their alpha values from their parents being mean-ed. 
%}

alphavaluesacrossframes = [];
for i = 1:size(stftmatrix,2)
    
    % We start by maintaining frame number
    currentframenumber = i;

    % the alpha values contributed for that particular frame
    temp = alphavalues1(:, maintaining_frame_identity==currentframenumber);
    % size(temp,2)
    
    % if sum(isnan(mean(temp,2))==0)
    if sum(isnan(mean(temp,2)))==0
        alphavaluesacrossframes = [alphavaluesacrossframes, mean(temp,2)];
    end
end


%% Bringing Back the Sequence Structure 9A=9A=9A=9A=9A=9A=9A=9A=9A=9A=9A=9A
alphaSequenceAveraged = zeros([size(alphavaluesacrossframes,1),M, size(sequences,3)]);
for i = 1:size(alphaSequenceAveraged,3)
    alphaSequenceAveraged(:,:,i) = alphavaluesacrossframes(:, S*(i-1)+1:S*(i-1)+M);
end

absnoiseSequence = zeros([fftlength, M, size(sequences,3)]);
for i = 1:size(sequences,3)
    absnoiseSequence(:,:,i) = absnoisestft(:,S*(i-1)+1:S*(i-1)+M);
end

%% Performing band-wise spectral subtraction
% the absolute value of sequences
abssequences = abs(sequences);

% 
subabssequences = abssequences(1:end/2+1,:,:);
subabsnoiseSequence = absnoiseSequence(1:end/2+1,:,:);
subfrequencyvalues = frequencyvalues(1:end/2+1,:);  % just considering the frequency bins from 0 to pi (included)

% Going through each sequence
SSSequence = zeros([size(subabssequences,1), M, size(sequences,3)]);
previosfrequencyindices00 = ones([fftlength/2+1,1]);

aggressivenessparameter = 10.0;
for i = 1:size(sequences,3)

    % Going through each overlapping frequency bands
    for j = 1:length(overlappingfrequencybandstart)

        % frequency bins corresponding to current frequency band
        if j~=length(overlappingfrequencybandstart)
            frequencyindices00 = (subfrequencyvalues>=overlappingfrequencybandstart(j)).*(subfrequencyvalues<(overlappingfrequencybandstart(j) +1000));
        else
            frequencyindices00 = (subfrequencyvalues>=overlappingfrequencybandstart(j)).*(subfrequencyvalues<=(overlappingfrequencybandstart(j) +1000));
        end
        overlappedfrequencyindices00 = frequencyindices00.*previosfrequencyindices00;
        
        % alpha value for this band
        alphavalueforthisband = alphaSequenceAveraged(j,:, i);
        alphavalueforthisband = repmat(alphavalueforthisband, [sum(frequencyindices00), 1]);

        % subtracting
        SSforthisband = subabssequences(frequencyindices00==1,:,i).^2 - (aggressivenessparameter.*alphavalueforthisband).*(subabsnoiseSequence(frequencyindices00==1,:,i));

        % smoothing the overlapping parts and writing the non-overlapping
        % part
        if j~=1
            var341 = zeros([sum(overlappedfrequencyindices00==1), M, 2]); var341(:,:,1) = SSSequence(overlappedfrequencyindices00==1, :, i);
            var341(:,:,2) = SSforthisband(1:sum(overlappedfrequencyindices00==1),:); SSSequence(overlappedfrequencyindices00==1, :, i) = mean(var341,3);
            var350 = frequencyindices00 - overlappedfrequencyindices00; SSSequence(var350==1,:,i) = SSforthisband(sum(overlappedfrequencyindices00==1)+1:end,:);
            previosfrequencyindices00 = frequencyindices00;
        else
            SSSequence(frequencyindices00==1,:,i) = SSforthisband; previosfrequencyindices00 = frequencyindices00;
        end
    end
end

% flooring
beta = 2e-3; SSSequence(SSSequence<0) = beta.*(subabssequences(SSSequence<0)).^2;

% Smoothing across sequences
SSOutput = [];
for i = 1:max(maintaining_frame_identity(:))
    currentframenumber = i; var372 = (maintaining_frame_identity == currentframenumber);
    var373 = SSSequence(:, (maintaining_frame_identity == currentframenumber)); SSOutput = [SSOutput, mean(var373,2)];
end

% Reflecting
SSOutput = [SSOutput; flipud(SSOutput(2:end-1,:))];

%% Smoothing the spectra across frames 11=11=11=11=11=11=11=11=11=11=11=11
gamma = 0.2; SSOutputSmoothed = [];
accumulate00 = zeros([fftlength,1]);
for i = 1:size(SSOutput,2)
    accumulate00 = gamma*accumulate00 + (1-gamma)*SSOutput(:,i); SSOutputSmoothed = [SSOutputSmoothed, accumulate00];
end
spectralsubtraction = SSOutputSmoothed;

%% Multiplying with phase of the original mixture 12=12=12=12=12=12=12=12
spectralsubtraction = sqrt(spectralsubtraction).*phaseofstft(:, 1:size(spectralsubtraction,2));

%% Taking the standard ISTFT 13=13=13=13=13=13=13=13=13=13=13=13=13=13=13
result = istft3(spectralsubtraction,...
                win,...
                overlaplength,...
                fftlength);

%% normalising 14=14=14=14=14=14=14=14=14=14=14=14=14=14=14=14=14=14=14=14
result = normaliseenergy(result);

%% Plotting the two signals before and after denoising 15=15=15=15=15=15=15
figure(4);
subplot(3,1,1); plot(maxamplitudeone(speechnoisemix)); title("input audio + noise");
subplot(3,1,2); plot(maxamplitudeone(result)); title("spectral subtracted audio");

%% Breaking into Frame
speechnoisemix_frames = fSplitIntoFrames(speechnoisemix, windownumsamples, shiftnumsamples);
SS_frames = fSplitIntoFrames(result, windownumsamples, shiftnumsamples);
cleanspeechframes = fSplitIntoFrames(cleanspeech, windownumsamples, shiftnumsamples);

%% Trimming so that the number of columns are same
smallestnumberofcolumns = min([size(speechnoisemix_frames,2),...
                              size(SS_frames,2), ...
                              size(cleanspeechframes, 2)]);

speechnoisemix_frames = speechnoisemix_frames(:, 1:smallestnumberofcolumns);
SS_frames = SS_frames(:, 1:smallestnumberofcolumns);
cleanspeechframes = cleanspeechframes(:, 1:smallestnumberofcolumns);

%% Normalizing the Frames
[speechnoisemix_frames, speechnoisemix_frames_mean, speechnoisemix_frames_max] = normalisematrix(speechnoisemix_frames);
[SS_frames, SS_frames_mean, SS_frames_max] = normalisematrix(SS_frames);
[cleanspeechframes, cleanspeechframes_mean, cleanspeechframes_max] = normalisematrix(cleanspeechframes);

%% Running The NEURAL NET
% fetching the time
paramstring = "SNR="+string(snrinput)+"_"+"windowlength="+string(windowlength)+"ms_"; timeastring = clock;
timeasstring = string(timeastring(1))+"_"+string(timeastring(2))+"_"+string(timeastring(3))+"_"+string(timeastring(4))+"_"+string(timeastring(5))+"_"+string(timeastring(6));

% saving the fft of the speechnoise mix
speechnoisemixfft_path = "/Users/vrsreeganesh/Desktop/BUClasses/Thesis/Code/PythonFiles/DenoisingArtifacts/Models/";
speechnoisemixfft_title = speechnoisemixfft_path + "test.mat";
if exist(speechnoisemixfft_title, 'file') == 2
    delete(speechnoisemixfft_title);
    fprintf("Deleting previous instance of test.mat \n");
end
if exist(speechnoisemixfft_path+"testresults.mat", 'file') == 2
    delete(speechnoisemixfft_path+"testresults.mat");
    fprintf("Deleting previous instance of testresults.mat \n");
end
speechnoisemixframes = speechnoisemix_frames;
save(speechnoisemixfft_title, "speechnoisemixframes", "SS_frames", "cleanspeechframes");


!/Users/vrsreeganesh/anaconda3/bin/python FrameToDFTTraining.py test
pause(5);
load("/Users/vrsreeganesh/Desktop/BUClasses/Thesis/Code/PythonFiles/DenoisingArtifacts/Models/testresults.mat")

%% Evaluating the Result
% reconstructing netoutputfull --------------------------------------------
netoutput = real(transpose(netoutput));

% % denormalizing using the clean-speech characteristics
% netoutput = netoutput + repmat(cleanspeechframes_mean, [size(netoutput,1),1]);
% netoutput = netoutput .* cleanspeechframes_max;

netoutput = netoutput + repmat(SS_frames_mean, [size(netoutput,1),1]);
netoutput = netoutput .* SS_frames_max;

% netoutput = netoutput + repmat(speechnoisemix_frames_mean, [size(netoutput,1),1]);
% netoutput = netoutput .* speechnoisemix_frames_max;

netoutput = netoutput(1:windownumsamples,:);
netoutputfull = [transpose(netoutput(:,1))];
for i = 2:size(netoutput,2)
    currentslice = transpose(netoutput(:,i));
    netoutputfull = [netoutputfull(1, 1:end-shiftnumsamples), ...
                     (netoutputfull(1, end-shiftnumsamples+1:end)/2 + currentslice(1:shiftnumsamples))/2, ...
                     currentslice(shiftnumsamples+1:end)];
end

figure(4); subplot(3,1,3); plot(netoutputfull);

%% then playing
% fprintf("Playing first audio \n");
% sound(maxamplitudeone(speechnoisemix), samplingfrequency); pause(7);
% fprintf("Playing second audio \n");
% sound(maxamplitudeone(result), samplingfrequency); pause(5);
sound(maxamplitudeone(netoutputfull), samplingfrequency); pause(5);



%% Saving Data
writepath = "/path/to/where/you/want/to/store/results" + mfilename+"_Results/";
audiowrite(writepath + "OriginalInput_SNR="+string(snrinput)+".wav", ...
           maxamplitudeone(speechnoisemix), samplingfrequency)
audiowrite(writepath + "SSOutput_SNR="+string(snrinput)+".wav",...
           maxamplitudeone(result), samplingfrequency)
audiowrite(writepath + "NeuralNetOutput_SNR="+string(snrinput)+".wav", ...
           maxamplitudeone(netoutputfull), samplingfrequency)

fprintf("All Done \n")


%%

%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%
function [output, mean_matrix, max_matrix] = normalisematrix(inputmatrix)
    mean_matrix = mean(inputmatrix, 1);
    inputmatrix = inputmatrix - mean_matrix;
    max_matrix = repmat(max(abs(inputmatrix), [], 1), [size(inputmatrix,1), 1]);
    inputmatrix = inputmatrix./max_matrix;
    output = inputmatrix;
end
function SplitIntoFrames = fSplitIntoFrames(speechnoisemix, windownumsamples, shiftnumsamples)
    SplitIntoFrames = [];
    numframes = ((length(speechnoisemix) - windownumsamples)/shiftnumsamples) + 1;
    for i = 1:numframes
        SplitIntoFrames = [SplitIntoFrames, ...
                           (speechnoisemix((i-1)*shiftnumsamples + 1: (i-1)*shiftnumsamples + windownumsamples))];
    end
end
function output = normaliseaudio(temp)
    temp = temp - mean(temp);
    temp = temp/max(abs(temp));
    output = temp;
end
function output = maxamplitudeone(temp)
    temp = normaliseenergy(temp);
    output = temp/max(abs(temp));
end
function output = normaliseenergy(temp)

    % subtracting by mean so that the DC component is zero. 
    temp = temp - mean(temp);

    % the following division will result in the vector having unit norm
    temp = temp/norm(temp); 

    % next we multiply with the length so that it has constant power
    output = temp;
end
function vidscore = FetchMyVIDScores(inputaudio, codebookname, win, shiftnumsamples, fftlength, samplingfrequency)
    % Calculating VAD score
    % vadscore = fCODEBOOK_testing(inputaudio, ...
    %                              codebookname, ...
    %                              win, ...
    %                              shiftnumsamples, ...
    %                              fftlength, ...
    %                              samplingfrequency);

    vadscore = fCODEBOOK_testingSTFTInput(inputaudio, ...
                                            codebookname, ...
                                            win, ...
                                            shiftnumsamples, ...
                                            fftlength, ...
                                            samplingfrequency);
    
    % Calculating VID score
    vidscore = max(vadscore) - vadscore;                    
    vidscore = vidscore/max(vidscore);
end
function output = NormalizeEnergyAndAmplitude(temp)
    output = NormalizeToUnitEnergy(temp);
    output = output/max(output);
end 
function output = NormalizeToUnitEnergy(temp)

    % subtracting by mean so that the DC component is zero. 
    temp = temp - mean(temp);

    % the following division will result in the vector having unit norm
    temp = temp/norm(temp); 

    % next we multiply with the length so that it has constant power
    output = temp;
end
function [sigma_smooth, dist_best_match_smooth,  sigma , dist_best_match ] = fCODEBOOK_testingSTFTInput(signal,CODEBOOK_name,codebook_window,win_shift,fft_len,Fs)
   
    % Loading some codebook related 
    win_len=length(codebook_window);
    load(CODEBOOK_name,'-mat'); % contains the CODEBOOK and Q_CODEBOOK
    
    % CODEBOOK best match
    CODEBOOK=CODEBOOK_updated_norm;
    CODEBOOK_updated_norm=CODEBOOK;
    %CODEBOOK_updated_norm=CODEBOOK;
    y_C=CODEBOOK;
    y_C_norm=y_C;
    %y_C=CODEBOOK_updated_norm; %this will be the codebook after adding input excitation
    %if Q_CODEBOOK(i)=NaN then the ith frame of CODEBOOK is unvoiced
    %% Normalized STES of input signal (L2 normalization)
    % sig_frame=v_enframe(signal,codebook_window,win_shift,Fs); % windowing at 20 ms sqrt hanning window
    % for j=1:size(sig_frame,1)
    %     frame_fft=(fft(sig_frame(j,:),fft_len));
    %     y_STES(j,:)=abs(frame_fft/win_len).^2;
    %     y_STES_norm(j,:)=y_STES(j,:)./norm(y_STES(j,:)); % L2 norm
    % end
    % 
    % y_STES = abs(signal./win_len).^2;
    % y_STES_norm = y_STES./vecnorm(y_STES);
    % 
    % 
    % y_STES=y_STES'; %each STES frame is a columns
    % y_STES_norm=y_STES_norm'; %each normalized STES frame is a column

    y_STES = abs(signal./win_len).^2;
    y_STES_norm = y_STES./vecnorm(y_STES);
    
    for i=1:size(y_STES,2)
        STES_frame=y_STES(:,i);
        STES_frame_norm=y_STES_norm(:,i);
        sig_i=(sum(STES_frame)/9)*[0:9]; %N_sig-1=9
     
        for j=1:size(CODEBOOK_updated_norm,2)
            
            if STES_frame == zeros(size(STES_frame))  %silence frame
                dist_best_match(i)=0;
                    break
            end
            
           if ~isnan(Q_CODEBOOK(j))
               y_C(:,j)= fCepstralLiftering( STES_frame,CODEBOOK(:,j),Q_CODEBOOK(j));
           end
           
           y_C_norm(:,j)=y_C(:,j)./norm(y_C(:,j));
           dist_match(j)=sum(abs(STES_frame_norm-y_C_norm(:,j)));
           %dist_match(j)=sum(abs(STES_frame_norm-y_C(:,j)));
           dist_match(j)=dist_match(j)./fft_len;
        end
    
        %save 'min(dist_best_match)' 
        if STES_frame ~= zeros(size(STES_frame))
            dist_best_match(i)=min(dist_match);
        end

        if STES_frame ~= zeros(size(STES_frame))  %if its not a silence frame  
            C_opt_index=find(dist_match==min(dist_match));
            C_opt_index=min(C_opt_index);
            %x_C_opt=y_C(:,C_opt_index);
            x_C_opt=y_C_norm(:,C_opt_index);
            
            for k=1:10
                dist_sig(k)=sum(abs(STES_frame-sig_i(k)*x_C_opt));
            end
            
            % FIND THE SPEECH GAIN
            sig_index=find(dist_sig==min(dist_sig));
            sigma(i)=sig_i(sig_index);
        else
            sigma(i)=0;
        end
    end
    
    %Smoothing the Gain(eq 13)
    sigma_smooth=sigma;
    for i=2:length(sigma)   %sig_smooth 
        
        if sigma(i)>=sigma_smooth(i-1)
        alpha=0.8;
        else
        alpha=0.91;
        end
        
        sigma_smooth(i)=(alpha*sqrt(sigma_smooth(i-1))+(1-alpha)*sqrt(sigma(i)))^2; %eq 13
    end
    
    % Also smooth Best matching distance
    dist_best_match_smooth=dist_best_match;
    for i=2:length(dist_best_match)   %sig_smooth 
        
        if dist_best_match(i)>=dist_best_match_smooth(i-1)
        alpha=0.8;
        else
        alpha=0.91;
        end
        
        dist_best_match_smooth(i)=(alpha*sqrt(dist_best_match_smooth(i-1))+(1-alpha)*sqrt(dist_best_match(i)))^2; %eq 13
    end


%sigma(end+1)=sigma(end);
%sigma_smooth(end+1)=sigma_smooth(end);
end
function [sigma_smooth, dist_best_match_smooth,  sigma , dist_best_match ] = fCODEBOOK_testing(signal,CODEBOOK_name,codebook_window,win_shift,fft_len,Fs)
    %PS: in this code 'hop' length is added i.e WSHIFT :
    
    %Returns a feature 'sigma' every 20ms @hop_len overlapp using a SPEECH CODEBOOK 
    %Although every 20ms frame results in a gain (sigma) the SHIFT can be variable
    
    %hop_len=hop_len*Fs; % What window shift you want to use
    %fft_len=512;
    win_len=length(codebook_window);
    load(CODEBOOK_name,'-mat'); % contains the CODEBOOK and Q_CODEBOOK
    
    % CODEBOOK best match
    CODEBOOK=CODEBOOK_updated_norm;
    CODEBOOK_updated_norm=CODEBOOK;
    %CODEBOOK_updated_norm=CODEBOOK;
    y_C=CODEBOOK;
    y_C_norm=y_C;
    %y_C=CODEBOOK_updated_norm; %this will be the codebook after adding input excitation
    %if Q_CODEBOOK(i)=NaN then the ith frame of CODEBOOK is unvoiced
    %% Normalized STES of input signal (L2 normalization)
    sig_frame=v_enframe(signal,codebook_window,win_shift,Fs); % windowing at 20 ms sqrt hanning window
    for j=1:size(sig_frame,1)
        frame_fft=(fft(sig_frame(j,:),fft_len));
        y_STES(j,:)=abs(frame_fft/win_len).^2;
        y_STES_norm(j,:)=y_STES(j,:)./norm(y_STES(j,:)); % L2 norm
    end
    y_STES=y_STES'; %each STES frame is a columns
    y_STES_norm=y_STES_norm'; %each normalized STES frame is a column
    
    for i=1:size(y_STES,2)
     STES_frame=y_STES(:,i);
     STES_frame_norm=y_STES_norm(:,i);
     sig_i=(sum(STES_frame)/9)*[0:9]; %N_sig-1=9
     
    for j=1:size(CODEBOOK_updated_norm,2)
        
        if STES_frame == zeros(size(STES_frame))  %silence frame
            dist_best_match(i)=0;
                break
        end
        
       if ~isnan(Q_CODEBOOK(j))
           y_C(:,j)= fCepstralLiftering( STES_frame,CODEBOOK(:,j),Q_CODEBOOK(j));
       end
       
       y_C_norm(:,j)=y_C(:,j)./norm(y_C(:,j));
       dist_match(j)=sum(abs(STES_frame_norm-y_C_norm(:,j)));
       %dist_match(j)=sum(abs(STES_frame_norm-y_C(:,j)));
       dist_match(j)=dist_match(j)./fft_len;
    end
    
    %save 'min(dist_best_match)' 
    if STES_frame ~= zeros(size(STES_frame))
        dist_best_match(i)=min(dist_match);
    end
    if STES_frame ~= zeros(size(STES_frame))  %if its not a silence frame  
        C_opt_index=find(dist_match==min(dist_match));
        C_opt_index=min(C_opt_index);
        %x_C_opt=y_C(:,C_opt_index);
        x_C_opt=y_C_norm(:,C_opt_index);
        
    for k=1:10
        dist_sig(k)=sum(abs(STES_frame-sig_i(k)*x_C_opt));
    end
    
    % FIND THE SPEECH GAIN
    sig_index=find(dist_sig==min(dist_sig));
    sigma(i)=sig_i(sig_index);
    else
        sigma(i)=0;
    end
    end
    
    %Smoothing the Gain(eq 13)
    sigma_smooth=sigma;
    for i=2:length(sigma)   %sig_smooth 
        
        if sigma(i)>=sigma_smooth(i-1)
        alpha=0.8;
        else
        alpha=0.91;
        end
        
        sigma_smooth(i)=(alpha*sqrt(sigma_smooth(i-1))+(1-alpha)*sqrt(sigma(i)))^2; %eq 13
    end
    
    % Also smooth Best matching distance
    dist_best_match_smooth=dist_best_match;
    for i=2:length(dist_best_match)   %sig_smooth 
        
        if dist_best_match(i)>=dist_best_match_smooth(i-1)
        alpha=0.8;
        else
        alpha=0.91;
        end
        
        dist_best_match_smooth(i)=(alpha*sqrt(dist_best_match_smooth(i-1))+(1-alpha)*sqrt(dist_best_match(i)))^2; %eq 13
    end


%sigma(end+1)=sigma(end);
%sigma_smooth(end+1)=sigma_smooth(end);
end
function [ liftered_frame ] = fCepstralLiftering( input,CODEBOOK,q )
    % Liftering the STES of input frame with a liftered CODEBOOK voiced frame 
    % in the CODEBOOK
    % This is used in the testing stage of the CODEBOOK
    
    % 
    M=length(input); %fft length
    input_Cep=log(input);
    input_Cep_ifft=0.5 * ifft(input_Cep); % eq: 4 for input
    
    CODEBOOK_Cep=log(CODEBOOK);
    CODEBOOK_Cep_ifft=0.5 * ifft(CODEBOOK_Cep); % eq: 4 for codebook
    
    CODEBOOK_Cep_ifft(q+1:M-q+1)=input_Cep_ifft(q+1:M-q+1); % adding input excitation to codebook
    
    
    CODEBOOK_Cep_ifft_fft=real(fft(CODEBOOK_Cep_ifft));
    CODEBOOK_Cep_ifft_fft_exp=exp(2*CODEBOOK_Cep_ifft_fft); %inverse cepstrum
    
    c=CODEBOOK_Cep_ifft_fft_exp;
    
    %normalizing the matched codebook (divide by l1 norm)
    norm_c=sum(c);norm_c=repmat(norm_c,[size(c,1),1]);
    c_norm=c./norm_c;
    
    liftered_frame=c_norm;

end
function [f,t,w]=v_enframe(x,win,hop,m,fs)
%V_ENFRAME split signal up into (overlapping) frames: one per row. [F,T]=(X,WIN,HOP)
%
% Usage:  (1) f=v_enframe(x,n)                          % split into frames of length n
%         (2) f=v_enframe(x,hamming(n,'periodic'),n/4)  % use a 75% overlapped Hamming window of length n
%         (3) calculate spectrogram in units of power per Hz
%
%               W=hamming(NW);                      % analysis window (NW = fft length)
%               P=v_enframe(S,W,HOP,'sdp',FS);        % computer first half of PSD (HOP = frame increment in samples)
%
%         (3) frequency domain frame-based processing:
%
%               S=...;                              % input signal
%               OV=2;                               % overlap factor of 2 (4 is also often used)
%               NW=160;                             % DFT window length
%               W=sqrt(hamming(NW,'periodic'));     % omit sqrt if OV=4
%               [F,T,WS]=v_enframe(S,W,1/OV,'fa');    % do STFT: one row per time frame, +ve frequencies only
%               ... process frames ...
%               X=v_overlapadd(v_irfft(F,NW,2),WS,HOP); % reconstitute the time waveform with scaled window (omit "X=" to plot waveform)
%
%  Inputs:   x    input signal
%          win    window or window length in samples
%          hop    frame increment or hop in samples or fraction of window [window length]
%            m    mode input:
%                  'z'  zero pad to fill up final frame
%                  'r'  reflect last few samples for final frame
%                  'A'  calculate the t output as the centre of mass
%                  'E'  calculate the t output as the centre of energy
%                  'f'  perform a 1-sided dft on each frame (like v_rfft)
%                  'F'  perform a 2-sided dft on each frame using fft
%                  'p'  calculate the 1-sided power/energy spectrum of each frame
%                  'P'  calculate the 2-sided power/energy spectrum of each frame
%                  'a'  scale window to give unity gain with overlap-add
%                  's'  scale window so that power is preserved: sum(mean(v_enframe(x,win,hop,'sp'),1))=mean(x.^2)
%                  'S'  scale window so that total energy is preserved: sum(sum(v_enframe(x,win,hop,'Sp')))=sum(x.^2)
%                  'd'  make options 's' and 'S' give power/energy per Hz: sum(mean(v_enframe(x,win,hop,'sp'),1))*fs/length(win)=mean(x.^2)
%           fs    sample frequency (only needed for 'd' option) [1]
%
% Outputs:   f    enframed data - one frame per row
%            t    fractional time in samples at the centre of each frame
%                 with the first sample being 1.
%            w    window function used
%
% By default, the number of frames will be rounded down to the nearest
% integer and the last few samples of x() will be ignored unless its length
% is lw more than a multiple of hop. If the 'z' or 'r' options are given,
% the number of frame will instead be rounded up and no samples will be ignored.
%

% Bugs/Suggestions:
%  (1) Possible additional mode options:
%        'u'  modify window for first and last few frames to ensure WOLA
%        'a'  normalize window to give a mean of unity after overlaps
%        'e'  normalize window to give an energy of unity after overlaps
%        'wm' use Hamming window
%        'wn' use Hanning window
%        'x'  hoplude all frames that hoplude any of the x samples

%	   Copyright (C) Mike Brookes 1997-2014
%      Version: $Id: v_enframe.m 10865 2018-09-21 17:22:45Z dmb $
%
%   VOICEBOX is a MATLAB toolbox for speech processing.
%   Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This program is free software; you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation; either version 2 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You can obtain a copy of the GNU General Public License from
%   http://www.gnu.org/copyleft/gpl.html or by writing to
%   Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    nx=length(x(:));
    if nargin<2 || isempty(win)
        win=nx;
    end
    if nargin<4 || isempty(m)
        m='';
    end
    nwin=length(win);
    if nwin == 1
        lw = win;
        w = ones(1,lw);
    else
        lw = nwin;
        w = win(:).';
    end
    if (nargin < 3) || isempty(hop)
        hop = lw; % if no hop given, make non-overlapping
    elseif hop<1
        hop=lw*hop;
    end
    if any(m=='a')
        w=w*sqrt(hop/sum(w.^2)); % scale to give unity gain for overlap-add
    elseif any(m=='s')
        w=w/sqrt(w*w'*lw);
    elseif any(m=='S')
        w=w/sqrt(w*w'*lw/hop);
    end
    if any(m=='d') % scale to give power/energy densities
        if nargin<5 || isempty(fs)
            w=w*sqrt(lw);
        else
            w=w*sqrt(lw/fs);
        end
    end
    nli=nx-lw+hop;
    nf = max(fix(nli/hop),0);   % number of full frames
    na=nli-hop*nf+(nf==0)*(lw-hop);       % number of samples left over
    fx=nargin>3 && (any(m=='z') || any(m=='r')) && na>0; % need an extra row
    f=zeros(nf+fx,lw);
    indf= hop*(0:(nf-1)).';
    inds = (1:lw);
    if fx
        f(1:nf,:) = x(indf(:,ones(1,lw))+inds(ones(nf,1),:));
        if any(m=='r')
            ix=1+mod(nf*hop:nf*hop+lw-1,2*nx);
            f(nf+1,:)=x(ix+(ix>nx).*(2*nx+1-2*ix));
        else
            f(nf+1,1:nx-nf*hop)=x(1+nf*hop:nx);
        end
        nf=size(f,1);
    else
        f(:) = x(indf(:,ones(1,lw))+inds(ones(nf,1),:));
    end
    if (nwin > 1)   % if we have a non-unity window
        f = f .* w(ones(nf,1),:);
    end
    if any(lower(m)=='p') % 'pP' = calculate the power spectrum
        f=fft(f,[],2);
        f=real(f.*conj(f));
        if any(m=='p')
            imx=fix((lw+1)/2); % highest replicated frequency
            f(:,2:imx)=f(:,2:imx)+f(:,lw:-1:lw-imx+2);
            f=f(:,1:fix(lw/2)+1);
        end
    elseif any(lower(m)=='f') % 'fF' = take the DFT
        f=fft(f,[],2);
        if any(m=='f')
            f=f(:,1:fix(lw/2)+1);
        end
    end
    if nargout>1
        if any(m=='E')
            t0=sum((1:lw).*w.^2)/sum(w.^2);
        elseif any(m=='A')
            t0=sum((1:lw).*w)/sum(w);
        else
            t0=(1+lw)/2;
        end
        t=t0+hop*(0:(nf-1)).';
    end


end



