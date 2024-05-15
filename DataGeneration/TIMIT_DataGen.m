%{
==========================================================================================
Aim:
    This script goes through the TIMIT database and recursiveless starts creating dataset
    based on the associated script.
Note:
    Where the dataset is stored is written in the script that this script is calling. 
==========================================================================================
%}

%% Basic Setup
clc; clear; close all;

%% Listing all wav files
directory = '/Users/vrsreeganesh/Desktop/BUClasses/Thesis/Data/archive/data/TRAIN'; % path to the TRAIN folder in TIMIT database
wavFiles = transpose(findAllWavFiles(directory));
reorderorder = randperm(size(wavFiles,1));
wavFiles = wavFiles(reorderorder,:);

% replace this to the paths containing the noise-files you want to create dataset from
noiseFiles = ["/Users/vrsreeganesh/Desktop/BUClasses/Thesis/Code/Matlab/Data/cafe_noise.m4a",...
              "/Users/vrsreeganesh/Desktop/BUClasses/Thesis/Code/Matlab/Data/engine_noise.wav"];

%% Building Dataset
for wavfileindex = 1:length(wavFiles)
    
    % reading the current title
    currentspeechtitle = wavFiles(wavfileindex)

    % % choosing noise 
    % noisetitle = noiseFiles(mod(wavfileindex, length(noiseFiles))+1);

    for noiseindex = 1:length(noiseFiles)
        noisetitle  = noiseFiles(noiseindex)
        

        % reading the speech title
        for snrinput = -10:1:20
            % reading speech
            [cleanspeech, samplingfrequency] = audioread(currentspeechtitle); 
        
            % reading noise
            [noisesignal, samplingfrequency1] = audioread(noisetitle);
        
            % printing inputs and running the script
            disp(snrinput)
            T26_Saving_RawFrame_SSFrame_CleanSpeechFrame
    
            % clearing
            vars_to_keep = {'wavFiles', 'wavfileindex', 'currentspeechtitle', 'snrinput', 'noisetitle', 'noiseFiles'};
            all_vars = who;
            vars_to_clear = setdiff(all_vars, vars_to_keep);
            clear(vars_to_clear{:});
            
        end
    
        fprintf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n")
        fprintf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n")
        fprintf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n")
        fprintf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n")
        fprintf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n")
        fprintf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n")
        fprintf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n")
        fprintf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n")
        fprintf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n")
        fprintf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n")
        fprintf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n")
        fprintf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n")
        fprintf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n")
        fprintf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n")
        fprintf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n")
        disp("One Speech-CLIP is done. Quit here, if you wish")
        pause(5);
    end

end







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function fileList = findAllWavFiles(directory)
    % Get the list of files and directories in the current directory
    filesAndDirs = dir(directory);
    
    % Initialize an empty cell array to store file paths
    % fileList = {};
    fileList = [];
    
    % Loop through each item in the directory
    for i = 1:length(filesAndDirs)
        % Skip '.' and '..' directories
        if strcmp(filesAndDirs(i).name, '.') || strcmp(filesAndDirs(i).name, '..')
            continue;
        end
        
        % Check if the current item is a directory
        if filesAndDirs(i).isdir
            % Recursively call findAllWavFiles on the subdirectory
            subDir = fullfile(directory, filesAndDirs(i).name);
            subDirFiles = findAllWavFiles(subDir);
            
            % Add the files from the subdirectory to the list
            fileList = [fileList, subDirFiles];
        else
            % Check if the file has a ".wav" extension
            [~, ~, ext] = fileparts(filesAndDirs(i).name);
            if strcmp(ext, '.wav')
                % Add the file to the list
                fileList = [fileList, string(fullfile(directory, filesAndDirs(i).name))];
            end
        end
    end
end





