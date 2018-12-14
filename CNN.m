%function CNN(pool_type)

%clear all;

train_images = reshape((load_features('train-images.idx3-ubyte')),[28,28,1,60000]);
train_labels = ((load_labels('train-labels.idx1-ubyte')));


test_images = reshape((load_features('t10k-images.idx3-ubyte')),[28,28,1,10000]);
test_labels = ((load_labels('t10k-labels.idx1-ubyte')));


%% Code to generate cross validated train, test data

train_images_fold=zeros(28,28,1,6000,10);
train_labels_fold=zeros(6000,1,10);

for i=(1:10)
train_images_fold(:,:,:,:,i)=train_images(:,:,:,(6000*(i-1))+1:6000*i);
train_labels_fold(:,1,i)=train_labels((6000*(i-1))+1:6000*i,1);
end

train_images_data=zeros(28,28,1,54000,10);
train_labels_data=zeros(54000,1,10);
cross_valid_images=zeros(28,28,1,6000,10);
cross_valid_labels=zeros(6000,1,10);

for (i=1:10)
    cross_valid_images(:,:,:,:,i)=train_images_fold(:,:,:,:,i);
    cross_valid_labels(:,:,i)=train_labels_fold(:,:,i);
    k=1;
    for j=(1:10)
        if (j~=i)
           train_images_data (:,:,:,(6000*(k-1))+1:6000*k,i)=train_images_fold(:,:,:,:,j);
           train_labels_data ((6000*(k-1))+1:6000*k,1,i)=train_labels_fold(:,:,j);
           k=k+1;
        end
    end
end


%% Code to display first 100 training images
figure(1)
for i=1:100 
        subplot(10,10,i);
        imshow(train_images(:,:,:,i));
end

%% Code to generate the ANN layers

%% constructing the input layer with [28 28 1 ] representing the width, height, depth of the image
inputlayer = imageInputLayer([28 28 1],'DataAugmentation','none',...
    'Normalization','none','Name','input');

%% constructing the convolutional layer with filter_size = 4
%% number of filters = 32, stride = 1, zero-padding = 0
 convlayer1 = convolution2dLayer(4,32,'Stride',1,'Padding',0, ...
     'BiasLearnRateFactor',2,'NumChannels',1,...
     'WeightLearnRateFactor',2, 'WeightL2Factor',1,...
     'BiasL2Factor',1,'Name','conv1');
 
%% initializing weights and biases for the convolution filters 
 convlayer1.Weights = randn([4 4 1 32])*0.1;
 convlayer1.Bias = randn([1 1 32])*0.1;

 %% relu for non-linear activation
 relulayer1 = reluLayer('Name','relu1');
 %% Normalization for speeding up the training process 
localnormlayer1 = crossChannelNormalizationLayer(3,'Name',...
    'localnorm1','Alpha',0.0001,'Beta',0.75,'K',2);

%% Pooling layer(change the pool_type variable to 'MAXPOOL' for maxpooling)
%% pooling layer has a filter size = 3, striden = 3, Padding = 1
pool_type = 'AVGPOOL';
switch pool_type
    case 'MAXPOOL'
        maxpoollayer1 = maxPooling2dLayer(3,'Stride',3,'Name','maxpool1','Padding',1);
        pool_layer = maxpoollayer1;
    case 'AVGPOOL'
        avgpoollayer1 = averagePooling2dLayer(3,'Stride',3,'Name','avgpool1','Padding',1);
        pool_layer = avgpoollayer1;
end
 
%% dropout layer 
droplayer1 = dropoutLayer(0.35);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                  Layer 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% convolution layer with filter size = 3, number of filters = 16
%% stride = 1, padding = 0
convlayer2 = convolution2dLayer(3,16,'Stride',1, 'Padding',0,...
    'BiasLearnRateFactor',1,'NumChannels',32,...
    'WeightLearnRateFactor',1, 'WeightL2Factor',1,...
    'BiasL2Factor',1,'Name','conv2');

%7*7*16
%% initializing weights and biases for the 2nd convolution layer
convlayer2.Weights = randn([3 3 32 16])*0.0001;
convlayer2.Bias = randn([1 1 16])*0.00001;

relulayer2 = reluLayer('Name','relu2');

localnormlayer2 = crossChannelNormalizationLayer(3,'Name',...
    'localnorm2','Alpha',0.0001,'Beta',0.75,'K',2);

%maxpoollayer2 = maxPooling2dLayer(2,'Stride',2,'Name','maxpool1','Padding',1);

% dropout layer
droplayer2 = dropoutLayer(0.25);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                  Output Layers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% fully connected layer with only 1 layer(i.e output layer)
fullconnectlayer = fullyConnectedLayer(10,'WeightLearnRateFactor',1,...
    'BiasLearnRateFactor',1,'WeightL2Factor',1,'BiasL2Factor',1,...
    'Name','fullconnect1');
%% initializing weights and biases for input to output
fullconnectlayer.Weights = randn([10 784])*0.0001;
fullconnectlayer.Bias = randn([10 1])*0.0001+1;

%% constructing a softmax layer for generating probability values 
smlayer = softmaxLayer('Name','sml1');
%% classification
coutputlayer = classificationLayer('Name','coutput');


%% Code to define training parameters

%% performs stocastic gradient descent on CNN model with the learning rate= 0.04
options = trainingOptions('sgdm',...
      'LearnRateSchedule','piecewise',...
      'LearnRateDropFactor',0.75,... 
      'LearnRateDropPeriod',1,'L2Regularization',0.0001,... 
      'MaxEpochs',2,'Momentum',0.9,'Shuffle','once',... 
      'MiniBatchSize',30,'Verbose',1,...
      'InitialLearnRate',0.04);
  
 %% Code to make the network
    layers =[inputlayer, convlayer1,relulayer1,localnormlayer1, ...
       pool_layer,...
       droplayer1,convlayer2,relulayer2,localnormlayer2,...
       droplayer2,fullconnectlayer, smlayer, coutputlayer]; 
   
   
   %% Train the CNN
   train_im=zeros(28,28,1,54000);
   train_lb=categorical(zeros(54000,1));
   cross_valid_im=zeros(28,28,1,6000);
   cross_valid_lb=zeros(6000,1);
   %for (i=1:2)
   train_im=train_images_data(:,:,:,:,1);
   train_lb=categorical(train_labels_data(:,:,1));
   cross_valid_im=cross_valid_images(:,:,:,:,1);
   cross_valid_lb=categorical(cross_valid_labels(:,:,1));
   trainedNet = trainNetwork(train_im,train_lb,layers,options);

%% Training accuracy
[Y_train_pred,scores] = classify(trainedNet,train_im);
train_score = sum((Y_train_pred==train_lb))/numel(train_lb);

%% cross_validation_accuracy and number of errors
[Ypred,cross_val_scores] = classify(trainedNet,cross_valid_im);
cross_val_score = sum((Ypred==cross_valid_lb))/numel(cross_valid_lb);
errors = sum((Ypred~=cross_valid_lb));
   %end

[max,index]=max(cross_val_score);
%% function call to train the neural network
CNN_model=trainedNet; 
   
%% plotting the Neural network architecture
lgraph = layerGraph(CNN_model.Layers);
figure
plot(lgraph);