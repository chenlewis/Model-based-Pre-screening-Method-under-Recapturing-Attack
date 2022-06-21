%训练部分
features = double(features);
R = randperm(numImages); 
num_train = round(0.8*numImages);%取80%为训练样本
featuresTrain = features(R(1:num_train),:);
[featuresTrain1,ps]=mapminmax(featuresTrain',0,1);%特征归一化
featuresTrain1=featuresTrain1';
labels_0 = double(labels_0);
labels_0 = labels_0';
labelsTrain = labels_0(R(1:num_train),:);
% name=name';
nameTrain = name(R(1:num_train),:);
% labels_train = labels_0(R(1:num_train),:);
modelTrain = libsvm_svmtrain(labelsTrain, featuresTrain1, '-t 2 -b 1');%训练
featuresVal = features(R(num_train+1:numImages),:);%取剩下的部分为验证样本
[featuresVal1,ps1]=mapminmax(featuresVal',0,1);
featuresVal1=featuresVal1';
labelsVal = labels_0(R(num_train+1:numImages),:);
nameVal = name(R(num_train+1:numImages),:);
[predictIndexVal,accuracyVal,scoreVal] = libsvm_svmpredict(labelsVal,featuresVal1,modelTrain,'-b 1');%验证结果


% features = double(features);
% 
% [featuresTrain1,ps]=mapminmax(features',0,1);  
% featuresTrain1=featuresTrain1';
% labels = double(labels);
% % [bestcvaccuracy,bestc,bestg]=SVMcgForClass(labelsTrain,featuresTrain1);
% % modelTrain = libsvm_svmtrain(labelsTrain, featuresTrain1, ['-c', num2str(bestc),'-g',num2str(bestg),'-t 2 -b 1']);
% modelTrain = libsvm_svmtrain(labels, featuresTrain1, '-t 2 -b 1');

% [featuresval1,ps1]=mapminmax(featuresval',0,1);
% featuresval1=featuresval1';
% [predictIndexVal,accuracyVal,scoreVal] = libsvm_svmpredict(labelsval,featuresval1,modelTrain,'-b 1');




% 测试部分
featureTest = double(featureTest);
featureTest1=mapminmax(featureTest',0,1);
featureTest1=featureTest1';
[predictIndextest,accuracytest, scoretest] = libsvm_svmpredict(labelsTest,featureTest1, modelTrain,'-b 1');
result_table = table(nameTest, scoretest(:,1));
writetable(result_table, '1.csv');