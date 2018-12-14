%% module helps in visualizing the 
function visualization(neural_net,layer_name)

index = 0;
for i=1:length(neural_net.Layers)
    tf = strcmp(neural_net.Layers(i).Name,layer_name);
    if tf == 1
        name = neural_net.Layers(i).Name;
        index = i;
        break;
    end
end


fprintf(name);
switch name
    case 'conv1'
        channels = 1:32;
        
    case 'maxpool1'
        channels = 1:32;
    case 'avgpool1'
        channels = 1:32; 
    case 'conv2'
        channels = 1:16;
    case 'relu2'
        channels = 1:16;
    case 'fullconnect1'
        channels = 1:10;
        
    case 'sml1'
        channels = 1:10;
    
    otherwise
        disp('input incorrect')
end

Image_extraction = deepDreamImage(neural_net,index,channels,'PyramidLevels',1);
figure
visualize = imtile(Image_extraction,'ThumbnailSize',[64 64]);
imshow(visualize)
title(['Layer ',name,' Features'])
end

