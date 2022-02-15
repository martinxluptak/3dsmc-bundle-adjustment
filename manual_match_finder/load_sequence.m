function [images, depths, image_timestamps, image_names, depth_timestamps, depth_names, N] = load_sequence(path_to_dataset_folder)
    % Return a vector of color images from the path to rgb.txt
    fid=fopen(path_to_dataset_folder+"\rgb.txt",'r');
    fid2=fopen(path_to_dataset_folder+"\depth.txt",'r');

    images = {};
    depths = {};
    image_timestamps = {};
    image_names = {};
    depth_timestamps = {};
    depth_names = {};
        
    i=1;
    
    disp('Loading the images')    
    tline = fgetl(fid);
    while ischar(tline)
        tline = fgetl(fid);
        if length(tline)>=3 & tline(end-2:end)=='png'
            tline = extractAfter(tline, '/');
            I=imread(path_to_dataset_folder+"\rgb\"+tline);
            images{i} = I;
            image_timestamps{i} = sscanf(tline(1:end-4),"%f");  
            image_names{i} = tline(1:end-4);
            i=i+1;
        end
    end
    
    N = i-1;
    i=1;
    
    disp('Loading the depths')    
    tline = fgetl(fid2);
    while ischar(tline)
        tline = fgetl(fid2);
        if length(tline)>=3 & tline(end-2:end)=='png'
            tline = extractAfter(tline, '/');
            I=imread(path_to_dataset_folder+"\depth\"+tline);
            depths{i} = I;
            depth_timestamps{i} = sscanf(tline(1:end-4),"%f");
            depth_names{i} = tline(1:end-4);
            i=i+1;
        end
    end

    fclose(fid);  
    fclose(fid2); 
        
end

