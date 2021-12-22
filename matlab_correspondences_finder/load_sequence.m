function [sequence, names, N] = load_sequence(path_to_dataset_folder)
    % Return a vector of color images from the path to rgb.txt
    fid=fopen(path_to_dataset_folder+"\rgb.txt",'r');
     
    sequence = {};
    names = {};
    i=1;
    
    tline = fgetl(fid);
    while ischar(tline)
        tline = fgetl(fid);
        if length(tline)>=3 & tline(end-2:end)=='png'
            tline = extractAfter(tline, '/');
            I=imread(path_to_dataset_folder+"\rgb\"+tline);
            sequence{i} = I;
            names{i} = tline;
            i=i+1;
        end
    end
    
    N = i-1;

    fclose(fid);
end

