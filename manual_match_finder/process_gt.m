function [timestamps, names, poses] = process_gt(gt_file)
    fid=fopen(gt_file,'r');
    
    timestamps = {};
    names = {};
    poses = {};
    i = 0;
    
    disp('Loading the ground truth data')    
    tline = fgetl(fid);
    while ischar(tline)
        tline = fgetl(fid);
        if length(tline)>=7 & tline(1)~='#'
            i = i+1;
            values = split(tline);
            timestamps{i}= sscanf(values{1},"%f");
            names{i} = values{1};
            pose = values(2:end);
            pose = cellfun(@str2double,pose);
            poses{i} = pose;
        end
    end

    fclose(fid);
end

