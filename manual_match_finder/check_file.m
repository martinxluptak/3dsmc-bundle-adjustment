%% Here we check that our correspondences are indeed good, by performing reprojection

intrinsics = [525.0 0 319.5;...
            0 525.0  239.5;...
            0,0,1];
        
fileID = fopen(strcat('C:\Code\University\TUM\3D_scanning\Project\data\rgbd_dataset_freiburg1_xyz\','five_frames.txt'),'r');

tline = fgetl(fileID);

index = [];

while ischar(tline)
    tline = fgetl(fileID);
    if length(tline)>=7 & tline(1)~='#'
        
        values = split(tline);
        
        if isempty(index) || str2double(values{1}) ~= index
            index = str2double(values{1});
            disp("New point! -------------------------------------");
            % Read the depth file
            D=imread('C:\Code\University\TUM\3D_scanning\Project\data\rgbd_dataset_freiburg1_xyz\'+"\depth\"+values{3}+".png");
            first = 1;
        end
        
        % Reprojection
        p_pix = [str2double(values{5}); str2double(values{6})];
        int_coord = ceil(flipud(p_pix));
        d = double(D(int_coord(1), int_coord(2)))/5000;
        q = cellfun(@str2double,values((end-3):end));
        q = [q(4), q(1), q(2), q(3)]; % w xyz
        
        % In the sense of B->W
        R = quat2rotm(q);
        t = cellfun(@str2double,values((end-6):(end-4)));        
        
        % Conversion
        est = R*inv(intrinsics)*[p_pix;1]*d+t;
        
        if first
            gt = est;
            first = 0;
        else
            disp(["Reprojection error: ",norm(gt-est)]);
        end
        
    end
    
end

fclose(fileID);
