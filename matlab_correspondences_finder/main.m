%% Load dataset
disp('Loading the dataset')
[images, names, N] = load_sequence('C:\Code\University\TUM\3D_scanning\Project\data\rgbd_dataset_freiburg1_xyz');
disp('Done')

%% Subsample and see images
interval = input("Subsample by these many frames: ");
% for i=1:interval:N
%     imshow(images{i})
% end

%% Choose keypoints, write file
correspondences = find_corr(images, interval, names, 'C:\Code\University\TUM\3D_scanning\Project\data\rgbd_dataset_freiburg1_xyz', 'five_frames');
