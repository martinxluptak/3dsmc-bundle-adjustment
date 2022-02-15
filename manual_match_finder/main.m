%% Load dataset
disp('Loading the dataset')
[images, depths, image_timestamps, image_names, depth_timestamps, depth_names, N] = load_sequence('C:\Code\University\TUM\3D_scanning\Project\data\rgbd_dataset_freiburg1_xyz');
disp('Done');

%% Process groundtruth
gt_file = 'C:\Code\University\TUM\3D_scanning\Project\data\rgbd_dataset_freiburg1_xyz\groundtruth.txt';
[gt_timestamps, gt_names, poses] = process_gt(gt_file);
disp('Done');

%% Align ground truth and images
img_ts_vec = cell2mat(image_timestamps);
gt_ts_vec = cell2mat(gt_timestamps);

gt_ts_img_vec = interp1(gt_ts_vec, gt_ts_vec, img_ts_vec, 'nearest');

nn_poses = {};
gt_names_2 = {};
j=1;
for i=1:length(gt_ts_img_vec)
    nn_poses{j} = poses{gt_ts_vec==gt_ts_img_vec(i)};
    gt_names2{j} = gt_names{gt_ts_vec==gt_ts_img_vec(i)};
    j=j+1;
end

%% Subsample and see images
interval = input("Subsample by these many frames: ");
% for i=1:interval:N
%     imshow(images{i})
% end

%% Choose keypoints, write file
correspondences = find_corr(images, depths, nn_poses, interval, image_names, depth_names, gt_names2, 'C:\Code\University\TUM\3D_scanning\Project\data\rgbd_dataset_freiburg1_xyz', 'five_frames');
