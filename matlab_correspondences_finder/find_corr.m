function correspondences = find_corr(images, interval, names, path, filename)
    % Input: list of images, and its length
    N = length(images);
    correspondences = {};   % This will contain the correspondences
    info = [];
    cond = [];
    kpi = 0;
    fileID = fopen(strcat(path,'\',filename,'.txt'),'w');

    while isempty(cond)

        cond = input("Insert any number to stop: ");
        if isempty(cond)

            nobs = 0;   % number of times we observed the current keypoint

            % Skip to some frame
            skip = input("Skip to this frame, enter for the first: ");
            if isempty(skip)
                skip=1;
            end

            skip = idivide(int8(skip-1),int8(interval))*interval+1; % so that we stay in our subsampling

            % Start selecting the point in the different frames
            while skip <= N

                % Some plotting
                imshow(images{skip})
                if kpi > 0
                    hold on;
                    for i=1:(kpi-double(nobs~=0))
                        for j = 1:size(correspondences{i},1)
                            n = correspondences{i}(j,2);
                            if n == skip
                                plot(correspondences{i}(j,3),correspondences{i}(j,4), 'r.', 'MarkerSize', 20)
                                text(correspondences{i}(j,3),correspondences{i}(j,4), string(correspondences{i}(j,1)), 'FontSize', 10, 'color', 'green');
                            end
                        end
                    end
                end

                % Skip or insert                       
                select = input("enter twice to skip by "+interval+", enter once and click to select keypoint, 1 to end this corresponcence search: ");
                if select==1
                    break
                end
                r = ginput(1);
                if isempty(select) && ~isempty(r)
                    x=r(1); y=r(2);
                    if nobs==0
                        kpi = kpi+1;            
                    end
                    nobs = nobs +1;
                    skip = double(skip);
                    info(nobs,:) = [kpi, skip, x, y];
                    name = names{skip};

                    % Add line on our txt file
                    fprintf(fileID,'%d ',kpi);
                    fprintf(fileID, '%s ',name);
                    fprintf(fileID, '%f %f\n',[x,y]);

                end
                skip = double(skip + interval);
                


                disp("# correspondences: "+kpi)
                if nobs>0
                    correspondences{kpi} = info;
                end

            end

            info = [];
            nobs = 0;

        end

    end

    fclose(fileID);

end
