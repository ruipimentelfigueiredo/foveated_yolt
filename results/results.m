close all
clear all
addpath('export_fig');

gt_folder='../dataset/gt/';
detections_file='../dataset/detections/raw_bbox_parse.txt';
images_folder='../dataset/images/';

% set to true to check detections
view_detections=true;

% parameters
detections_resolution=227;
sigmas_number=5;
thresholds_number=5;
images_number=100;
overlap_correct=0.5;
top_k=5;

% get ground truth
gt=parse_ground_truth(gt_folder,images_number);

% get detections
[sigmas,threshs,classes,scores,detections]=parse_detections(...
    sigmas_number,...
    thresholds_number,...
    images_number,...
    detections_file);

% view images

if view_detections
    for i=1:images_number
        figure(i)
        imshow(strcat(images_folder,gt(i).filename))
        hold on
        for g=1:size(gt(i).bboxes,1)
            % gt bbox
            gt_bbox=gt(i).bboxes(g,:);
            rectangle('Position',...
                gt_bbox,...
                'EdgeColor',...
                [0 1 0],...
                'LineWidth',...
                3);
        end
        hold off
    end
end

% check overlaps
overlaps=[];
for s=1:sigmas_number
    for t=1:thresholds_number
        for i=1:images_number
            gt_size=gt(i).size;
            aspect_ratio_x=gt_size(1)/detections_resolution;
            aspect_ratio_y=gt_size(2)/detections_resolution;
            % for each detection (of the 5)
            for j=1:top_k
                
                % check overlaps for each ground truth bounding box
                overlaps(s,t,i,j).overlap=zeros(size(gt(i).bboxes,1),1);
                
                for g=1:size(gt(i).bboxes,1)
                    % gt bbox
                    gt_bbox=gt(i).bboxes(g,:);
                    
                    % scale detections
                    detection=reshape(detections(s,t,i,j,:),1,4);
                    detection(1)=detection(1)*aspect_ratio_x;
                    detection(2)=detection(2)*aspect_ratio_y;
                    detection(3)=detection(3)*aspect_ratio_x;
                    detection(4)=detection(4)*aspect_ratio_y;
                    overlaps(s,t,i,j).overlap(g)=bboxOverlapRatio(gt_bbox,detection);
                    
                end
            end
        end
    end
end

% consider only the maximum overlap over ground truths
max_overlap=zeros(sigmas_number,thresholds_number, images_number,top_k);
for s=1:sigmas_number
    for t=1:thresholds_number
        for i=1:images_number
            for j=1:top_k
                max_overlap(s,t,i,j)=max(overlaps(s,t,i,j).overlap);
            end
        end
    end
end

% consider only the maximum overlap over all 5 detections
max_overlap=max(max_overlap,[],4);

% evaluate if the detections were good
max_overlap(max_overlap>=overlap_correct)=1;
max_overlap(max_overlap<overlap_correct)=0;

% compute detection rate
detection_rate=sum(max_overlap,3)/size(max_overlap,3);

%% plots

% fix one threshold and plot all sigmas
thresh_index=1;

figure(2)
fontsize=15;
set(gcf, 'Color', [1,1,1]);
plot(sigmas,100*detection_rate(:,thresh_index))
xlabel('$\sigma$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Localization Error (%)','Interpreter','LaTex','FontSize',fontsize);
ylim([0 100])

export_fig localization_error_sigma -pdf

% fix one sigma and plot all saliency thresholds
sigma_index=1;

figure(3)
fontsize=15;
set(gcf, 'Color', [1,1,1]);
plot(threshs,100*detection_rate(sigma_index,:))
xlabel('$th$','Interpreter','LaTex','FontSize',fontsize);
ylabel('Localization Error (%)','Interpreter','LaTex','FontSize',fontsize);
ylim([0 100])

export_fig localization_error_th -pdf


