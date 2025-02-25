function mtSVMs_wOptNcv_1vO_insula_linear_subd(task_id)

    % For Discovery job

    % Toolbox
    addpath(genpath('/dartfs-hpc/rc/lab/C/CANlab/modules/MediationToolbox'));
    addpath(genpath('/dartfs-hpc/rc/lab/C/CANlab/modules/Neuroimaging_Pattern_Masks'));
    addpath(genpath('/dartfs-hpc/rc/lab/C/CANlab/modules/spm12'));
    addpath(genpath('/dartfs-hpc/rc/lab/C/CANlab/modules/CanlabCore'));
    addpath(genpath('/dartfs-hpc/rc/lab/C/CANlab/modules/CANlab_help_examples'));
    addpath(genpath('/dartfs-hpc/rc/lab/C/CANlab/modules/MasksPrivate'));
    addpath(genpath('/dartfs-hpc/rc/lab/C/CANlab/modules/RobustToolbox'));

    % Display helper functions: Called by later scripts
    dashes = '----------------------------------------------';
    printstr = @(dashes) disp(dashes);
    printhdr = @(str) fprintf('%s\n%s\n%s\n', dashes, str, dashes);
    
    % To save results

    results_mtSVMs_wOptNcv_1vO_insula_linear_subd = struct('model_n', [], 'model', [], ...
        'resub_results', struct('label_resub', [], 'NegLoss_resub', [], 'PBScore_resub', [], 'Posterior_resub', [], 'label_true_train', []), ...
        'pred_results', struct('label_pred', [], 'NegLoss_pred', [], 'PBScore_pred', [], 'Posterior_pred', [], 'label_true_test', []));
    output_dir = '/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/ANiC/insula/output';

    % Load data and select study for the run

    dat = load('/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/ANiC/insula/data/dat_svm.mat','img_anic_4domains_all').img_anic_4domains_all;

    % insula mask
    file_to_render = which('SPM8_insula_ribbon_LR.img'); % location of the insula mask
    insula_mask = fmri_data(file_to_render); % loading the insula mask as fmri_data
    insula_mask = preprocess(insula_mask, 'smooth', 3); % preprosessing the insula mask: smoothing
    insula_mask = threshold(insula_mask, [.02 Inf], 'raw-between'); % preprosessing the insula mask: thresholding; Q: why .02?

    % masking
    dat = apply_mask(dat, insula_mask);

    % scaling (z-score)
    dat = rescale(dat, 'zscoreimages');

    % Getting studies
    s_svm_all = load('/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/ANiC/insula/data/dat_sub_svm.mat','s_sub_svm_all').s_sub_svm_all;

    % Main script
    
    tic
    
    % Getting task info
    tasks_per_job = 10

    printhdr(strcat('Task ID : ', string(task_id)))

    n = task_id;
    model_num_current = n;
    model_current = strcat('Model (w CV) ', string(model_num_current));
    printhdr(strcat('Initiating ', model_current))

    % Studies for this loop

    s_current = s_svm_all{n};

    % === TRAINING DATA
    % Getting selected studies for this loop

    wh_studies_train = [];
    for i = 1:length(s_current)
        wh_studies_train = [wh_studies_train s_current(i).s_sub_train];
    end

    % Getting subjts who belong to the studies selected for training

    wh_subjects_train_tmp = ismember(dat.metadata_table.StudyNumber, wh_studies_train);
    wh_subjects_train = find(wh_subjects_train_tmp);

    % List the unique subdomains
    table(unique(dat.metadata_table.Subdomain(wh_subjects_train)));

    % Sample data that belongs to the selected studies
    dat_train = get_wh_image(dat, wh_subjects_train');
    label_train = dat_train.metadata_table.Subdomain; % true label

    % === TEST DATA
    wh_studies_test = [];
    for j = 1:length(s_current)
        wh_studies_test = [wh_studies_test s_current(j).s_sub_test];
    end

    % Getting subjts who belong to the studies selected for training

    wh_subjects_test_tmp = ismember(dat.metadata_table.StudyNumber, wh_studies_test);
    wh_subjects_test = find(wh_subjects_test_tmp);

    % List the unique subdomains
    table(unique(dat.metadata_table.Subdomain(wh_subjects_test)));

    % Sample data that belongs to the selected studies
    dat_test = get_wh_image(dat, wh_subjects_test');
    label_test = dat_test.metadata_table.Subdomain; % true label

    % === TRAINING MULTICLASS SVM MODEL

    X = dat_train.dat';
    Y = dat_train.metadata_table.Subdomain;
    classNames = cellstr(unique(dat.metadata_table.Subdomain, 'stable'));
    tabulate(Y)

%     t = templateSVM('Standardize',true, ...
%         'KernelFunction','gaussian');

    t = templateSVM('Standardize',true)

    Mdl = fitcecoc(X,Y, 'Coding','onevsall', 'Learners',t,'FitPosterior',true, 'ClassNames',classNames, ...
        'OptimizeHyperparameters','auto', 'HyperparameterOptimizationOptions',struct('Kfold', 5, 'MaxObjectiveEvaluations', 5), ...
        'Verbose',2); % didn't work (still run onevsone coding scheme)

    model_param_current = Mdl.ModelParameters.BinaryLearners

    % === Prediction on training data
    [label_resub, NegLoss_resub, PBScore_resub, Posterior_resub] = resubPredict(Mdl,'Verbose',1);

    % === Prediction on training data
    [label_pred, NegLoss_pred, PBScore_pred, Posterior_pred] = predict(Mdl, dat_test.dat'); % predicted label

    results_mtSVMs_wOptNcv_1vO_insula_linear_subd(n).model_n = model_current;
    results_mtSVMs_wOptNcv_1vO_insula_linear_subd(n).model = Mdl;
    results_mtSVMs_wOptNcv_1vO_insula_linear_subd(n).resub_results.label_resub = label_resub;
    results_mtSVMs_wOptNcv_1vO_insula_linear_subd(n).resub_results.NegLoss_resub = NegLoss_resub;
    results_mtSVMs_wOptNcv_1vO_insula_linear_subd(n).resub_results.PBScore_resub = PBScore_resub;
    results_mtSVMs_wOptNcv_1vO_insula_linear_subd(n).resub_results.Posterior_resub = Posterior_resub;
    results_mtSVMs_wOptNcv_1vO_insula_linear_subd(n).resub_results.label_true_train = dat_train.metadata_table.Subdomain;
    results_mtSVMs_wOptNcv_1vO_insula_linear_subd(n).pred_results.label_pred = label_pred;
    results_mtSVMs_wOptNcv_1vO_insula_linear_subd(n).pred_results.NegLoss_pred = NegLoss_pred;
    results_mtSVMs_wOptNcv_1vO_insula_linear_subd(n).pred_results.PBScore_pred = PBScore_pred;
    results_mtSVMs_wOptNcv_1vO_insula_linear_subd(n).pred_results.Posterior_pred = Posterior_pred;
    results_mtSVMs_wOptNcv_1vO_insula_linear_subd(n).pred_results.label_true_test = dat_test.metadata_table.Subdomain;

    % Results without model
    results_mtSVMs_wOptNcv_1vO_insula_linear_subd_woMdl = rmfield(results_mtSVMs_wOptNcv_1vO_insula_linear_subd, 'model')

    % Save the results
    output_file = sprintf('Results_wOptNcv_1vO_insula_linear_subd_%d.mat', n); % Create filename with array task id
    output_file_woMdl = sprintf('woMdl_Results_wOptNcv_1vO_insula_linear_subd_%d.mat', n); % Create filename with array task id

    current_output_dir = fullfile(output_dir, 'mtSVM_subd_linear')
    save(fullfile(current_output_dir, output_file), 'results_mtSVMs_wOptNcv_1vO_insula_linear_subd', '-v7.3') % whole
    save(fullfile(current_output_dir, output_file_woMdl), 'results_mtSVMs_wOptNcv_1vO_insula_linear_subd_woMdl', '-v7.3') % without model
end