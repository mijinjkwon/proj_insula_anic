function mtSVMs_wOptNcv_insula_linear_apptVSaver_domain(task_id)

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

    results_mtSVMs_wOptNcv_insula_linear_apptVSaver_domain = struct('model_n', [], 'model', [], ...
        'resub_results', struct('label_resub', [], 'NegLoss_resub', [], 'PBScore_resub', [], 'label_true_train', []), ...
        'pred_results', struct('label_pred', [], 'NegLoss_pred', [], 'PBScore_pred', [], 'label_true_test', []));
    output_dir = '/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/ANiC/insula/output';

    % Load data and select study for the run

    dat = load('/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/ANiC/insula/data/dat_svm.mat','img_anic_4domains_all').img_anic_4domains_all;

    % Insula mask
    file_to_render = which('SPM8_insula_ribbon_LR.img'); % location of the insula mask
    insula_mask = fmri_data(file_to_render); % loading the insula mask as fmri_data
    insula_mask = preprocess(insula_mask, 'smooth', 3); % preprosessing the insula mask: smoothing
    insula_mask = threshold(insula_mask, [.02 Inf], 'raw-between'); % preprosessing the insula mask: thresholding; Q: why .02?

    % masking
    dat = apply_mask(dat, insula_mask);

    % scaling (z-score)
    dat = rescale(dat, 'zscoreimages');

    % Getting studies
    s_svm_all = load('/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/ANiC/insula/data/dat_svm.mat','s_svm_all').s_svm_all;

    % Main script
    
    tic
    
    % Getting task info
    tasks_per_job = 10

    printhdr(strcat('Task ID : ', string(task_id)))

    n = task_id;
    model_num_current = n;
    model_current = strcat('Model (w CV) for pairwise classification between aversive vs. appetitive', string(model_num_current));
    printhdr(strcat('Initiating ', model_current))
    
    % Studies for this loop
    s_current = s_svm_all{n};

    % === TRAINING DATA
    % Getting selected studies for this loop

    wh_studies_train = [];
    for i = [2, 3]
        wh_studies_train = [wh_studies_train s_current(i).s_train];
    end

    % Getting subjts who belong to the studies selected for training

    wh_subjects_train_tmp = ismember(dat.metadata_table.StudyNumber, wh_studies_train);
    wh_subjects_train = find(wh_subjects_train_tmp);

    % List the unique subdomains
    table(unique(dat.metadata_table.Subdomain(wh_subjects_train)))

    % Sample data that belongs to the selected studies
    dat_train = get_wh_image(dat, wh_subjects_train');
    label_train = dat_train.metadata_table.Domain; % true label

    % === TEST DATA
    wh_studies_test = [];
    for j = [2, 3]
        wh_studies_test = [wh_studies_test s_current(j).s_test];
    end

    % Getting subjts who belong to the studies selected for training

    wh_subjects_test_tmp = ismember(dat.metadata_table.StudyNumber, wh_studies_test);
    wh_subjects_test = find(wh_subjects_test_tmp);

    % List the unique subdomains
    table(unique(dat.metadata_table.Subdomain(wh_subjects_test)))

    % Sample data that belongs to the selected studies
    dat_test = get_wh_image(dat, wh_subjects_test');
    label_test = dat_test.metadata_table.Domain; % true label

    % === TRAINING MULTICLASS SVM MODEL
    X = dat_train.dat';
    Y = dat_train.metadata_table.Domain;
    classNames = {'Appetitive Process','Aversive Process'};
    tabulate(Y)

    % === Different linear classification options

    rng default

    t = templateLinear("Learner","svm", "Type", "classification");    
    Mdl = fitcecoc(X, Y, 'Learners', t, 'ClassNames', classNames, ...
    'OptimizeHyperparameters','Lambda', 'HyperparameterOptimizationOptions',struct('Kfold', 5, 'MaxObjectiveEvaluations', 5), ...
    'Verbose',2); % didn't work (still run onevsone coding scheme)

    % === Prediction on training data
    [label_resub, NegLoss_resub, PBScore_resub] = predict(Mdl, dat_train.dat');

    % === Prediction on training data
    [label_pred, NegLoss_pred, PBScore_pred] = predict(Mdl, dat_test.dat'); % predicted label


%     results_mtSVMs_wOptNcv_insula_linear_apptVSaver_domain(n).model_n = model_current;
    results_mtSVMs_wOptNcv_insula_linear_apptVSaver_domain(n).model = Mdl;
    results_mtSVMs_wOptNcv_insula_linear_apptVSaver_domain(n).resub_results.label_resub = label_resub;
    results_mtSVMs_wOptNcv_insula_linear_apptVSaver_domain(n).resub_results.NegLoss_resub = NegLoss_resub;
    results_mtSVMs_wOptNcv_insula_linear_apptVSaver_domain(n).resub_results.PBScore_resub = PBScore_resub;
    results_mtSVMs_wOptNcv_insula_linear_apptVSaver_domain(n).resub_results.label_true_train = dat_train.metadata_table.Domain;
    results_mtSVMs_wOptNcv_insula_linear_apptVSaver_domain(n).pred_results.label_pred = label_pred;
    results_mtSVMs_wOptNcv_insula_linear_apptVSaver_domain(n).pred_results.NegLoss_pred = NegLoss_pred;
    results_mtSVMs_wOptNcv_insula_linear_apptVSaver_domain(n).pred_results.PBScore_pred = PBScore_pred;
    results_mtSVMs_wOptNcv_insula_linear_apptVSaver_domain(n).pred_results.label_true_test = dat_test.metadata_table.Domain;

    % Results without model
    results_mtSVMs_wOptNcv_insula_linear_apptVSaver_domain_woMdl = rmfield(results_mtSVMs_wOptNcv_insula_linear_apptVSaver_domain, 'model')

    % Save the results
    
    % Just 
    output_file = sprintf('Results_wOptNcv_insula_linear_apptVSaver_domain_%d.mat', n); % Create filename with array task id
    output_file_woMdl = sprintf('woMdl_Results_wOptNcv_insula_linear_apptVSaver_domain_%d.mat', n); % Create filename with array task id

    current_output_dir = fullfile(output_dir, '082724_mtSVM_domain_linear_insula_apptVSaver');
    if ~exist(current_output_dir, 'dir')
        mkdir(current_output_dir);
    end
    save(fullfile(current_output_dir, output_file), 'results_mtSVMs_wOptNcv_insula_linear_apptVSaver_domain', '-v7.3') % whole
    save(fullfile(current_output_dir, output_file_woMdl), 'results_mtSVMs_wOptNcv_insula_linear_apptVSaver_domain_woMdl', '-v7.3') % without model
end