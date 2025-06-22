import h5py
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
import lcpfn
from lcpfn import bar_distribution, encoders, train
from lcpfn import train as lctrain
from sklearn.metrics import root_mean_squared_log_error
import pandas as pd
import seaborn as sns

### hyperparameter
OPENML_ID = {0: '3', 1: '6', 2: '11', 3: '12', 4: '13', 5: '14', 6: '15', 7: '16', 8: '18', 9: '21', 10: '22', 11: '23',
             12: '24', 13: '26', 14: '28', 15: '29', 16: '30', 17: '31', 18: '32', 19: '36', 20: '37', 21: '38',
             22: '44', 23: '46', 24: '50', 25: '54', 26: '55', 27: '57', 28: '60', 29: '61', 30: '151', 31: '179',
             32: '180', 33: '181', 34: '182', 35: '184', 36: '185', 37: '188', 38: '201', 39: '273', 40: '293',
             41: '299', 42: '300', 43: '307', 44: '336', 45: '346', 46: '351', 47: '354', 48: '357', 49: '380',
             50: '389', 51: '390', 52: '391', 53: '392', 54: '393', 55: '395', 56: '396', 57: '398', 58: '399',
             59: '401', 60: '446', 61: '458', 62: '469', 63: '554', 64: '679', 65: '715', 66: '718', 67: '720',
             68: '722', 69: '723', 70: '727', 71: '728', 72: '734', 73: '735', 74: '737', 75: '740', 76: '741',
             77: '743', 78: '751', 79: '752', 80: '761', 81: '772', 82: '797', 83: '799', 84: '803', 85: '806',
             86: '807', 87: '813', 88: '816', 89: '819', 90: '821', 91: '822', 92: '823', 93: '833', 94: '837',
             95: '843', 96: '845', 97: '846', 98: '847', 99: '849', 100: '866', 101: '871', 102: '881', 103: '897',
             104: '901', 105: '903', 106: '904', 107: '910', 108: '912', 109: '913', 110: '914', 111: '917', 112: '923',
             113: '930', 114: '934', 115: '953', 116: '958', 117: '959', 118: '962', 119: '966', 120: '971', 121: '976',
             122: '977', 123: '978', 124: '979', 125: '980', 126: '991', 127: '993', 128: '995', 129: '1000',
             130: '1002', 131: '1018', 132: '1019', 133: '1020', 134: '1021', 135: '1036', 136: '1040', 137: '1041',
             138: '1042', 139: '1049', 140: '1050', 141: '1053', 142: '1056', 143: '1063', 144: '1067', 145: '1068',
             146: '1069', 147: '1083', 148: '1084', 149: '1085', 150: '1086', 151: '1087', 152: '1088', 153: '1116',
             154: '1119', 155: '1120', 156: '1128', 157: '1130', 158: '1134', 159: '1138', 160: '1139', 161: '1142',
             162: '1146', 163: '1161', 164: '1166', 165: '1216', 166: '1233', 167: '1235', 168: '1236', 169: '1441',
             170: '1448', 171: '1450', 172: '1457', 173: '1461', 174: '1462', 175: '1464', 176: '1465', 177: '1468',
             178: '1475', 179: '1477', 180: '1478', 181: '1479', 182: '1480', 183: '1483', 184: '1485', 185: '1486',
             186: '1487', 187: '1488', 188: '1489', 189: '1494', 190: '1497', 191: '1499', 192: '1501', 193: '1503',
             194: '1509', 195: '1510', 196: '1515', 197: '1566', 198: '1567', 199: '1575', 200: '1590', 201: '1592',
             202: '1597', 203: '4134', 204: '4135', 205: '4137', 206: '4534', 207: '4538', 208: '4541', 209: '6332',
             210: '23381', 211: '23512', 212: '23517', 213: '40498', 214: '40499', 215: '40664', 216: '40668',
             217: '40670', 218: '40672', 219: '40677', 220: '40685', 221: '40687', 222: '40701', 223: '40713',
             224: '40900', 225: '40910', 226: '40923', 227: '40927', 228: '40966', 229: '40971', 230: '40975',
             231: '40978', 232: '40979', 233: '40981', 234: '40982', 235: '40983', 236: '40984', 237: '40994',
             238: '40996', 239: '41027', 240: '41142', 241: '41143', 242: '41144', 243: '41145', 244: '41146',
             245: '41150', 246: '41156', 247: '41157', 248: '41158', 249: '41159', 250: '41161', 251: '41163',
             252: '41164', 253: '41165', 254: '41166', 255: '41167', 256: '41168', 257: '41169', 258: '41228',
             259: '41972', 260: '42734', 261: '42742', 262: '42769', 263: '42809', 264: '42810'}
LEARNER_ZOO = {0: 'SVC_linear', 1: 'SVC_poly', 2: 'SVC_rbf', 3: 'SVC_sigmoid', 4: 'Decision Tree', 5: 'ExtraTree',
               6: 'LogisticRegression', 7: 'PassiveAggressive', 8: 'Perceptron', 9: 'RidgeClassifier',
               10: 'SGDClassifier', 11: 'MLP', 12: 'LDA', 13: 'QDA', 14: 'BernoulliNB', 15: 'MultinomialNB',
               16: 'ComplementNB', 17: 'GaussianNB', 18: 'KNN', 19: 'NearestCentroid', 20: 'ens.ExtraTrees',
               21: 'ens.RandomForest', 22: 'ens.GradientBoosting', 23: 'DummyClassifier'}
ANCHOR_SIZE = np.ceil(16 * 2 ** ((np.arange(137)) / 8)).astype(int)

### load data: validation accuracy
lc_data = h5py.File(Path.cwd() / 'dataset/LCDB11_ACC_265_noFS_raw.hdf5', 'r')['accuracy'][...][:, :, :, :, :, 1]

mean_valid_lc_nofs = np.nanmean(lc_data, axis=(2, 3))


def splitData(Learner_A, Learner_B, train_split=0.8, test_split=0.5, SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.manual_seed(SEED)

    ### dataset split
    train_data_indices, test_data_indices = train_test_split(np.arange(len(OPENML_ID)), test_size=0.2, random_state=42)
    ### learner split
    train_learner_indices_A = np.array([Learner_A])
    train_learner_indices_B = np.array([Learner_B])

    ### UD, UL, UDUL
    train_data_A = lc_data[train_data_indices][:, train_learner_indices_A, :].transpose(0, 2, 3, 1, 4).reshape(-1, 1,
                                                                                                               137)
    train_data_B = lc_data[train_data_indices][:, train_learner_indices_B, :].transpose(0, 2, 3, 1, 4).reshape(-1, 1,
                                                                                                               137)

    test_data_UD_A = lc_data[test_data_indices][:, train_learner_indices_A, :].transpose(0, 2, 3, 1, 4).reshape(-1, 1,
                                                                                                                137)
    test_data_UD_B = lc_data[test_data_indices][:, train_learner_indices_B, :].transpose(0, 2, 3, 1, 4).reshape(-1, 1,
                                                                                                                137)

    size_to_sample = int(np.ceil(len(train_data_A) * train_split))
    sample1 = train_data_A[np.random.choice(train_data_A.shape[0], size_to_sample, replace=False), :]
    sample2 = train_data_B[np.random.choice(train_data_B.shape[0], len(train_data_A) - size_to_sample, replace=False),
              :]

    train_data = np.concatenate((sample1, sample2), axis=0)

    # size_to_sample_test = int(np.ceil(len(test_data_UD_A) * test_split))
    # test_sample1 = test_data_UD_A[np.random.choice(test_data_UD_A.shape[0], size_to_sample_test, replace=False), :]
    # test_sample2 = test_data_UD_B[
    #                np.random.choice(test_data_UD_B.shape[0], len(test_data_UD_A) - size_to_sample_test, replace=False),
    #                :]
    #
    # test_data_UD = np.concatenate((test_sample1, test_sample2), axis=0)
    tt = np.array([Learner_A, Learner_B])
    test_data_UD = lc_data[test_data_indices][:, tt, :].transpose(0, 2, 3, 1, 4).reshape(-1, 1, 137)

    print(f"Train data shape: {train_data.shape}")
    print(f"Test data UD shape: {test_data_UD.shape}")

    return train_data, test_data_UD


def evaluate(curve, model):
    y = torch.from_numpy(curve).float().unsqueeze(-1)
    x = torch.arange(1, y.shape[0] + 1).unsqueeze(-1).float()

    # construct
    num_last_anchor = 20
    cutoff = len(curve) - num_last_anchor

    x_train = x[:cutoff]
    y_train = y[:cutoff]
    x_test = x[cutoff:]
    qs = [0.05, 0.5, 0.95]

    normalizer = lcpfn.utils.identity_normalizer()

    y_train_norm = normalizer[0](y_train)

    # forward
    single_eval_pos = x_train.shape[0]
    x = torch.cat([x_train, x_test], dim=0).unsqueeze(1)
    y = y_train.unsqueeze(1)

    logits = model((x, y), single_eval_pos=single_eval_pos)

    predictions = normalizer[1](
        torch.cat([model.criterion.icdf(logits, q) for q in qs], dim=1)
    )

    x_test_np = x[cutoff:].detach().cpu().numpy().flatten()
    pred_mean = predictions[:, 1].detach().cpu().numpy()
    pred_lower = predictions[:, 0].detach().cpu().numpy()
    pred_upper = predictions[:, 2].detach().cpu().numpy()

    grountruth = curve[cutoff:]
    # mse = np.sqrt(mean_squared_error(grountruth, pred_mean))
    # mse = np.mean(np.abs(grountruth - pred_mean))
    mse = root_mean_squared_log_error(grountruth, pred_mean)
    within_ci = (grountruth >= pred_lower) & (grountruth <= pred_upper)
    coverage = within_ci.mean() * 100

    return mse, coverage


def plot_metric_with_error(name, values, color='skyblue', ylabel='Value', units=''):
    mean = np.mean(values)
    std = np.std(values)

    plt.figure()
    plt.errorbar(
        x=[0], y=[mean], yerr=[std],
        fmt='o', capsize=5, color=color, ecolor='black',
        elinewidth=2, markersize=8
    )
    plt.xticks([0], [name])
    plt.ylabel(f'{ylabel} {units}')
    plt.title(f'{name} with Standard Deviation')
    plt.ylim(bottom=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def evaluate_all(curves, model):
    mseList = []
    coverageList = []
    for curve in curves:
        curve = curve[~np.isnan(curve)]
        if len(curve) == 0:
            continue
        mse, coverage = evaluate(curve, model)
        mseList.append(mse)
        coverageList.append(coverage)
    return mseList, coverageList


Seeds = [5, 10, 23]
Learner_A = 5
Learner_B = 8
train_split = 0.8
test_splits = np.linspace(0.0, 1.0, num=11)
print(test_splits)


# for seed in Seeds:
#
#     model = torch.load(f'Mixed_Training_{Learner_A}_{Learner_B}_{train_split}_{seed}.pth', weights_only=False)
#     model.eval()
#     MAE_mean = []
#     Coverage_mean = []
#
#     MAE_std = []
#     Coverage_std = []
#
#     for test_split in test_splits:
#         train_data, test_data_UD = splitData(Learner_A, Learner_B, train_split=train_split, test_split=test_split,
#                                              SEED=seed)
#         mse_list, coverage_list = evaluate_all(test_data_UD)
#         MAE_mean.append(np.mean(mse_list))
#         Coverage_mean.append(np.mean(coverage_list))
#         MAE_std.append(np.std(mse_list))
#         Coverage_std.append(np.std(coverage_list))
#         plot_metric_with_error(f'RMSE for seed {seed}, {test_split}', mse_list, color='skyblue', ylabel='RMSE', units='(%)')
#         plot_metric_with_error(f'Coverage for seed {seed}, {test_split}', coverage_list, color='lightgreen', ylabel='Coverage', units='(%)')
#
def plot_mean():
    for seed in Seeds:
        model = torch.load(f'Mixed_Training_{Learner_A}_{Learner_B}_{train_split}_{seed}.pth', weights_only=False)
        model.eval()

        MAE_mean = []
        Coverage_mean = []
        MAE_std = []
        Coverage_std = []

        for test_split in test_splits:
            print(f"\n[Seed {seed}] Evaluating with test_split = {test_split}")
            train_data, test_data_UD = splitData(Learner_A, Learner_B, train_split=train_split, test_split=test_split,
                                                 SEED=seed)

            mse_list, coverage_list = evaluate_all(test_data_UD, model)

            def filter_iqr(data):
                data = np.array(data)
                Q1 = np.percentile(data, 25)
                Q3 = np.percentile(data, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                return data[(data >= lower_bound) & (data <= upper_bound)]

            filtered_mse = filter_iqr(mse_list)

            MAE_mean.append(np.mean(mse_list))
            MAE_std.append(np.std(mse_list))

            Coverage_mean.append(np.mean(coverage_list))
            Coverage_std.append(np.std(coverage_list))
            # print("Test split: ", test_split, "MAE Mean:", np.mean(mse_list), "Std Dev:", np.std(mse_list))
            # print("Test split: ", test_split, "Coverage Mean:", np.mean(coverage_list), "Std Dev:",
            #       np.std(coverage_list))
        print("MAE Means:", MAE_mean, "Std Devs:", MAE_std)
        print("Coverage Means:", Coverage_mean, "Std Devs:", Coverage_std)
        # Summary Plot across test splits
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.errorbar(test_splits, MAE_mean, yerr=MAE_std, fmt='o-', capsize=5, label=f'Seed {seed}', color='blue')
        plt.xlabel('Test Split')
        plt.ylabel('Mean Absolute Error')
        plt.title(f'MAE vs Test Split (Seed {seed}) (Learner {Learner_A} vs {Learner_B})')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.errorbar(test_splits, Coverage_mean, yerr=Coverage_std, fmt='o-', capsize=5, label=f'Seed {seed}',
                     color='green')
        plt.xlabel('Test Split')
        plt.ylabel('Coverage (%)')
        plt.title(f'Coverage vs Test Split (Seed {seed}) (Learner {Learner_A} vs {Learner_B})')
        plt.grid(True)

        plt.tight_layout()
        plt.show()


def plot_box():

    for seed in Seeds:
        model = torch.load(f'Mixed_Training_{Learner_A}_{Learner_B}_{train_split}_{seed}.pth', weights_only=False)
        model.eval()

        all_mse = []
        all_coverage = []

        for test_split in test_splits:
            print(f"\n[Seed {seed}] Evaluating with test_split = {test_split}")
            train_data, test_data_UD = splitData(Learner_A, Learner_B, train_split=train_split, test_split=test_split,
                                                 SEED=seed)

            mse_list, coverage_list = evaluate_all(test_data_UD, model)

            def filter_iqr(data):
                data = np.array(data)
                Q1 = np.percentile(data, 25)
                Q3 = np.percentile(data, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                return data[(data >= lower_bound) & (data <= upper_bound)]

            filtered_mse = filter_iqr(mse_list)

            all_mse.append(filtered_mse)
            all_coverage.append(coverage_list)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.boxplot(all_mse, positions=range(len(test_splits)), widths=0.6)
        plt.xticks(range(len(test_splits)), [f"{ts:.2f}" for ts in test_splits])
        plt.xlabel('Test Split')
        plt.ylabel('RMSE')
        plt.title(f'RMSE Distribution vs Test Split (Seed {seed}) (Learner {Learner_A} vs {Learner_B})')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        plt.subplot(1, 2, 2)
        plt.boxplot(all_coverage, positions=range(len(test_splits)), widths=0.6)
        plt.xticks(range(len(test_splits)), [f"{ts:.2f}" for ts in test_splits])
        plt.xlabel('Test Split')
        plt.ylabel('Coverage (%)')
        plt.title(f'Coverage Distribution vs Test Split (Seed {seed}) (Learner {Learner_A} vs {Learner_B})')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()


def plot_hist():
    for seed in Seeds:
        model = torch.load(f'Mixed_Training_{Learner_A}_{Learner_B}_{train_split}_{seed}.pth', weights_only=False)
        model.eval()

        all_mse = []
        all_coverage = []

        for test_split in test_splits:
            print(f"\n[Seed {seed}] Evaluating with test_split = {test_split}")
            train_data, test_data_UD = splitData(Learner_A, Learner_B, train_split=train_split, test_split=test_split,
                                                 SEED=seed)

            mse_list, coverage_list = evaluate_all(test_data_UD, model)

            def filter_iqr(data):
                data = np.array(data)
                Q1 = np.percentile(data, 25)
                Q3 = np.percentile(data, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                return data[(data >= lower_bound) & (data <= upper_bound)]

            filtered_mse = filter_iqr(mse_list)

            all_mse.append(filtered_mse)
            all_coverage.append(coverage_list)

        # Histogram of RMSE and Coverage
        num_splits = len(test_splits)
        plt.figure(figsize=(16, 4 * num_splits))

        for i, (mse_vals, cov_vals) in enumerate(zip(all_mse, all_coverage)):
            plt.subplot(num_splits, 2, 2 * i + 1)
            plt.hist(mse_vals, bins=20, color='skyblue', edgecolor='black')
            plt.xlabel('RMSE')
            plt.ylabel('Frequency')
            plt.title(f'RMSE Histogram (Test Split {test_splits[i]:.2f}, Seed {seed})')

            plt.subplot(num_splits, 2, 2 * i + 2)
            plt.hist(cov_vals, bins=20, color='lightgreen', edgecolor='black')
            plt.xlabel('Coverage (%)')
            plt.ylabel('Frequency')
            plt.title(f'Coverage Histogram (Test Split {test_splits[i]:.2f}, Seed {seed})')

        plt.tight_layout()
        plt.show()


def plot_violin():
    for seed in Seeds:
        model = torch.load(f'Mixed_Training_{Learner_A}_{Learner_B}_{train_split}_{seed}.pth', weights_only=False)
        model.eval()

        mse_data = []
        coverage_data = []

        for test_split in test_splits:
            print(f"\n[Seed {seed}] Evaluating with test_split = {test_split}")
            train_data, test_data_UD = splitData(Learner_A, Learner_B, train_split=train_split, test_split=test_split,
                                                 SEED=seed)

            mse_list, coverage_list = evaluate_all(test_data_UD, model)
            coverage_list = np.array(coverage_list)
            print(np.where(coverage_list > 100))

            def filter_iqr(data):
                data = np.array(data)
                Q1 = np.percentile(data, 25)
                Q3 = np.percentile(data, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                return data[(data >= lower_bound) & (data <= upper_bound)]

            filtered_mse = filter_iqr(mse_list)

            for val in filtered_mse:
                mse_data.append({'MAE': val, 'Test Split': f"{test_split:.2f}"})
            for val in coverage_list:
                coverage_data.append({'Coverage': val, 'Test Split': f"{test_split:.2f}"})

        # Convert to DataFrames for seaborn
        df_mse = pd.DataFrame(mse_data)
        df_coverage = pd.DataFrame(coverage_data)
        # Create violin plots
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.violinplot(x='Test Split', y='MAE', data=df_mse, inner='box', palette='Pastel1')
        plt.title(f'Violin Plot of MAE (Seed {seed}) (Learner {Learner_A} vs {Learner_B})')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        plt.subplot(1, 2, 2)
        sns.violinplot(x='Test Split', y='Coverage', data=df_coverage, inner='box', palette='Pastel2')
        plt.title(f'Violin Plot of Coverage (Seed {seed}) (Learner {Learner_A} vs {Learner_B})')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()


def plot_mean_scatter():
    all_mae = []
    all_cov = []
    all_test_splits = []
    all_seeds = []
    mae_stds = []
    cov_stds = []

    for seed in Seeds:
        model = torch.load(f'Mixed_Training_{Learner_A}_{Learner_B}_{train_split}_{seed}.pth', weights_only=False)
        model.eval()

        for test_split in test_splits:
            print(f"\n[Seed {seed}] Evaluating with test_split = {test_split}")
            train_data, test_data_UD = splitData(Learner_A, Learner_B, train_split=train_split, test_split=test_split,
                                                 SEED=seed)

            mse_list, coverage_list = evaluate_all(test_data_UD, model)

            def filter_iqr(data):
                data = np.array(data)
                Q1 = np.percentile(data, 25)
                Q3 = np.percentile(data, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                return data[(data >= lower_bound) & (data <= upper_bound)]

            filtered_mse = filter_iqr(mse_list)
            filtered_coverage = filter_iqr(coverage_list)

            mean_mae = np.mean(mse_list)
            std_mae = np.std(mse_list)
            mean_cov = np.mean(coverage_list)
            std_cov = np.std(coverage_list)

            all_mae.append(mean_mae)
            all_cov.append(mean_cov)
            all_test_splits.append(test_split)
            all_seeds.append(seed)
            mae_stds.append(std_mae)
            cov_stds.append(std_cov)

    rng = np.random.default_rng(seed=42)
    jitter = rng.normal(0, 0.01, size=len(all_test_splits))
    test_splits_jittered = np.array(all_test_splits) + jitter

    cmap = plt.get_cmap("viridis")

    # Plot MAE
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sc1 = plt.scatter(
        test_splits_jittered, all_mae,
        s=np.array(mae_stds) * 100,  # scale marker size
        c=mae_stds, cmap=cmap, alpha=0.8, edgecolor='k'
    )
    plt.colorbar(sc1, label='MAE Std Dev')
    plt.xlabel('Test Split')
    plt.ylabel('Mean Absolute Error')
    plt.title(f'MAE Scatter (Learner {Learner_A} vs {Learner_B})')
    plt.grid(True)

    z_mae = np.polyfit(all_test_splits, all_mae, 1)
    p_mae = np.poly1d(z_mae)
    xs = np.linspace(min(test_splits), max(test_splits), 100)
    plt.plot(xs, p_mae(xs), 'r--', label='Trend Line')
    plt.legend()

    # Plot Coverage
    plt.subplot(1, 2, 2)
    sc2 = plt.scatter(
        test_splits_jittered, all_cov,
        s=np.array(cov_stds) * 100,  # scale marker size
        c=cov_stds, cmap=cmap, alpha=0.8, edgecolor='k'
    )
    plt.colorbar(sc2, label='Coverage Std Dev')
    plt.xlabel('Test Split')
    plt.ylabel('Coverage (%)')
    plt.title(f'Coverage Scatter (Learner {Learner_A} vs {Learner_B})')
    plt.grid(True)

    # Add trend line (linear fit)
    z_cov = np.polyfit(all_test_splits, all_cov, 1)
    p_cov = np.poly1d(z_cov)
    plt.plot(xs, p_cov(xs), 'r--', label='Trend Line')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("MAE Mean:", np.mean(all_mae), "Std Dev:", np.std(all_mae))
    print("Coverage Mean:", np.mean(all_cov), "Std Dev:", np.std(all_cov))

def plot_mean_across_seeds():
    all_MAE_means = []
    all_Coverage_means = []

    for seed in Seeds:
        model = torch.load(f'Mixed_Training_{Learner_A}_{Learner_B}_{train_split}_{seed}.pth', weights_only=False)
        model.eval()

        MAE_mean = []
        Coverage_mean = []

        for test_split in test_splits:
            print(f"\n[Seed {seed}] Evaluating with test_split = {test_split}")
            train_data, test_data_UD = splitData(Learner_A, Learner_B, train_split=train_split, test_split=test_split,
                                                 SEED=seed)

            mse_list, coverage_list = evaluate_all(test_data_UD, model)

            MAE_mean.append(np.mean(mse_list))
            Coverage_mean.append(np.mean(coverage_list))

        all_MAE_means.append(MAE_mean)
        all_Coverage_means.append(Coverage_mean)

    # Convert to NumPy arrays for easier manipulation
    all_MAE_means = np.array(all_MAE_means)           # shape: (num_seeds, num_test_splits)
    all_Coverage_means = np.array(all_Coverage_means)

    # Compute averages and std deviations across seeds
    avg_MAE = np.mean(all_MAE_means, axis=0)
    std_MAE = np.std(all_MAE_means, axis=0)

    avg_Coverage = np.mean(all_Coverage_means, axis=0)
    std_Coverage = np.std(all_Coverage_means, axis=0)

    # Plotting
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.errorbar(test_splits, avg_MAE, yerr=std_MAE, fmt='o-', capsize=5, color='blue')
    plt.xlabel('Test Split')
    plt.ylabel('Mean Absolute Error')
    plt.title(f'Average MAE across Seeds (Learner {Learner_A} vs {Learner_B})')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.errorbar(test_splits, avg_Coverage, yerr=std_Coverage, fmt='o-', capsize=5, color='green')
    plt.xlabel('Test Split')
    plt.ylabel('Coverage (%)')
    plt.title(f'Average Coverage across Seeds (Learner {Learner_A} vs {Learner_B})')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# plot_violin()
#plot_mean()
plot_box()
#plot_mean_across_seeds()