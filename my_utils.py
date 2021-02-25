import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold


def get_cross_validated_subjects(x, y, names, n_folds = 5):
    ''' A function to implement the cross-validation manually '''

    new_person = "ss_ss_ss_ss"

    subjects_pt = np.array([], dtype=str)
    subjects_co = np.array([], dtype=str)
    subjects_pt_idx = np.array([], dtype=int)
    subjects_co_idx = np.array([], dtype=int)

    for i in range(len(names)):
        if new_person.split("_")[0:2] != names[i].split("_")[0:2]:
            new_person = names[i]
            if new_person.__contains__("Pt"):
                subjects_pt = np.append(subjects_pt, [new_person], axis=0)
                subjects_pt_idx = np.append(subjects_pt_idx, [i], axis=0)
            else:
                subjects_co = np.append(subjects_co, [new_person], axis=0)
                subjects_co_idx = np.append(subjects_co_idx, [i], axis=0)

    print("------", subjects_pt.shape, subjects_pt_idx.shape)
    kf = KFold(n_splits=n_folds)
    KFold(n_splits=n_folds, random_state=42, shuffle=True)
    splits = kf.split(subjects_pt, subjects_pt_idx)

    subjects_pt_train_all = np.asarray([])
    subjects_pt_train_idx_all = np.asarray([])
    subjects_pt_test_all = np.asarray([])
    subjects_pt_test_idx_all = np.asarray([])

    n_fold_train_samples = []
    n_fold_test_samples = []

    train_starting = 0
    test_starting = 0
    for pt_name_train_index, pt_name_test_index in kf.split(subjects_pt, subjects_pt_idx):
        subjects_pt_train_all = np.append(subjects_pt_train_all, subjects_pt[pt_name_train_index])
        subjects_pt_train_idx_all = np.append(subjects_pt_train_idx_all, subjects_pt_idx[pt_name_train_index])
        n_fold_train_samples.append(train_starting)
        train_starting += subjects_pt[pt_name_train_index].shape[0]

        subjects_pt_test_all = np.append(subjects_pt_test_all, subjects_pt[pt_name_test_index])
        subjects_pt_test_idx_all = np.append(subjects_pt_test_idx_all, subjects_pt_idx[pt_name_test_index])
        n_fold_test_samples.append(test_starting)
        test_starting += subjects_pt[pt_name_test_index].shape[0] - 1

    patients = [subjects_pt_train_all, subjects_pt_train_idx_all, subjects_pt_test_all, subjects_pt_test_idx_all, n_fold_train_samples, n_fold_test_samples]

    subjects_co_train_all = np.asarray([])
    subjects_co_train_idx_all = np.asarray([])
    subjects_co_test_all = np.asarray([])
    subjects_co_test_idx_all = np.asarray([])

    n_fold_train_samples = []
    n_fold_test_samples = []

    train_starting = 0
    test_starting = 0
    for co_name_train_index, co_name_test_index in kf.split(subjects_co, subjects_co_idx):
        subjects_co_train_all = np.append(subjects_co_train_all, subjects_co[co_name_train_index])
        subjects_co_train_idx_all = np.append(subjects_co_train_idx_all, subjects_co_idx[co_name_train_index])
        n_fold_train_samples.append(train_starting)
        train_starting += subjects_co[co_name_train_index].shape[0] - 1

        subjects_co_test_all = np.append(subjects_co_test_all, subjects_co[co_name_test_index])
        subjects_co_test_idx_all = np.append(subjects_co_test_idx_all, subjects_co_idx[co_name_test_index])
        n_fold_test_samples.append(test_starting)
        test_starting += subjects_co[co_name_test_index].shape[0]

    controls = [subjects_co_train_all, subjects_co_train_idx_all, subjects_co_test_all, subjects_co_test_idx_all,
                n_fold_train_samples, n_fold_test_samples]

    return patients, controls

def split_humanly_v2(x, y, names, pt_name_train_x, pt_name_test_x, pt_name_train_y, pt_name_test_y, co_name_train_x, co_name_test_x, co_name_train_y, co_name_test_y):

    name_train_x = np.append(pt_name_train_x[:co_name_train_x.shape[0]], co_name_train_x, axis=0)
    name_train_x = name_train_x.reshape(name_train_x.shape[0], 1)
    name_train_y = np.append(pt_name_train_y[:co_name_train_y.shape[0]], co_name_train_y, axis=0)
    name_train_y = name_train_y.reshape(name_train_y.shape[0], 1)

    train = np.concatenate((name_train_x, name_train_y), axis=1)
    np.random.shuffle(train)
    name_train_x = train[:, 0]
    name_train_y = np.asarray(train[:, 1], dtype=int)

    name_test_x = np.append(pt_name_test_x[:co_name_test_x.shape[0]], co_name_test_x, axis=0)
    name_test_x = name_test_x.reshape(name_test_x.shape[0], 1)
    name_test_y = np.append(pt_name_test_y[:co_name_test_y.shape[0]], co_name_test_y, axis=0)
    name_test_y = name_test_y.reshape(name_test_y.shape[0], 1)

    test = np.concatenate((name_test_x, name_test_y), axis=1)
    np.random.shuffle(test)
    name_test_x = test[:, 0]
    name_test_y = np.asarray(test[:, 1], dtype=int)

    name_train_x = name_train_x.reshape(name_train_x.shape[0], )
    name_train_y = name_train_y.reshape(name_train_y.shape[0], )
    name_test_x = name_test_x.reshape(name_test_x.shape[0], )
    name_test_y = name_test_y.reshape(name_test_y.shape[0], )

    print("-total train:", len(name_train_x), "-Pt in train:", len([i for i in name_train_x if i.__contains__("Pt")]),
          "-Co in train:", len([i for i in name_train_x if i.__contains__("Co")]))

    print("-total test:", len(name_test_x), "-Pt in test:", len([i for i in name_test_x if i.__contains__("Pt")]),
          "-Co in test:", len([i for i in name_test_x if i.__contains__("Co")]))

    train_x = np.array([])
    train_y = np.array([])
    for i in range(len(name_train_y)):
        unique_name = name_train_x[i]

        if np.size(train_x) == 0:
            train_x = [x[name_train_y[i], :, :]]
            train_y = [y[name_train_y[i], :]]
        else:
            train_x = np.append(train_x, [x[name_train_y[i], :, :]], axis=0)
            train_y = np.append(train_y, [y[name_train_y[i], :]], axis=0)
        next = name_train_y[i] + 1

        while next < len(names) and unique_name.split("_")[0:2] == names[next].split("_")[0:2]:
            train_x = np.append(train_x, [x[next, :, :]], axis=0)
            train_y = np.append(train_y, [y[next, :]], axis=0)
            next = next + 1

    test_x = np.array([])
    test_y = np.array([])
    for i in range(len(name_test_y)):
        unique_name = name_test_x[i]
        if np.size(test_x) == 0:
            test_x = [x[name_test_y[i], :, :]]
            test_y = [y[name_test_y[i], :]]
        else:
            test_x = np.append(test_x, [x[name_test_y[i], :, :]], axis=0)
            test_y = np.append(test_y, [y[name_test_y[i], :]], axis=0)
        next = name_test_y[i] + 1

        while next < len(names) and unique_name.split("_")[0:2] == names[next].split("_")[0:2]:
            test_x = np.append(test_x, [x[next, :, :]], axis=0)
            test_y = np.append(test_y, [y[next, :]], axis=0)
            next = next + 1

    return train_x, train_y, test_x, test_y


def split_humanly(x, y, names):
    new_person = "ss_ss_ss_ss"

    subjects_pt = np.array([], dtype=str)
    subjects_co = np.array([], dtype=str)
    subjects_pt_idx = np.array([], dtype=int)
    subjects_co_idx = np.array([], dtype=int)

    # subjects = np.array([new_person], dtype=str)
    # subjects_idx = np.array([0], dtype=int)
    for i in range(len(names)):
        if new_person.split("_")[0:2] != names[i].split("_")[0:2]:
            new_person = names[i]
            if new_person.__contains__("Pt"):
                subjects_pt = np.append(subjects_pt, [new_person], axis=0)
                subjects_pt_idx = np.append(subjects_pt_idx, [i], axis=0)
            else:
                subjects_co = np.append(subjects_co, [new_person], axis=0)
                subjects_co_idx = np.append(subjects_co_idx, [i], axis=0)

            # subjects = np.append(subjects, [new_person], axis=0)
            # subjects_idx = np.append(subjects_idx, [i], axis=0)

    pt_name_train_x, pt_name_test_x, pt_name_train_y, pt_name_test_y = train_test_split(subjects_pt, subjects_pt_idx,
                                                                                        test_size=0.53,
                                                                                        random_state=42, shuffle=True)
    pt_name_valid_x, pt_name_test_x, pt_name_valid_y, pt_name_test_y = train_test_split(pt_name_test_x, pt_name_test_y,
                                                                                        test_size=0.60,
                                                                                        random_state=42, shuffle=True)

    co_name_train_x, co_name_test_x, co_name_train_y, co_name_test_y = train_test_split(subjects_co, subjects_co_idx,
                                                                                        test_size=0.14,
                                                                                        random_state=42, shuffle=True)
    co_name_valid_x, co_name_test_x, co_name_valid_y, co_name_test_y = train_test_split(co_name_test_x, co_name_test_y,
                                                                                        test_size=0.60,
                                                                                        random_state=42, shuffle=True)
    name_train_x = np.append(pt_name_train_x[:co_name_train_x.shape[0]], co_name_train_x, axis=0)
    name_train_x = name_train_x.reshape(name_train_x.shape[0], 1)
    name_train_y = np.append(pt_name_train_y[:co_name_train_y.shape[0]], co_name_train_y, axis=0)
    name_train_y = name_train_y.reshape(name_train_y.shape[0], 1)

    train = np.concatenate((name_train_x, name_train_y), axis=1)
    np.random.shuffle(train)
    name_train_x = train[:, 0]
    name_train_y = np.asarray(train[:, 1], dtype=int)

    name_valid_x = np.append(pt_name_valid_x[:co_name_valid_x.shape[0]], co_name_valid_x, axis=0)
    name_valid_x = name_valid_x.reshape(name_valid_x.shape[0], 1)
    name_valid_y = np.append(pt_name_valid_y[:co_name_valid_y.shape[0]], co_name_valid_y, axis=0)
    name_valid_y = name_valid_y.reshape(name_valid_y.shape[0], 1)

    valid = np.concatenate((name_valid_x, name_valid_y), axis=1)
    np.random.shuffle(valid)
    name_valid_x = valid[:, 0]
    name_valid_y = np.asarray(valid[:, 1], dtype=int)

    name_test_x = np.append(pt_name_test_x[:co_name_test_x.shape[0]], co_name_test_x, axis=0)
    name_test_x = name_test_x.reshape(name_test_x.shape[0], 1)
    name_test_y = np.append(pt_name_test_y[:co_name_test_y.shape[0]], co_name_test_y, axis=0)
    name_test_y = name_test_y.reshape(name_test_y.shape[0], 1)

    test = np.concatenate((name_test_x, name_test_y), axis=1)
    np.random.shuffle(test)
    name_test_x = test[:, 0]
    name_test_y = np.asarray(test[:, 1], dtype=int)

    name_train_x = name_train_x.reshape(name_train_x.shape[0], )
    name_train_y = name_train_y.reshape(name_train_y.shape[0], )
    name_valid_x = name_valid_x.reshape(name_valid_x.shape[0], )
    name_valid_y = name_valid_y.reshape(name_valid_y.shape[0], )
    name_test_x = name_test_x.reshape(name_test_x.shape[0], )
    name_test_y = name_test_y.reshape(name_test_y.shape[0], )

    print("-total train:", len(name_train_x), "-Pt in train:", len([i for i in name_train_x if i.__contains__("Pt")]),
          "-Co in train:", len([i for i in name_train_x if i.__contains__("Co")]))
    print("-total valid:", len(name_valid_x), "-Pt in valid:", len([i for i in name_valid_x if i.__contains__("Pt")]),
          "-Co in valid:", len([i for i in name_valid_x if i.__contains__("Co")]))
    print("-total test:", len(name_test_x), "-Pt in test:", len([i for i in name_test_x if i.__contains__("Pt")]),
          "-Co in test:", len([i for i in name_test_x if i.__contains__("Co")]))
    train_x = np.array([])
    train_y = np.array([])
    for i in range(len(name_train_y)):
        unique_name = name_train_x[i]

        if np.size(train_x) == 0:
            train_x = [x[name_train_y[i], :, :]]
            train_y = [y[name_train_y[i], :]]
        else:
            train_x = np.append(train_x, [x[name_train_y[i], :, :]], axis=0)
            train_y = np.append(train_y, [y[name_train_y[i], :]], axis=0)
        next = name_train_y[i] + 1

        while next < len(names) and unique_name.split("_")[0:2] == names[next].split("_")[0:2]:
            train_x = np.append(train_x, [x[next, :, :]], axis=0)
            train_y = np.append(train_y, [y[next, :]], axis=0)
            next = next + 1

    valid_x = np.array([])
    valid_y = np.array([])
    for i in range(len(name_valid_y)):
        unique_name = name_valid_x[i]
        if np.size(valid_x) == 0:
            valid_x = [x[name_valid_y[i], :, :]]
            valid_y = [y[name_valid_y[i], :]]
        else:
            valid_x = np.append(valid_x, [x[name_valid_y[i], :, :]], axis=0)
            valid_y = np.append(valid_y, [y[name_valid_y[i], :]], axis=0)
        next = name_valid_y[i] + 1

        while next < len(names) and unique_name.split("_")[0:2] == names[next].split("_")[0:2]:
            valid_x = np.append(valid_x, [x[next, :, :]], axis=0)
            valid_y = np.append(valid_y, [y[next, :]], axis=0)
            next = next + 1

    test_x = np.array([])
    test_y = np.array([])
    for i in range(len(name_test_y)):
        unique_name = name_test_x[i]
        if np.size(test_x) == 0:
            test_x = [x[name_test_y[i], :, :]]
            test_y = [y[name_test_y[i], :]]
        else:
            test_x = np.append(test_x, [x[name_test_y[i], :, :]], axis=0)
            test_y = np.append(test_y, [y[name_test_y[i], :]], axis=0)
        next = name_test_y[i] + 1

        while next < len(names) and unique_name.split("_")[0:2] == names[next].split("_")[0:2]:
            test_x = np.append(test_x, [x[next, :, :]], axis=0)
            test_y = np.append(test_y, [y[next, :]], axis=0)
            next = next + 1

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def load_data(path="data/", function="None", normalization=False):
    x = np.array([])
    y = np.array([])
    names = np.genfromtxt(path + sorted(os.listdir(path))[1], dtype=str)[1:, -1]

    for filename in sorted(os.listdir(path)):
        if not filename.endswith(".csv"):  # if the file is not valid
            continue
        new_x = np.genfromtxt(path + filename, dtype=float)[1:, :-1]  # convert each csv file to numpy array
        new_name = np.genfromtxt(path + filename, dtype=str)[1:, -1]  # name of each sample ex : JuCo23_01_28_L1
        new_y = np.zeros((np.size(new_name, axis=0), 1))

        for i in range(np.size(new_y, axis=0)):
            if str(new_name[i,]).__contains__("Pt"):  # Patient as positive
                new_y[i,] = 1
            else:
                new_y[i,] = 0
        if np.size(x) == 0:
            x = new_x
            y = new_y
            names = new_name
        else:
            x = np.dstack((x, new_x))
            y = np.dstack((y, new_y))

    n_values = x.shape[2]  # number of total sensors

    if function == "average":
        n_values = 2
        y = y[:, :, n_values - 1]
        average_left = np.mean(x[:, :, :n_values], axis=2)
        average_right = np.mean(x[:, :, n_values:], axis=2)
        x = np.dstack((average_left, average_right))

    if function == "None":  # 16 sensors
        y = y[:, :, n_values - 1]

    if function == "difference":  # to reduce the total sensors to half
        n_values = int(n_values / 2)
        x = x[:, :, n_values:] - x[:, :, :n_values]
        y = y[:, :, n_values - 1]

    if normalization:
        for i in range(x.shape[0]):
            for j in range(x.shape[2]):
                m = np.mean(x[i, :, j])
                s = np.std(x[i, :, j]) + np.finfo(np.float32).eps
                x[i, :, j] = (x[i, :, j] - m) / s

    return x, y, names


def plot_data(x, y, type="normall"):
    plt.style.use("seaborn")
    my_dpi = 96
    plt.figure(figsize=(2000 / my_dpi, 1000 / my_dpi), dpi=my_dpi)

    T_x = x.shape[0]  # length of the sequence
    n_values = x.shape[1]  # number of the values(sensors)

    t = np.arange(0, T_x / 100, 0.01)

    if type == "average":
        y1 = np.ones(t.size)
        y2 = np.ones(t.size) * 2

        z1 = x[:, 0]
        z2 = x[:, 1]

        ax = plt.subplot(projection='3d')
        ax.plot(t, y1, z1, color='r')
        ax.plot(t, y2, z2, color='g')

        ax.add_collection3d(plt.fill_between(t, z1, z1, color='r', alpha=0.3), zs=1, zdir='y')
        ax.add_collection3d(plt.fill_between(t, z2, z2, color='g', alpha=0.3), zs=2, zdir='y')

        ax.set_xlabel('Time(s)', fontsize=20)
        ax.set_zlabel('Average vGRFs(N)', fontsize=20)
        if y == 1:
            subject_type = "Patient Subject"
        elif y == 0:
            subject_type = "Control Subject"
        ax.text2D(0.05, 0.95, subject_type, transform=ax.transAxes, fontsize=20)
        plt.show()
    else:
        y1 = np.ones(t.size)
        y2 = np.ones(t.size) * 2
        y3 = np.ones(t.size) * 3
        y4 = np.ones(t.size) * 4
        y5 = np.ones(t.size) * 5
        y6 = np.ones(t.size) * 6
        y7 = np.ones(t.size) * 7
        y8 = np.ones(t.size) * 8

        z1 = x[:, 0]
        z2 = x[:, 1]
        z3 = x[:, 2]
        z4 = x[:, 3]
        z5 = x[:, 4]
        z6 = x[:, 5]
        z7 = x[:, 6]
        z8 = x[:, 7]

        ax = plt.subplot(projection='3d')
        ax.plot(t, y1, z1, color='r')
        ax.plot(t, y2, z2, color='g')
        ax.plot(t, y3, z3, color='b')
        ax.plot(t, y4, z4, color='c')
        ax.plot(t, y5, z5, color='m')
        ax.plot(t, y6, z6, color='y')
        ax.plot(t, y7, z7, color='w')
        ax.plot(t, y8, z8, color='k')

        ax.add_collection3d(plt.fill_between(t, z1, z1, color='r', alpha=0.3), zs=1, zdir='y')
        ax.add_collection3d(plt.fill_between(t, z2, z2, color='g', alpha=0.3), zs=2, zdir='y')
        ax.add_collection3d(plt.fill_between(t, z3, z3, color='b', alpha=0.3), zs=3, zdir='y')
        ax.add_collection3d(plt.fill_between(t, z4, z4, color='c', alpha=0.3), zs=4, zdir='y')
        ax.add_collection3d(plt.fill_between(t, z5, z5, color='m', alpha=0.3), zs=5, zdir='y')
        ax.add_collection3d(plt.fill_between(t, z6, z6, color='y', alpha=0.3), zs=6, zdir='y')
        ax.add_collection3d(plt.fill_between(t, z7, z7, color='w', alpha=0.3), zs=7, zdir='y')
        ax.add_collection3d(plt.fill_between(t, z8, z8, color='k', alpha=0.3), zs=8, zdir='y')

        ax.set_xlabel('Time(s)', fontsize=20)
        ax.set_zlabel('vGRFs(N)', fontsize=20)
        if y == 1:
            subject_type = "Patient Subject"
        elif y == 0:
            subject_type = "Control Subject"
        ax.text2D(0.05, 0.95, subject_type, transform=ax.transAxes, fontsize=20)

        plt.show()
