import numpy as np

# Normalizing ranges
xRange = [-100, 100]
yRange = [-100, 100]
lRange = [0, 20]
wRange = [0, 5]
VRange = [-15, 15]
hRange = [-3, 3]


def normalizeData(dataArray):
    x = (dataArray[0] - xRange[0]) / (xRange[1] - xRange[0])
    y = (dataArray[1] - yRange[0]) / (yRange[1] - yRange[0])
    l = (dataArray[2] - lRange[0]) / (lRange[1] - lRange[0])
    w = (dataArray[3] - wRange[0]) / (wRange[1] - wRange[0])
    Vx = (dataArray[4] - VRange[0]) / (VRange[1] - VRange[0])
    Vy = (dataArray[5] - VRange[0]) / (VRange[1] - VRange[0])
    hd = (dataArray[6] - hRange[0]) / (hRange[1] - hRange[0])
    cl = dataArray[7]

    data = [x, y, l, w, Vx, Vy, hd, cl]
    return data


def normalize2dData(dataArray):

    normData = []

    for i in range(dataArray.shape[0]):

        x = (dataArray[i][0] - xRange[0]) / (xRange[1] - xRange[0])
        y = (dataArray[i][1] - yRange[0]) / (yRange[1] - yRange[0])
        l = (dataArray[i][2] - lRange[0]) / (lRange[1] - lRange[0])
        w = (dataArray[i][3] - wRange[0]) / (wRange[1] - wRange[0])
        Vx = (dataArray[i][4] - VRange[0]) / (VRange[1] - VRange[0])
        Vy = (dataArray[i][5] - VRange[0]) / (VRange[1] - VRange[0])
        hd = (dataArray[i][6] - hRange[0]) / (hRange[1] - hRange[0])

        cl = dataArray[i][7]

        p_exist = dataArray[i][8]

        if dataArray.shape[1] > 9:
            x_std_dev = (dataArray[i][9] - xRange[0]) / (xRange[1] - xRange[0])
            y_std_dev = (dataArray[i][10] - yRange[0]) / (yRange[1] - yRange[0])
            l_std_dev = (dataArray[i][11] - lRange[0]) / (lRange[1] - lRange[0])
            w_std_dev = (dataArray[i][12] - wRange[0]) / (wRange[1] - wRange[0])
            Vx_std_dev = (dataArray[i][13] - VRange[0]) / (VRange[1] - VRange[0])
            Vy_std_dev = (dataArray[i][14] - VRange[0]) / (VRange[1] - VRange[0])
            hd_std_dev = (dataArray[i][15] - hRange[0]) / (hRange[1] - hRange[0])

            data = [
                x,
                y,
                l,
                w,
                Vx,
                Vy,
                hd,
                cl,
                p_exist,
                x_std_dev,
                y_std_dev,
                l_std_dev,
                w_std_dev,
                Vx_std_dev,
                Vy_std_dev,
                hd_std_dev,
            ]
        else:
            data = [x, y, l, w, Vx, Vy, hd, cl, p_exist]

        normData.append(data)

    return np.asarray(normData)


def denormalize2dData(dataArray):

    normData = []

    for i in range(dataArray.shape[0]):

        x = dataArray[i][0] * (xRange[1] - xRange[0]) + xRange[0]
        y = dataArray[i][1] * (yRange[1] - yRange[0]) + yRange[0]
        l = dataArray[i][2] * (lRange[1] - lRange[0]) + lRange[0]
        w = dataArray[i][3] * (wRange[1] - wRange[0]) + wRange[0]
        Vx = dataArray[i][4] * (VRange[1] - VRange[0]) + VRange[0]
        Vy = dataArray[i][5] * (VRange[1] - VRange[0]) + VRange[0]
        hd = dataArray[i][6] * (hRange[1] - hRange[0]) + hRange[0]

        cl = dataArray[i][7]

        p_exist = dataArray[i][8]

        if dataArray.shape[1] > 9:
            x_std_dev = dataArray[i][9] * (xRange[1] - xRange[0]) + xRange[0]
            y_std_dev = dataArray[i][10] * (yRange[1] - yRange[0]) + yRange[0]
            l_std_dev = dataArray[i][11] * (lRange[1] - lRange[0]) + lRange[0]
            w_std_dev = dataArray[i][12] * (wRange[1] - wRange[0]) + wRange[0]
            Vx_std_dev = dataArray[i][13] * (VRange[1] - VRange[0]) + VRange[0]
            Vy_std_dev = dataArray[i][14] * (VRange[1] - VRange[0]) + VRange[0]
            hd_std_dev = dataArray[i][15] * (hRange[1] - hRange[0]) + hRange[0]

            data = [
                x,
                y,
                l,
                w,
                Vx,
                Vy,
                hd,
                cl,
                p_exist,
                x_std_dev,
                y_std_dev,
                l_std_dev,
                w_std_dev,
                Vx_std_dev,
                Vy_std_dev,
                hd_std_dev,
            ]
        else:
            data = [x, y, l, w, Vx, Vy, hd, cl, p_exist]

        normData.append(data)

    return np.asarray(normData)


def denormalizeBBox(normalized_data):
    # Assuming your tensor is named 'normalized_tensor'
    denormalized_data = np.copy(
        normalized_data
    )  # Create a copy of the tensor to store denormalized values

    # Iterate through the tensor and denormalize each element
    for i in range(normalized_data.shape[0]):
        denormalized_data[i, -4] = (
            normalized_data[i, -4] * (xRange[1] - xRange[0])
        ) + xRange[
            0
        ]  # Denormalize 'x'
        denormalized_data[i, -3] = (
            normalized_data[i, -3] * (yRange[1] - yRange[0])
        ) + yRange[
            0
        ]  # Denormalize 'y'
        denormalized_data[i, -2] = (
            normalized_data[i, -2] * (lRange[1] - lRange[0])
        ) + lRange[
            0
        ]  # Denormalize 'l'
        denormalized_data[i, -1] = (
            normalized_data[i, -1] * (wRange[1] - wRange[0])
        ) + wRange[
            0
        ]  # Denormalize 'w'

    return denormalized_data


def denormalizeMot(normalized_data):
    # Assuming your tensor is named 'normalized_tensor'
    denormalized_data = np.copy(
        normalized_data
    )  # Create a copy of the tensor to store denormalized values

    # Iterate through the tensor and denormalize each element
    for i in range(normalized_data.shape[0]):
        denormalized_data[i, -3] = (
            normalized_data[i, -3] * (VRange[1] - VRange[0])
        ) + VRange[
            0
        ]  # Denormalize 'Vx'
        denormalized_data[i, -2] = (
            normalized_data[i, -2] * (VRange[1] - VRange[0])
        ) + VRange[
            0
        ]  # Denormalize 'Vy'
        denormalized_data[i, -1] = (
            normalized_data[i, -1] * (hRange[1] - hRange[0])
        ) + hRange[
            0
        ]  # Denormalize 'h'

    return denormalized_data
