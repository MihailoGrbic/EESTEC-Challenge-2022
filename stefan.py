N_WINDOW = 50
dataWindows = []

for i in reversed(range(1, N_WINDOW+1)):
    X_down = np.pad(X ,((i,0),(0,0)), mode='edge')[:-i, :]
    dataWindows.append(X_down)
dataWindows.append(X[:, :])
for i in range(1, N_WINDOW+1):
    X_up = np.pad(X ,((0,i),(0,0)), mode='edge')[i:, :]
    dataWindows.append(X_up)

# stackedWindows = np.stack(dataWindows)
# meanWindows = np.nanmean(stackedWindows, axis=0)
# X = np.concatenate(dataWindows, axis=1)

num_rows, num_cols = X.shape
additional = []
for row in range(num_rows):
    depth = X[row, 0]
    vals = []
    for adj in range(-N_WINDOW, +N_WINDOW):
        if (row+adj) < num_rows and abs(X[row+adj, 0] - depth) < 0.005:
            vals.append(X[row+adj, 0:-1])
    stacked_vals = np.stack(vals)
    mean_vals = np.mean(stacked_vals, axis=0)
    additional.append(mean_vals)
meanWindows = np.stack(additional)

X = np.concatenate([X, meanWindows], axis=1)