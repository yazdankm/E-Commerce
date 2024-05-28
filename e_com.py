import pandas as pd
import numpy as np
from ipywidgets import interact, widgets, fixed
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter


def get_names(ids, data):
    names = []
    if ids.name == "Customer ID":
        for id in ids:
            names.append(data[data["Customer ID"] == id]["Customer Name"].unique()[0])
    elif ids.name == "Product ID":
        for id in ids:
            names.append(data[data["Product ID"] == id]["Product Name"].unique()[0])
    return names


def get_cost_price(data):
    """Get the Cost and Sell Price of the Product from Order's Sales, Profit and Quantity"""
    data.loc[:, "Cost"] = round((data["Sales"] - data["Profit"]) / data["Quantity"], 4)
    data.loc[:, "Sell Price"] = round(
        data["Sales"] / ((1 - data["Discount"]) * data["Quantity"]), 4
    )
    return data


def gen_product_id(data):
    """Generating unique Product IDs based on it's name and price"""
    for category in data["Category"].unique():
        for subcategory in data[data["Category"] == category]["Sub-Category"].unique():
            products_grouped = data[
                (data["Category"] == category) & (data["Sub-Category"] == subcategory)
            ].groupby(["Product Name", "Cost"])
            for i, (key, item) in enumerate(products_grouped):
                group = products_grouped.get_group(key)
                cat = group["Category"].unique()[0][:3].upper()
                sub = group["Sub-Category"].unique()[0][:3].upper()
                item["id"] = cat + "-" + sub + "-" + str(i).zfill(8)
                for index, row in item["id"].items():
                    data.loc[index, "id"] = row
    return data


def filter_outliers(data):
    """Filter ouliers for Series based on +- 1.5 Interquartile range"""
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = 1.5 * (q3 - q1)
    return data[(data > (q1 - iqr)) & (data < (q3 + iqr))]


def get_outliers(data):
    """Get the ouliers of the dataframe, based on all the columns"""
    filtered = pd.DataFrame()
    for col in data.columns[1:]:
        filtered[col] = filter_outliers(data[col])
    return data.iloc[data.iloc[:, 1].index.difference(filtered.iloc[:, 1].index)]


def k_cluster_analysis(data, show_scores):
    ssd = []
    range_n_clusters = np.arange(2, 10)
    max_score = 0
    optimal_n_clusters = 0
    for num_clusters in range_n_clusters:
        kmeans = KMeans(
            n_clusters=num_clusters, max_iter=50, n_init="auto", random_state=420
        )
        kmeans.fit(data)
        # Elbow-curve /SSD
        ssd.append(kmeans.inertia_)

        cluster_labels = kmeans.labels_
        # silhouette score
        silhouette_avg = silhouette_score(data, cluster_labels)
        if silhouette_avg > max_score:
            max_score = silhouette_avg
            optimal_n_clusters = num_clusters
        if show_scores:
            print(
                "For n_clusters={0}, the silhouette score is {1}".format(
                    num_clusters, silhouette_avg
                )
            )
    if show_scores:
        # plot the SSDs for each n_clusters
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.lineplot(ssd, ax=ax)
    return optimal_n_clusters


def k_clustering(dataframe, remove_outliers=True, show_plots=False):
    # Make a copy of a dataframe
    data = dataframe.copy()
    # Determine the number of columns in the dataframe
    features = data.columns[1:]
    b = len(features)
    color = ["#8000ff", "#82ffb6", "#ff9017", '#1502fb', '#ff0000']

    if show_plots:
        # Plot the features
        fig, ax = plt.subplots(1, b, figsize=(b * 4, 4))
        for i, col in enumerate(features):
            sns.boxplot(data=data[col], ax=ax[i], color=color[i])
            ax[i].set_xlabel(col)
        plt.tight_layout()

    if remove_outliers:
        for col in features:
            # Removing outliers
            data[col] = filter_outliers(data[col])
        data.dropna(inplace=True)
    if show_plots:
        # Plot the features with removed outliers
        fig, ax = plt.subplots(1, b, figsize=(b * 4, 4))
        for i, col in enumerate(features):
            sns.boxplot(data=data[col], ax=ax[i], color=color[i])
            ax[i].set_xlabel(col)
        ax[1].set_title("Outliers Removed")
        plt.tight_layout()

    # Scale the data with StandartScaler
    scaled_data = StandardScaler().fit_transform(data[data.columns[1:]])
    scaled_data = pd.DataFrame(scaled_data)
    scaled_data.columns = data.columns[1:]

    # Initiate k-means and find the k with the highest silhouette score
    kmeans = KMeans(
        n_clusters=k_cluster_analysis(scaled_data, show_scores=False),
        max_iter=50,
        n_init="auto",
        random_state=42
    )
    kmeans.fit(scaled_data)

    # Assign the labels to each data point, and execute the following script.
    kmeans.labels_
    label_list = kmeans.labels_
    print(sorted(Counter(label_list).items()))

    # Assign the label
    data["Cluster ID"] = kmeans.labels_

    fig, ax = plt.subplots(1, b, figsize=(b * 4, 4))
    for i, col in enumerate(features):
        sns.boxplot(x="Cluster ID", y=col, data=data, ax=ax[i])
        ax[i].set_xlabel("Cluster")
    plt.tight_layout()

    return data


def cluster_outliers(outliers, cluster_data):
    """Join the outliers back to the dataframe and assign a new cluster to them"""
    outliers = outliers.copy()
    new_cluster = cluster_data["Cluster ID"].max() + 1
    outliers.loc[:, "Cluster ID"] = new_cluster
    return pd.concat([cluster_data, outliers], axis=0)


def get_xyz(data):
    x = data.iloc[:, 3]
    y = data.iloc[:, 1]
    z = data.iloc[:, 2]
    c = data.iloc[:, -1]
    return (x, y, z, c)


def set_ax(ax, xyzc):
    color = "#303030"
    ax.set_facecolor("none")
    ax.set_xlabel(xyzc[0].name)
    ax.set_ylabel(xyzc[1].name)
    ax.set_zlabel(xyzc[2].name, labelpad=2)
    ax.xaxis.set_pane_color(color)
    ax.yaxis.set_pane_color(color)
    ax.zaxis.set_pane_color(color)

    return ax


def plot_3d(clustered_data, label: str, elev=30, azim=15):
    fig = plt.figure(figsize=(7, 7), facecolor="none")
    fig.set_facecolor("none")
    xyzc = get_xyz(clustered_data)
    # syntax for 3-D projection
    ax = plt.axes(projection="3d")
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "custom_colormap", ["#8000ff", "#82ffb6", "#ff9017", "#1500ff"]
    )
    scatter_plot = ax.scatter3D(
        xyzc[0], xyzc[1], xyzc[2], c=xyzc[3], cmap='rainbow', alpha=0.8,
    )
    set_ax(ax, xyzc)
    ax.set_title(f"{label}")

    legend_labels = sorted(xyzc[-1].unique())
    ax.legend(
        *[scatter_plot.legend_elements()[0], legend_labels],
        title="Clusters",
        loc="upper left",
    )
    ax.view_init(elev=elev, azim=azim)
    plt.show()


def interact_3d_plot(data, label):
    elevation_slider = widgets.IntSlider(
        value=30, min=0, max=89, description="Elevation"
    )
    azimuth_slider = widgets.IntSlider(value=30, min=0, max=89, description="Azimuth")
    interact(
        plot_3d,
        elev=elevation_slider,
        azim=azimuth_slider,
        clustered_data=fixed(data),
        label=fixed(label),
    )