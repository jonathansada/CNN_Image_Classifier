import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import math

def getFrequencyTable(df, column):
    return pd.concat([df[column].value_counts(), df[column].value_counts(normalize=True).round(2)], axis=1)

def describe_complete(df, skipna=False):
    stats = {}
    stats["count"] = df.count()
    stats["unique"] = 0
    stats["mean"] = df.mean(skipna=skipna)
    stats["min"] = df.min(skipna=skipna)
    stats["quant_25"] = df.quantile(0.25)
    stats["quant_50"] = df.quantile(0.5)
    stats["quant_75"] = df.quantile(0.75)
    stats["max"] = df.max(skipna=skipna)
    stats["mode"] = ""
    stats["var"] = df.var(numeric_only=True, skipna= skipna)
    stats["std"] = df.std(skipna= skipna)
    stats["skew"] = df.skew(numeric_only=True, skipna= skipna)
    stats["kurt"] = df.kurt(numeric_only=True, skipna= skipna)

    stats_df = pd.DataFrame(stats)
    for column in df.columns:
        stats_df.loc[stats_df.index == column, "mode"] = "|".join([str(x) for x in df[column].mode()])
        stats_df.loc[stats_df.index == column, "unique"] = df.value_counts(column).count()

    return stats_df

def get_img_xy(df, path_column, label_column, img_width=28, img_height=28, color_mode="rgb", seed=None, 
               keep_aspect_ratio = True, random_flip=False, random_brightness=False, random_contrast=False, random_hue=False, random_saturation=False):
    set_seed(seed)
    # Prepare X
    images = []
    for index, row in df.iterrows():
        image_array = tf.keras.utils.img_to_array(tf.keras.utils.load_img(row[path_column], target_size=(img_height, img_width), keep_aspect_ratio=keep_aspect_ratio, color_mode=color_mode))
        if random_flip:
            image_array = tf.image.random_flip_left_right(image_array, seed=seed)
        if random_brightness:
            image_array = tf.image.random_brightness(image_array, 0.25, seed=seed)
        if random_contrast and seed == None:
            image_array = tf.image.random_contrast(image_array, 0.25, 0.75)
        if random_hue:
            image_array = tf.image.random_hue (image_array, 0.25, seed=seed)
        if random_saturation and image_array.shape[2] == 3: 
            image_array = tf.image.random_saturation(image_array, 0.25, 0.75, seed=seed)
        images.append(image_array)
    X = np.array(images)
    
    # Prepare y
    cat = pd.Categorical(df[label_column])
    class_names = cat.categories # class_names is needed to decode the y
    y = np.array(cat.codes)

    return X, y, class_names

def preview_img_dataset_xy(x, y, class_names):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        
        if x.dtype != "uint8":
            plt.imshow(x[i].astype("uint8"))
        else:
            plt.imshow(x[i])
            
        plt.title(class_names[y[i]])
        
        plt.axis("off")

def show_labels_distribution(labels, class_names):
    # Count the number of samples per class
    class_counts = Counter(labels)

    # Sort by class index
    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, counts)
    plt.xlabel('Class')
    plt.ylabel('Number of samples')
    plt.title('Number of samples per class')
    plt.xticks(rotation=45)
    plt.show()

def plot_confusion_matrix(y_test, y_pred, class_names):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,  yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

def cat_analisis(df, column, log=False, top=False, bottom=False, chart_size=False):
    frequency_table = getFrequencyTable(df, column)
    if len(df[column].unique()) <= 2: # If it is binary draw a pie plot
        chart_size = chart_size if chart_size else (6,6)
        plt.figure(figsize=chart_size)
        plt.pie(frequency_table["proportion"], autopct='%1.1f%%')
    else: # else draw a barchart
        chart_size = chart_size if chart_size else (6,3)
        plt.figure(figsize=chart_size)
        ax = sns.barplot(y=frequency_table["count"], x=frequency_table.index, data=frequency_table, legend=False);
        if log:
            ax.set(yscale='log')
    plt.show();

    if not top and not bottom:
        display(frequency_table)
    else:
        if top:
            display(frequency_table.head(top))
        if bottom:
            display(frequency_table.tail(bottom))

def balance_labels(df, label_colum, shuffle = False, seed=None):
    subset = []
    freq_table = getFrequencyTable(df, label_colum)
    if shuffle:
        subset = [df[df[label_colum] == label].sample(n = freq_table["count"].min(), random_state=seed) for label in freq_table.index]
    else:
        subset = [df[df[label_colum] == label][:freq_table["count"].min()] for label in freq_table.index]
        
    df = pd.concat(subset)
    df.sort_index(inplace=True)
    return df

def set_seed(seed=None):
    if seed != None:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()


def preprocess_data(df, img_width, img_height, path_column, label_column, normalizer=False, seed=None, validation_set = False,
                    keep_aspect_ratio = True, random_flip=False, random_brightness=False, random_contrast=False, random_hue=False, random_saturation=False):
    set_seed(seed)

    # Split DataSets
    X, y, class_names = get_img_xy(df = df, path_column = path_column, label_column = label_column, img_width=img_width, img_height=img_height, color_mode="rgb", seed=seed,
                                   keep_aspect_ratio = keep_aspect_ratio, random_flip=random_flip, random_brightness=random_brightness, random_contrast=random_contrast, random_hue=random_hue, random_saturation=random_saturation)
    if validation_set:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.15, stratify=y, random_state=seed)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.15, stratify=y, random_state=seed)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=(0.15/0.85), stratify=y_train, random_state=seed)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=seed)

    # Preview Data Distribution
    show_labels_distribution(labels = y_train, class_names = class_names)

    # Reshape to add channel dimension
    X_train = X_train.reshape(-1, img_width, img_height, 3)
    X_test = X_test.reshape(-1, img_width, img_height, 3)
    #X_val = X_val.reshape(-1, img_width, img_height, 3)

    # Convert labels to one-hot vectors
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=len(class_names))
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=len(class_names))
    if validation_set:
        y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=len(class_names))

    # Create a preprocessing layer for normalization
    if normalizer == False:
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        X_train = normalization_layer(X_train)
        X_test = normalization_layer(X_test)
        if validation_set:
            X_val = normalization_layer(X_val)
    else:
        X_train = normalizer(X_train)
        X_test = normalizer(X_test)
        if validation_set:
            X_val = normalizer(X_val)

    # Review Shapes
    print("Training data shape X: ", X_train.shape, ", y:", y_train_cat.shape)
    print("Test data shape X: ", X_test.shape, ", y:", y_test_cat.shape)
    if validation_set:
        print("Validation data shape X: ", X_val.shape, ", y:", y_val_cat.shape)
    
    if validation_set:
        return {"train": X_train, "test": X_test, "val": X_val}, {"train": y_train, "test": y_test, "val": y_val}, {"train": y_train_cat, "test": y_test_cat, "val":y_val_cat}, class_names
    else:
        return {"train": X_train, "test": X_test}, {"train": y_train, "test": y_test}, {"train": y_train_cat, "test": y_test_cat}, class_names