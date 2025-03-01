import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.stats.diagnostic as diag
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
import random
import sklearn.preprocessing
import pyclustertend
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

columnas_a_filtrar = [
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "2ndFlrSF",
    "LowQualFinSF", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
    "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal"
]

def prueba_de_normalidad(column, column_name):
    column = column.dropna()  # Eliminar valores NaN

    # Si la columna está en la lista, eliminar los valores 0
    if column_name in columnas_a_filtrar:
        column = column[column != 0]

    # Verificar que quedan datos para el análisis
    if column.empty:
        print(f"La columna '{column_name}' quedó vacía después de filtrar NaN y/o ceros. No se puede realizar la prueba de normalidad.\n")
        return

    ks_statistic, p_value = stats.kstest(column, 'norm', args=(np.mean(column), np.std(column)))

    # Imprimir los resultados
    print(f"Estadístico de prueba (ks_statistic) = {ks_statistic:.20f}")
    print(f"p-value = {p_value:.20f}")

    alpha = 0.05
    if p_value < alpha:
        print(f"Se rechaza la hipótesis nula: los datos de '{column_name}' NO provienen de una distribución normal.\n")
    else:
        print(f"No se rechaza la hipótesis nula: los datos de '{column_name}' parecen provenir de una distribución normal.\n")
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    # Graficar histograma
    axes[0].hist(column, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_title(f'Histograma de {column_name}')
    axes[0].set_xlabel(column_name)
    axes[0].set_ylabel('Frecuencia')

    # Graficar boxplot
    axes[1].boxplot(column)
    axes[1].set_title(f'Boxplot de {column_name}')
    axes[1].set_ylabel(column_name)

    # Mostrar gráficos
    plt.tight_layout()
    plt.show()
        
def frecuencias(column, column_name):
    frecuencia = column.value_counts()

    # Graficar frecuencias
    plt.figure(figsize=(10, 6))
    frecuencia.plot(kind='bar', color='skyblue', edgecolor='black', alpha=0.7)

    plt.title(f'Gráfico de {column_name}')
    plt.xlabel('Categorías')  # Categorías en el eje X
    plt.ylabel('Frecuencia')   # Frecuencia en el eje Y
    plt.xticks(range(len(frecuencia)), frecuencia.index, rotation=45, ha='right')  # Etiquetas del eje X
    plt.tight_layout()
    plt.show()
    
def trans_categorical(df):
    # Variables ordinales con asignación de valores
    ordinal_mappings = {
        'ExterQual': {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
        'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'BsmtQual': {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
        'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'KitchenQual': {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
        'FireplaceQu': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'GarageFinish': {'Unf': 1, 'RFn': 2, 'Fin': 3},
        'PoolQC': {'Fa': 1, 'Gd': 2, 'Ex': 3},
        'Fence': {'MnWw': 1, 'MnPrv': 2, 'GdWo': 3, 'GdPrv': 4}
    }
    
    for col, mapping in ordinal_mappings.items():
        df[col] = df[col].map(mapping).fillna(0)
    
    # Variables nominales -> Label Encoding o One-Hot Encoding
    nominal_cols = [
        'Alley', 'LotShape', 'LotConfig', 'Neighborhood', 'BldgType',
        'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd',
        'MasVnrType', 'Foundation', 'BsmtFinType1', 'GarageType'
    ]
    
    for col in nominal_cols:
        if df[col].nunique() > 4:
            df = pd.get_dummies(df, columns=[col], drop_first=True)  # One-Hot Encoding
        else:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))  # Label Encoding
    
    return df

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def trans_categorical(df):
    df = df.copy()  # Evitar modificar el dataframe original
    
    # Eliminar las variables que no queremos en el análisis de clusters
    drop_columns = [
        'Id', 'PoolArea', 'MiscVal', 'BsmtFinSF2', 'BsmtFinSF1', 'MasVnrArea',
        'BsmtUnfSF', '2ndFlrSF', 'LowQualFinSF', 'WoodDeckSF', 'OpenPorchSF',
        'EnclosedPorch', '3SsnPorch', 'ScreenPorch'
    ]
    df = df.drop(columns=drop_columns, errors='ignore')
    
    # Definir las variables categóricas a transformar
    ordinal_mappings = {
        'ExterQual': {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
        'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'BsmtQual': {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
        'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'KitchenQual': {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
        'FireplaceQu': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'GarageFinish': {'Unf': 1, 'RFn': 2, 'Fin': 3},
        'PoolQC': {'Fa': 1, 'Gd': 2, 'Ex': 3},
        'Fence': {'MnWw': 1, 'MnPrv': 2, 'GdWo': 3, 'GdPrv': 4}
    }
    
    for col, mapping in ordinal_mappings.items():
        df[col] = df[col].map(mapping).fillna(0)
    
    # Variables nominales -> Label Encoding o One-Hot Encoding
    nominal_cols = [
        'Alley', 'LotShape', 'LotConfig', 'Neighborhood', 'BldgType',
        'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd',
        'MasVnrType', 'Foundation', 'BsmtFinType1', 'GarageType'
    ]
    
    for col in nominal_cols:
        if df[col].nunique() > 4:
            df = pd.get_dummies(df, columns=[col], drop_first=True)  # One-Hot Encoding
        else:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))  # Label Encoding
    
    return df

def preprocess(df):

    # Copia del DataFrame para evitar modificar el original
    df_clean = df.copy()

    # Seleccionar solo las columnas numéricas
    numeric_cols = df_clean.select_dtypes(include=[float, int]).columns

    # Calcular Q1 (percentil 25) y Q3 (percentil 75)
    Q1 = df_clean[numeric_cols].quantile(0.25)
    Q3 = df_clean[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1

    # Filtrar valores dentro del rango intercuartílico
    df_clean = df_clean[~((df_clean[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                           (df_clean[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

    return df_clean

def test_random_clusters(X, num_tests=10, min_cols=2, max_cols=None, random_state=None):
    # Semilla aleatoria (si se pasa random_state, se asegura reproducibilidad)
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    # Si no se especifica max_cols, usar el número total de columnas
    if max_cols is None or max_cols > X.shape[1]:
        max_cols = X.shape[1]
    
    results = []
    
    for _ in range(num_tests):
        # Seleccionar aleatoriamente un número de columnas entre min_cols y max_cols
        num_selected = random.randint(min_cols, max_cols)
        selected_cols = random.sample(list(X.columns), num_selected)
        
        # Seleccionar el subconjunto de datos de esas columnas aleatorias
        X_subset = X[selected_cols]
        
        # Eliminar filas con NaN (si existieran)
        X_subset_clean = X_subset.dropna()

        # Si no quedan filas después de eliminar NaN, continuar con la siguiente iteración
        if X_subset_clean.empty:
            continue
        
        # Normalizar los datos seleccionados
        X_scaled = sklearn.preprocessing.scale(X_subset_clean)

        # Calcular la estadística de Hopkins
        hopkins_stat = pyclustertend.hopkins(X_scaled, len(X_scaled))

        # Almacenar los resultados
        results.append({
            "Columns": selected_cols,
            "Hopkins_Stat": hopkins_stat
        })
    
    # Si se generaron resultados, ordenar por la estadística de Hopkins
    if results:
        results_df = pd.DataFrame(results).sort_values(by="Hopkins_Stat", ascending=False)
        return results_df
    else:
        return pd.DataFrame()  # Retornar un DataFrame vacío si no hay resultados
    
def elbow(X_scale):
    random.seed(123)
    numeroClusters = range(1,10)
    wcss = []
    for i in numeroClusters:
        kmeans = sklearn.cluster.KMeans(n_clusters=i)
        kmeans.fit(X_scale)
        wcss.append(kmeans.inertia_)

    plt.plot(numeroClusters, wcss, marker='o')
    plt.xticks(numeroClusters)
    plt.xlabel("K clusters")
    plt.ylabel("WSS")
    plt.title("Gráfico de Codo")
    plt.show()

def sillhouette(range_n_clusters, X):

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

    plt.show()
    
def plotFeatures1(X, cluster_n):
    for i in range(0,cluster_n):
        cl0 = X[X['Cluster']==i]
        fi, ax = plt.subplots(ncols=2,nrows=2, figsize=(12,8))
        fi.suptitle(f'Características de Cluster {i+1}', fontsize=14, fontweight='bold')  # Overarching title
        ax[0,0].set_title(f'Histograma SalePrice')
        ax[0,0].hist(cl0['SalePrice'],bins=20)
        ax[1,0].set_title(f'Histograma MoSold')
        ax[1,0].hist(cl0['MoSold'],bins=20)
        ax[0,1].set_title(f'Histograma LotFrontage')
        ax[0,1].hist(cl0['LotFrontage'],bins=20)
        plt.show()

def plotFeatures2(X, cluster_n):
    for i in range(0,cluster_n):
        cl0 = X[X['Cluster']==i]
        fi, ax = plt.subplots(ncols=2,nrows=2, figsize=(12,8))
        fi.suptitle(f'Características de Cluster {i+1}', fontsize=14, fontweight='bold')  # Overarching title
        ax[0,0].set_title(f'Histograma LotArea')
        ax[0,0].hist(cl0['LotArea'],bins=20)
        ax[1,0].set_title(f'Histograma SalePrice')
        ax[1,0].hist(cl0['SalePrice'],bins=20)
        ax[0,1].set_title(f'Histograma MoSold')
        ax[0,1].hist(cl0['MoSold'],bins=20)
        plt.show()
        
def plotFeatures3(X, cluster_n):
    for i in range(0,cluster_n):
        cl0 = X[X['Cluster']==i]
        fi, ax = plt.subplots(ncols=2,nrows=2, figsize=(12,8))
        fi.suptitle(f'Características de Cluster {i+1}', fontsize=14, fontweight='bold')  # Overarching title
        ax[0,0].set_title(f'Histograma LotArea')
        ax[0,0].hist(cl0['LotArea'],bins=20)
        ax[1,0].set_title(f'Histograma MoSold')
        ax[1,0].hist(cl0['MoSold'],bins=20)
        ax[0,1].set_title(f'Histograma GarageArea')
        ax[0,1].hist(cl0['GarageArea'],bins=20)
        plt.show()