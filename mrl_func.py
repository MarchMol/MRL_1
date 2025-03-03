from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

def linear_analysis(X,ind_label):
    # Var definition
    mrl_X = X[[ind_label,'SalePrice']]
    mrl_y = X['SalePrice']
    X_train, X_test,y_train, y_test= train_test_split(
        mrl_X, mrl_y,
        test_size=0.3, train_size=0.7,
        random_state=42)

    ind = X_train[ind_label].values.reshape(-1,1)
    ind_t = X_test[ind_label].values.reshape(-1,1)
    dep = y_train.values.reshape(-1,1)
    dep_t = y_test.values.reshape(-1,1)
    plt.figure(figsize=(5,5))
    plt.scatter(ind,dep, label='Train')
    plt.scatter(ind_t,dep_t, color='lightgreen',label='Test')
    plt.legend()
    plt.show()
    
    # Model
    lm = LinearRegression()
    lm.fit(ind, dep) # Entreno en train
    dep_pred = lm.predict(ind_t) # Prediccion de test
    m = lm.coef_[0][0]
    c = lm.intercept_[0]
    # Valores adicionales
    eq = r'sale_price = %0.4f*lot_area % + 0.4f '%(m,c)
    mse =mean_squared_error(dep_t,dep_pred)
    r2 =r2_score(dep_t,dep_pred)
    plt.figure(figsize=(5,5))
    plt.scatter(ind_t, dep_t, label='Prueba')
    plt.plot(ind_t, dep_pred, color='r', label='Prediccion')
    plt.title("Lot Area VS Sale Price (Prueba MRL)")
    plt.xlabel("Area First Floor (ft^2)")
    plt.ylabel("Sale Price ($)")
    plt.legend()
    plt.show()
    print("EQ: "+eq)
    print("Mean Squared Error: %.2f"%mse)
    print("R squared: %.2f"%r2)
    
    # Residuales
    residuales = dep_t - dep_pred
    differences = [abs(t - p) for t, p in zip(dep_t, dep_pred)]
    max_index = differences.index(max(differences))
    ## Max
    max_real_value = dep_t[max_index]
    max_pred_value = dep_pred[max_index]
    max_difference = differences[max_index]
    print(f"Max Real: {max_real_value}")
    print(f"Max Predicho: {max_pred_value}")
    print(f"Max Diferencia: {max_difference}")
    
    ## Grafico de residules
    plt.figure(figsize=(5,5))
    plt.scatter(ind_t,residuales)
    plt.title("Lot Area VS Sale Price (Grafico de Residuales)")
    plt.xlabel("Residual")
    plt.ylabel("Variacion Independiente")
    plt.axhline(0,color='blue')

    ## Histograma Residuales
    plt.figure(figsize=(5,5))
    sns.histplot(residuales, kde=True)
    plt.xlabel("Residuales")
    plt.title("Distribución de los Residuales")
    plt.show()
    plt.figure(figsize=(5,5))
    plt.boxplot(residuales)
    plt.show()
    ## Prueba de Normalidad
    ## Descripcion OLS
    ind = sm.add_constant(ind)  # Agregar intercepto
    est = sm.OLS(dep, ind)
    est2 = est.fit()
    print(est2.summary())