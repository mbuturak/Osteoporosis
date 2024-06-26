import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error

# Verileri yükle (kök dizininden)
# y = mx + b
df = pd.read_csv("Data.csv")

df["estrogen"]=df["estrogen"].str.replace("-","0")
df["estrogen"]=pd.to_numeric(df["estrogen"])

df=pd.get_dummies(df,columns=['gender','menopause','osteoporosis-in-the-family','hip-fracture-in-the-family','fracture-history','supplement'],drop_first=True)

y=df["osteoporosis-risk"]
x=df.drop("osteoporosis-risk",axis=1)

# Streamlit uygulamasını oluştur
st.title('Osteoporosis Risk Analysis')
st.write("With this app you can visualize the osteoporosis risk score based on the patient's data.")

# Kullanıcı girişini al
st.sidebar.title('Osteoporosis Risk Analysis')
age = st.sidebar.number_input('Age', value=0)
calcium = st.sidebar.number_input('Calcium', value=0.0)
phosphor = st.sidebar.number_input('Phosphor', value=0.0)
alkaline = st.sidebar.number_input('Alkaline Phosphatase', value=0)
vitamin_d = st.sidebar.number_input('Vitamin D', value=0)
parathormon = st.sidebar.number_input('Parathormon', value=0)
tsh = st.sidebar.number_input('TSH', value=0.0)
estrogen = st.sidebar.number_input('Estrogen', value=0)
testosterone = st.sidebar.number_input('Testosterone', value=0)
gender = st.sidebar.selectbox("Gender",("Male", "Female"))
menopause = st.sidebar.selectbox("Menopause",("True", "False"))
osteoporosis_family = st.sidebar.selectbox("Is there a family history of osteoporosis",("True", "False"))
hip_fracture_family = st.sidebar.selectbox("Hip fracture in the family",("True", "False"))
fracture_history = st.sidebar.selectbox("Fracture history",("True", "False"))
supplement = st.sidebar.selectbox("Supplement",("True", "False"))


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.70,random_state=1)

lm=LinearRegression()
model = lm.fit(x_train,y_train)


# Veri kümesinin boyutlarını göster
total_records = len(df)
st.write(f'Total Record : {total_records}')

# Önceki sütun adları
eski_adlar = ['age', 'calcium', 'phosphor','alkaline-phosphatase','vitamin-d','parathormon','tsh','estrogen','testosterone','osteoporosis-risk','gender_Kadın','menopause_Hayır','osteoporosis-in-the-family_Hayır','hip-fracture-in-the-family_Hayır','fracture-history_Hayır','supplement_Hayır']

# Yeni sütun adları
yeni_adlar = ['Age', 'Calcium', 'Phospor','Alkaline Phospatase','Vitamin D','Parathormon','TSH','Estrogen','Testosterone','Osteoporosis Risk %','Gender','Menopause','Osteoporosis In the Family','Hip Fracture In The Family','Fracture History','Supplement']

# Tüm satırları görüntülemek için max_rows seçeneğini None olarak ayarla
df_gecici = df.rename(columns=dict(zip(eski_adlar, yeni_adlar)))

# Değiştirilmiş DataFrame'i görüntüleme
st.write(df_gecici.head(3).transpose())

# Eğitim ve test kümesi boyutlarını göster
train_records = len(x_train)
test_records = len(x_test)

# Tahmin işlemi ve yeni kayıtları eklemek
results = None
if st.sidebar.button('Send'):
    
    if all([age == 0,alkaline==0,vitamin_d==0,parathormon==0,estrogen==0,testosterone==0]):
        st.warning("At least 1 value must be entered")
    else:

        gender_encoded = 1 if gender == "Female" else 0
        menopause_encoded = 1 if menopause == "True" else 0
        osteoporosis_family_encoded = 1 if osteoporosis_family == "True" else 0
        hip_fracture_family_encoded = 1 if hip_fracture_family == "True" else 0
        fracture_history_encoded = 1 if fracture_history == "True" else 0
        supplement_encoded = 1 if supplement == "True" else 0

        user_input = np.array([[age,calcium, phosphor, alkaline,vitamin_d,parathormon, tsh, estrogen, testosterone,gender_encoded, menopause_encoded, osteoporosis_family_encoded,
                                hip_fracture_family_encoded, fracture_history_encoded, supplement_encoded]])
        
        prediction = model.predict(user_input)
        st.write(f'Prediction Result: <span style="font-size:20px">{prediction[0]:.2f} % You are potentially at risk</span>', unsafe_allow_html=True)

        # Tahmin edilen değerlerle gerçek değerleri içeren bir DataFrame oluştur
        results = pd.DataFrame({'Actual': y_test, 'Predicted': model.predict(x_test)})

# Model skorunu hesapla
score = model.score(x_test, y_test)
cv_scores = cross_val_score(model, x, y, cv=2)  # 2 katlı çapraz doğrulama

# Düğmelere tıklanınca çapraz doğrulama skorlarını, kullanılan kayıt sayılarını ve model doğruluk oranını göster/gizle
show_scores = st.checkbox('Show/Hide Scores')
show_graph = st.checkbox('Show/Hide Graph')
show_result_rates = st.checkbox('Show/Hide Result Rates')

# Çapraz doğrulama skorlarını göster
if show_scores:
    # Çapraz doğrulama skorları
    st.write('Cross Validation Scores:', cv_scores)
    
    # Kullanılan kayıt sayıları
    st.write(f'Number of Training Records Used: {train_records}/{total_records}')
    st.write(f'Model Accuracy Rate: **{score:.2f}**')
    st.write(f'Average Accuracy Rate: **{np.mean(cv_scores):.2f}**')

if show_graph:
    if results is not None:
        
        fig, ax = plt.subplots()
        ax.scatter(results['Actual'], results['Predicted'])
        ax.plot([results['Actual'].min(), results['Actual'].max()], [results['Actual'].min(), results['Actual'].max()], 'k--', lw=4)
        ax.set_xlabel('Actual %')
        ax.set_ylabel('Predicted %')
        ax.set_title('Actual vs. Predicted')
        plt.show()  # Grafikleri kontrol etmek için plt.show() ekleyin
        st.pyplot(fig)  # st.pyplot(fig) yerine st.pyplot() kullanın

        st.write("Success percentages of the data used for training;", results)
    else:
        st.warning("The prediction has not yet been realized")

if show_result_rates:    
    # Hata Yakalama
    df_hata = pd.DataFrame()
    df_hata['Actual'] = y
    y_tahmin = model.predict(x)
    df_hata['Predicted'] = y_tahmin
    df_hata['Difference'] = y - y_tahmin

    df_hata['Mean Squared Error'] = mean_squared_error(y, y_tahmin)
    df_hata['Mean Absolute Error'] = mean_absolute_error(y, y_tahmin)
    df_hata['Mean Percentage Error'] = mean_absolute_percentage_error(y, y_tahmin)
    
    # Ortalamaları hesapla
    means = df_hata.mean()

    # Sonuçları göster
    st.write("Mean Squared Error:", means['Mean Squared Error'])
    st.write("Mean Absolute Error:", means['Mean Absolute Error'])
    st.write("Mean Percentage Error:", means['Mean Percentage Error'])

    # Hatanın dağılımını göster
    st.write("Error Distribution:", df_hata)

