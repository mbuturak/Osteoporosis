import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

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
gender = st.sidebar.selectbox("Gender",("Male", "Female"))
menopause = st.sidebar.selectbox("Menopause",("True", "False"))
osteoporosis = st.sidebar.selectbox("Is there a family history of osteoporosis",("True", "False"))
hip_fracture = st.sidebar.selectbox("Hip fracture in the family",("True", "False"))
fracture_history = st.sidebar.selectbox("Fracture history",("True", "False"))
supplement = st.sidebar.selectbox("Supplement",("True", "False"))
calcium = st.sidebar.number_input('Calcium', value=0.0)
phospor = st.sidebar.number_input('Phospor', value=0.0)
alkaline = st.sidebar.number_input('Alkaline Phosphatase', value=0)
vitamin_d = st.sidebar.number_input('Vitamin D', value=0)
parathormon = st.sidebar.number_input('Parathormon', value=0)
tsh = st.sidebar.number_input('TSH', value=0.0)
estrogen = st.sidebar.number_input('Estrogen', value=0)
testosterone = st.sidebar.number_input('Testosterone', value=0)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.70,random_state=1)
lm=LinearRegression()
model = lm.fit(x_train,y_train)
model.score(x_test,y_test)

# Tahmin işlemi ve yeni kayıtları eklemek
if st.sidebar.button('Hesapla'):
    
    if all([age == 0,calcium==0,phospor==0,alkaline==0,vitamin_d==0,parathormon==0,tsh==0,estrogen==0,testosterone==0]):
        st.warning("At least 1 value must be entered")
    else:
        user_input = np.array([[age,gender,menopause,osteoporosis,
                                hip_fracture,fracture_history,supplement,
                                calcium,phospor,alkaline,vitamin_d,parathormon,tsh,estrogen]])

        prediction = model.predict(user_input)

        st.write(f'Prediction Result: <span style="color:red;font-size:20px">{prediction[0]}</span>', unsafe_allow_html=True)



# Veri kümesinin boyutlarını göster
total_records = len(df)
st.write(f'Total Record : {total_records}')

# Eğitim ve test kümesi boyutlarını göster
train_records = len(x_train)
test_records = len(x_test)

# Model skorunu hesapla
score = model.score(x_test, y_test)
cv_scores = cross_val_score(model, x, y, cv=5)  # 5 katlı çapraz doğrulama

# Düğmelere tıklanınca çapraz doğrulama skorlarını, kullanılan kayıt sayılarını ve model doğruluk oranını göster/gizle
show_scores = st.checkbox('Show/Hide Scores')

# Çapraz doğrulama skorlarını göster
if show_scores:
    # Çapraz doğrulama skorları
    st.write('Cross Validation Scores:', cv_scores)
    
    # Kullanılan kayıt sayıları
    st.write(f'Number of Training Records Used: {train_records}/{total_records}')
    st.write(f'Model Accuracy Rate: **{score:.2f}**')
    st.write(f'Average Accuracy Rate: {np.mean(cv_scores):.2f}')

# Düğmeye tıklanınca ağacın görselleştirmesini göster/gizle
show_tree_plot = st.checkbox('Show/Hide Two-Dimensional Chart')

# Karar ağacı görselleştirmesini göster
if show_tree_plot:
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Görselin boyutunu ve çözünürlüğünü ayarla
    plt.figure(figsize=(65, 35), dpi=300)
    plot_tree(model, filled=True, feature_names=x_train.columns.tolist(), max_depth=3)

    # Görselin çıktısını ayarla
    st.pyplot(bbox_inches='tight')


def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age):
    try:
        # Modeli eğitiyorsanız bu kısmı bu fonksiyona taşıyabilirsiniz
        model = DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=2)
        model.fit(x_train, y_train)  # x_train ve y_train verileri kullanılarak eğitim yapılıyor

        user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
        prediction = model.predict(user_input)

        if prediction[0] == 0:
            result = "Diyabet değil"
        else:
            result = "Diyabet"

        return result
    except Exception as e:
        return str(e)