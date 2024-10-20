import streamlit as st
import numpy as np
import train_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# Tải và tiền xử lý dữ liệu
X, y = train_model.load_data()
X_train, X_val, X_test, y_train, y_val, y_test, scaler = train_model.preprocess_data(X, y)

# Huấn luyện các mô hình
svm_model, logistic_model, mlp_model, stacking_model = train_model.train_models(X_train, y_train)

# Ánh xạ các mô hình với tên
models = {
    "SVM": svm_model,
    "Logistic Regression": logistic_model,
    "Neural Network": mlp_model,
    "Stacking": stacking_model
}

# Giao diện Streamlit
st.title("Ứng dụng Dự đoán Tiểu đường")

# Sidebar cho lựa chọn mô hình
st.sidebar.title("Chọn Mô Hình")
model_choice = st.sidebar.selectbox("Mô Hình", list(models.keys()))

# Input dữ liệu của người dùng
st.write("## Nhập Dữ Liệu Bệnh Nhân:")

pregnancies = st.number_input('Số lần mang thai', min_value=0, max_value=20, value=6)
glucose = st.number_input('Mức Glucose (mg/dL)', min_value=0, max_value=200, value=148)
blood_pressure = st.number_input('Huyết áp (mmHg)', min_value=0, max_value=150, value=72)
skin_thickness = st.number_input('Độ dày da (mm)', min_value=0, max_value=100, value=35)
insulin = st.number_input('Mức Insulin (µU/mL)', min_value=0, max_value=900, value=0)
bmi = st.number_input('Chỉ số BMI (kg/m²)', min_value=0.0, max_value=60.0, value=33.6)
dpf = st.number_input('Chỉ số Di truyền tiểu đường', min_value=0.0, max_value=2.5, value=0.627)
age = st.number_input('Tuổi ', min_value=1, max_value=100, value=50)

# Dữ liệu đầu vào để dự đoán
input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]).reshape(1, -1)
input_data_scaled = scaler.transform(input_data)

# Nút dự đoán
if st.button('Dự đoán'):
    model = models[model_choice]
    prediction = model.predict(input_data_scaled)
    result = 'Tiểu đường' if prediction == 1 else 'Không Tiểu đường'
    st.write(f"Dự đoán: {result}")

# Hiển thị các chỉ số mô hình và biểu đồ confusion matrix/ROC-AUC
if st.checkbox('Hiển thị Chỉ số Mô Hình và Biểu đồ'):
    st.write("### Các chỉ số và Biểu đồ Mô Hình:")
    
    # Lấy kết quả đánh giá của mô hình được chọn
    model_metrics = train_model.evaluate_models({model_choice: models[model_choice]}, X_test, y_test, X_train, y_train, X_val, y_val)
    
    for model_name, metrics in model_metrics.items():
        st.write(f"**{model_name}:**")
        st.write(f"- Độ chính xác tập huấn luyện: {metrics['train_accuracy']:.2f}")
        st.write(f"- Độ chính xác tập validation: {metrics['val_accuracy']:.2f}")
        st.write(f"- Độ chính xác tập kiểm tra: {metrics['Accuracy']:.2f}")
        st.write(f"- Precision: {metrics['Precision']:.2f}")
        st.write(f"- Recall: {metrics['Recall']:.2f}")
        st.write(f"- F1 Score: {metrics['F1 Score']:.2f}")
        if metrics['ROC-AUC'] is not None:
            st.write(f"- ROC-AUC: {metrics['ROC-AUC']:.2f}")
        
        # Confusion Matrix
        try:
            y_pred_test = models[model_name].predict(X_test)
            conf_matrix = confusion_matrix(y_test, y_pred_test)
            plt.figure(figsize=(6, 4))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix cho {model_name}')
            plt.ylabel('Thực tế')
            plt.xlabel('Dự đoán')
            st.pyplot(plt.gcf())  # Hiển thị đồ thị trên Streamlit
            plt.clf()  # Xóa đồ thị để chuẩn bị cho đồ thị tiếp theo
        except Exception as e:
            st.error(f"Lỗi khi tạo Confusion Matrix: {e}")
        
        # ROC Curve
        try:
            if hasattr(models[model_name], 'predict_proba'):
                fpr, tpr, _ = roc_curve(y_test, models[model_name].predict_proba(X_test)[:, 1])
                plt.figure(figsize=(6, 4))
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {metrics["ROC-AUC"]:.2f})')
                plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                plt.title(f'ROC Curve cho {model_name}')
                plt.xlabel('Tỷ lệ dương tính giả')
                plt.ylabel('Tỷ lệ dương tính thật')
                plt.legend(loc='lower right')
                st.pyplot(plt.gcf())
                plt.clf()
        except Exception as e:
            st.error(f"Lỗi khi tạo ROC Curve: {e}")
