import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

# GUI Imports
import tkinter as tk
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

warnings.filterwarnings('ignore')

# Set application theme and style
ctk.set_appearance_mode("System")  # Options: "Dark", "Light", "System"
ctk.set_default_color_theme("green")  # Options: "blue", "green", "dark-blue"

class AirQualityApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("🍃 Air Quality Analytics & Prediction")
        self.geometry("1100x680")
        
        # Load data and train model behind the scenes
        self.load_and_train()

        # Create layout frames
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.pack(side="left", fill="y")
        
        self.main_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="transparent")
        self.main_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        self.setup_sidebar()
        self.show_dashboard_view() # Default view

    def load_and_train(self):
        try:
            self.df = pd.read_csv('air_quality_dataset.csv')
            X = self.df.drop('Quality', axis=1)
            y = self.df['Quality']
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)
        except FileNotFoundError:
            print("Error: air_quality_dataset.csv not found.")
            # Dummy data fallback for testing if file missing
            data = np.random.rand(100, 6) * 100
            quality = np.random.randint(1, 6, 100)
            self.df = pd.DataFrame(data, columns=['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3'])
            self.df['Quality'] = quality
            self.load_and_train()

    def setup_sidebar(self):
        # App Title
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Air Quality AI", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.pack(padx=20, pady=30)

        # Navigation Buttons
        self.btn_dash = ctk.CTkButton(self.sidebar_frame, text="📊 Dashboard", command=self.show_dashboard_view)
        self.btn_dash.pack(padx=20, pady=10, fill="x")

        self.btn_pred = ctk.CTkButton(self.sidebar_frame, text="🔮 Predict Quality", command=self.show_prediction_view)
        self.btn_pred.pack(padx=20, pady=10, fill="x")
        
        # Appearance mode toggles at bottom
        self.appearance_label = ctk.CTkLabel(self.sidebar_frame, text="Theme Mode:", anchor="w")
        self.appearance_label.pack(side="bottom", padx=20, pady=(0, 5))
        self.appearance_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Dark", "Light", "System"], command=ctk.set_appearance_mode)
        self.appearance_optionemenu.pack(side="bottom", padx=20, pady=(0, 20))
        self.appearance_optionemenu.set("System")

    def clear_main_frame(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def show_dashboard_view(self):
        self.clear_main_frame()
        
        # Header text
        header = ctk.CTkLabel(self.main_frame, text="Data Exploratory Insights", font=ctk.CTkFont(size=24, weight="bold"))
        header.pack(anchor="w", pady=(0, 20))

        # KPI metric badges row
        kpi_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        kpi_frame.pack(fill="x", pady=(0, 20))
        
        m1 = ctk.CTkFrame(kpi_frame, height=70)
        m1.pack(side="left", expand=True, fill="both", padx=5)
        ctk.CTkLabel(m1, text="Total Records", font=ctk.CTkFont(size=12)).pack(pady=(10,0))
        ctk.CTkLabel(m1, text=str(self.df.shape[0]), font=ctk.CTkFont(size=18, weight="bold")).pack()

        m2 = ctk.CTkFrame(kpi_frame, height=70)
        m2.pack(side="left", expand=True, fill="both", padx=5)
        ctk.CTkLabel(m2, text="Model Accuracy", font=ctk.CTkFont(size=12)).pack(pady=(10,0))
        acc = accuracy_score(self.y_test, self.y_pred)
        ctk.CTkLabel(m2, text=f"{acc*100:.1f}%", font=ctk.CTkFont(size=18, weight="bold"), text_color="#2E7D32").pack()

        # Build Charts inside UI
        chart_frame = ctk.CTkFrame(self.main_frame)
        chart_frame.pack(fill="both", expand=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor('#242424' if ctk.get_appearance_mode() == "Dark" else '#ebebeb')
        
        # 1. Bar Chart
        quality_counts = self.df['Quality'].value_counts().sort_index()
        quality_labels = ['Hazardous', 'Poor', 'Moderate', 'Good', 'Excellent'][:len(quality_counts)]
        colors = ['#ef5350', '#ff9800', '#ffee58', '#9ccc65', '#66bb6a'][:len(quality_counts)]
        
        ax1.bar(quality_labels, quality_counts.values, color=colors, alpha=0.85, edgecolor='black')
        ax1.set_title('Distribution Category Count', color='gray', fontweight='bold')
        ax1.tick_params(colors='gray')
        
        # 2. Correlation Matrix Heatmap
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax2, cbar=False, annot_kws={"size": 8})
        ax2.set_title('Feature Correlation Matrix', color='gray', fontweight='bold')
        ax2.tick_params(colors='gray')
        
        plt.tight_layout()

        # Embed fig into Tkinter
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def show_prediction_view(self):
        self.clear_main_frame()

        header = ctk.CTkLabel(self.main_frame, text="Real-Time Diagnostics Simulator", font=ctk.CTkFont(size=24, weight="bold"))
        header.pack(anchor="w", pady=(0, 20))

        # Create Input form grid arrangement
        form_frame = ctk.CTkFrame(self.main_frame)
        form_frame.pack(fill="x", padx=10, pady=10)

        # We will dynamically map labels and store slider tracking variables
        self.sliders = {}
        pollutants = [
            ("PM2.5 (µg/m³)", 0, 500, 35),
            ("PM10 (µg/m³)", 0, 500, 50),
            ("NO2 (ppb)", 0, 200, 25),
            ("SO2 (ppb)", 0, 200, 15),
            ("CO (ppm)", 0, 50, 1),
            ("O3 (ppb)", 0, 200, 40)
        ]

        # Arrange form fields into 2 modern clean columns
        for idx, (name, min_v, max_v, default_v) in enumerate(pollutants):
            row = idx // 2
            col = (idx % 2) * 2

            lbl = ctk.CTkLabel(form_frame, text=name, font=ctk.CTkFont(size=13))
            lbl.grid(row=row, column=col, padx=15, pady=15, sticky="w")

            # Value indicator text updated on slider changes
            val_lbl = ctk.CTkLabel(form_frame, text=str(default_v), font=ctk.CTkFont(weight="bold"))
            val_lbl.grid(row=row, column=col+1, padx=(0, 15), pady=15, sticky="e")

            slider = ctk.CTkSlider(form_frame, from_=min_v, to=max_v, number_of_steps=100,
                                   command=lambda val, vl=val_lbl: vl.configure(text=f"{val:.1f}"))
            slider.set(default_v)
            slider.grid(row=row, column=col+1, padx=(20, 40), pady=15, sticky="ew")
            
            self.sliders[name.split(" ")[0].lower()] = slider

        # Run Live Predict Action Button
        btn_predict = ctk.CTkButton(self.main_frame, text="Analyze Air Sample", height=40, font=ctk.CTkFont(size=15, weight="bold"), command=self.execute_prediction)
        btn_predict.pack(fill="x", padx=10, pady=20)

        # Output results notification frame
        self.result_box = ctk.CTkFrame(self.main_frame, height=80, corner_radius=10, fg_color="#2b2b2b")
        self.result_box.pack(fill="x", padx=10, pady=10)
        
        self.result_lbl = ctk.CTkLabel(self.result_box, text="Adjust sliders and press Analyze to get calculations.", font=ctk.CTkFont(size=16))
        self.result_lbl.pack(expand=True, pady=20)

    def execute_prediction(self):
        # Pull values out of slider object elements
        pm25 = self.sliders['pm2.5'].get()
        pm10 = self.sliders['pm10'].get()
        no2 = self.sliders['no2'].get()
        so2 = self.sliders['so2'].get()
        co = self.sliders['co'].get()
        o3 = self.sliders['o3'].get()

        input_data = np.array([[pm25, pm10, no2, so2, co, o3]])
        prediction = self.model.predict(input_data)[0]

        quality_map = {1: ('Hazardous', '#d32f2f'), 
                       2: ('Poor', '#f57c00'), 
                       3: ('Moderate', '#fbc02d'), 
                       4: ('Good', '#388e3c'), 
                       5: ('Excellent', '#2e7d32')}

        status_text, bg_color = quality_map.get(prediction, ("Unknown", "#2b2b2b"))
        
        # Display dynamically configured theme alerts
        self.result_box.configure(fg_color=bg_color)
        self.result_lbl.configure(text=f"Air Condition Assessment: {status_text} (Class Index: {prediction})", text_color="white" if prediction != 3 else "black")


if __name__ == "__main__":
    app = AirQualityApp()
    app.mainloop()