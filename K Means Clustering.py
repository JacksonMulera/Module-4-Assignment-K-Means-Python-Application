import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

class KMeansClusterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("K-Means Clustering App")

        self.frame = ttk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # File selection
        self.file_label = ttk.Label(self.frame, text="Select CSV file:")
        self.file_label.grid(row=0, column=0, padx=5, pady=5)

        self.file_button = ttk.Button(self.frame, text="Browse", command=self.load_file)
        self.file_button.grid(row=0, column=1, padx=5, pady=5)

        # Number of clusters
        self.cluster_label = ttk.Label(self.frame, text="Number of clusters:")
        self.cluster_label.grid(row=1, column=0, padx=5, pady=5)

        self.cluster_entry = ttk.Entry(self.frame)
        self.cluster_entry.grid(row=1, column=1, padx=5, pady=5)

        self.cluster_button = ttk.Button(self.frame, text="Run K-Means", command=self.run_kmeans)
        self.cluster_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

        self.elbow_button = ttk.Button(self.frame, text="Show Elbow Plot", command=self.show_elbow_plot)
        self.elbow_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

        # Plot area
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

        self.data = None

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.data = pd.read_csv(file_path)
            messagebox.showinfo("File Loaded", f"Loaded data with {self.data.shape[0]} rows and {self.data.shape[1]} columns.")

    def run_kmeans(self):
        if self.data is not None:
            try:
                n_clusters = int(self.cluster_entry.get())
                numerical_data = self.data[['Rating', 'Salary']]
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numerical_data)
                kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(scaled_data)
                self.data['Cluster'] = kmeans.labels_
                self.plot_clusters(scaled_data, kmeans.labels_)
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number of clusters.")
        else:
            messagebox.showerror("Error", "Data not loaded. Please load a data file.")

    def show_elbow_plot(self):
        if self.data is not None:
            numerical_data = self.data[['Rating', 'Salary']]
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numerical_data)
            inertias = []
            range_of_clusters = range(1, 11)  # You can adjust the range as needed
            for k in range_of_clusters:
                kmeans = KMeans(n_clusters=k, random_state=0)
                kmeans.fit(scaled_data)
                inertias.append(kmeans.inertia_)
            plt.figure()
            plt.plot(range_of_clusters, inertias, marker='o')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Inertia')
            plt.title('Elbow Method For Optimal k')
            plt.show()
        else:
            messagebox.showerror("Error", "Data not loaded. Please load a data file to analyze.")

    def plot_clusters(self, data, labels):
        self.ax.clear()
        scatter = self.ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o')
        self.ax.set_xlabel('Rating')
        self.ax.set_ylabel('Salary')
        self.ax.set_title("K-Means Clustering")
        handles, _ = scatter.legend_elements()
        self.ax.legend(handles, [f"Cluster {i}" for i in range(len(handles))])
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = KMeansClusterApp(root)
    root.mainloop()
