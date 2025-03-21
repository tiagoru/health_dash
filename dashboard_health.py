import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
import scipy.stats as stats

# Streamlit App Title
st.title("📊ViradaHealth Multi-Subject Signal Analysis Dashboard")

# File Upload (Multiple Files Supported)
uploaded_files = st.file_uploader("📂 Upload one or multiple TXT files", type=["txt"], accept_multiple_files=True)

if uploaded_files:
    all_stats = []  # Store stats for multiple files
    subject_names = []  # Store subject names
    comparison_data = {}  # Dictionary to hold comparison data

    for uploaded_file in uploaded_files:
        subject_name = uploaded_file.name
        subject_names.append(subject_name)

        st.subheader(f"📄 File: {subject_name}")

        # Read the file
        df = pd.read_csv(uploaded_file, sep='\t', header=None)

        # Split the second column (index 1) by comma into separate columns
        values = df[1].str.split(', ', expand=True).astype(float)

        # Combine the ID column with the new values
        df = pd.concat([df[0], values], axis=1)

        # Rename the columns
        df.columns = ['ID', 'Val1', 'Val2', 'Val3', 'Val4', 'Val5', 'Val6']

        # Select which signal to analyze
        selected_col = st.selectbox(f"📌 Select a column for {subject_name}:", ['Val1', 'Val2', 'Val3', 'Val4', 'Val5', 'Val6'])

        # Ignore first 100 seconds
        df_filtered = df[df['ID'] > 100]
        x = df_filtered['ID'].values
        y = df_filtered[selected_col].values

        # Raw Signal Plot
        st.subheader("📈 Raw Signal Data")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['ID'], df[selected_col], label="Raw Signal", color="blue")
        ax.set_title(f"Raw Signal for {selected_col}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.grid()
        st.pyplot(fig)

        # Detect valleys (lower plateaus)
        valleys, _ = find_peaks(-y, height=-np.mean(y) - np.std(y))

        # Detect peaks
        peaks, _ = find_peaks(y, height=np.mean(y) + np.std(y))

        # Identify valid peaks (one per valley)
        valid_peaks = []
        peak_durations = []
        plateau_durations = []

        for i in range(len(valleys) - 1):
            # Find peaks between valleys
            peaks_in_valley = [p for p in peaks if valleys[i] < p < valleys[i+1]]

            if peaks_in_valley:
                highest_peak = max(peaks_in_valley, key=lambda p: y[p])  # Select highest peak

                valid_peaks.append(highest_peak)  # Store valid peak
                peak_durations.append(x[valleys[i+1]] - x[valleys[i]])  # Peak duration
                plateau_durations.append(x[valleys[i+1]] - x[valleys[i]])  # Plateau duration

        # Compute Signal Complexity (Entropy)
        signal_entropy = stats.entropy(np.histogram(y, bins=10)[0])

        # Store comparison data
        comparison_data[subject_name] = {
            "Total Peaks": len(valid_peaks),
            "Mean Peak Duration": np.mean(peak_durations) if peak_durations else 0,
            "Mean Plateau Duration": np.mean(plateau_durations) if plateau_durations else 0,
            "Total Time in Plateaus": sum(plateau_durations),
            "Total Time in Peaks": sum(peak_durations),
            "Signal Entropy": signal_entropy,
            "Signal Variability": np.std(y)
        }

        # Show descriptive statistics in a table
        st.subheader("📊 Descriptive Statistics")
        stats_df = pd.DataFrame([comparison_data[subject_name]])
        st.table(stats_df)

        # Processed Signal Plot
        st.subheader("📈 Processed Signal with Peaks and Valleys")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(x, y, label="Processed Signal", color="orange")
        ax.plot(x[valid_peaks], y[valid_peaks], "ro", label="Valid Peaks")
        ax.plot(x[valleys], y[valleys], "go", label="Valleys")
        ax.legend()
        ax.set_title(f"Processed Signal Analysis for {selected_col}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.grid()
        st.pyplot(fig)

    # ---------------- Multiple Subject Comparison ----------------
    if len(uploaded_files) > 1:
        st.subheader("📊 Multi-Subject Comparison")

        # Convert comparison data into DataFrame
        comp_df = pd.DataFrame(comparison_data).T

        # Display as table
        st.write("### 📋 Comparison Table")
        st.table(comp_df)

        # Peaks Comparison
        st.write("### 📊 Number of Peaks per Subject")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=comp_df.index, y=comp_df["Total Peaks"], ax=ax, palette="coolwarm")
        ax.set_ylabel("Number of Peaks")
        ax.set_title("Comparison of Peak Counts")
        st.pyplot(fig)

        # Time in Peaks vs. Plateaus (Stacked Bar)
        st.write("### 📊 Time Spent in Peaks vs. Plateaus")
        fig, ax = plt.subplots(figsize=(8, 5))
        comp_df[["Total Time in Plateaus", "Total Time in Peaks"]].plot(kind="bar", stacked=True, ax=ax, colormap="coolwarm")
        ax.set_ylabel("Time")
        ax.set_title("Comparison of Time Spent in Peaks vs. Plateaus")
        st.pyplot(fig)

        # Boxplot for Peak & Plateau Durations
        st.write("### 📦 Peak & Plateau Duration Comparison")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=comp_df[["Mean Peak Duration", "Mean Plateau Duration"]], ax=ax)
        ax.set_title("Boxplot of Peak & Plateau Durations")
        st.pyplot(fig)

        # Signal Complexity (Entropy) Comparison
        st.write("### 🔬 Signal Complexity Comparison (Entropy)")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=comp_df.index, y=comp_df["Signal Entropy"], ax=ax, palette="magma")
        ax.set_ylabel("Entropy")
        ax.set_title("Comparison of Signal Complexity")
        st.pyplot(fig)

