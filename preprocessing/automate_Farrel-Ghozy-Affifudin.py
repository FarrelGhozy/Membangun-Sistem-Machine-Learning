import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"[INFO] Data loaded from {filepath}")
    print(f"[INFO] Shape: {df.shape}")
    return df


def eda(df: pd.DataFrame, output_dir: str = "eda_output") -> None:
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-darkgrid')

    print("\n[EDA] Dataset Info:")
    print(df.info())
    print(f"\n[EDA] Missing values:\n{df.isnull().sum()}")
    print(f"\n[EDA] Duplicated rows: {df.duplicated().sum()}")
    print(f"\n[EDA] Descriptive statistics:\n{df.describe().T}")

    fig, ax = plt.subplots(figsize=(8, 5))
    target_counts = df['Outcome'].value_counts()
    ax.bar(['Non-Diabetic (0)', 'Diabetic (1)'], target_counts.values,
           color=['#2ecc71', '#e74c3c'], edgecolor='black')
    ax.set_title('Target Distribution', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count')
    for i, v in enumerate(target_counts.values):
        ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/target_distribution.png', dpi=100, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for i, col in enumerate(df.columns):
        axes[i].hist(df[col], bins=30, edgecolor='black', alpha=0.7, color='#3498db')
        axes[i].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_distributions.png', dpi=100, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
    plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_matrix.png', dpi=100, bbox_inches='tight')
    plt.close()

    columns_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    zero_counts = [(df[col] == 0).sum() for col in columns_with_zero]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(columns_with_zero, zero_counts, color='#e67e22', edgecolor='black')
    ax.set_title('Invalid Zero Values per Column', fontsize=14, fontweight='bold')
    ax.set_xlabel('Count')
    for i, v in enumerate(zero_counts):
        ax.text(v + 2, i, str(v), va='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/zero_values.png', dpi=100, bbox_inches='tight')
    plt.close()

    print(f"[INFO] EDA plots saved to {output_dir}/")


def handle_zero_values(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df_clean = df.copy()
    for col in columns:
        median_val = df_clean[col][df_clean[col] != 0].median()
        df_clean[col] = df_clean[col].replace(0, median_val)
        zero_sisa = (df_clean[col] == 0).sum()
        print(f"[PREPROC] {col}: zeros replaced with median ({median_val:.2f}), remaining zeros: {zero_sisa}")
    return df_clean


def preprocessing(
    filepath: str,
    output_dir: str = "diabetes_preprocessing",
    test_size: float = 0.2,
    random_state: int = 42
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(filepath)

    eda(df, output_dir=output_dir)

    columns_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df_clean = handle_zero_values(df, columns_with_zero)

    X = df_clean.drop('Outcome', axis=1)
    y = df_clean['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    feature_names = X.columns.tolist()

    train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    train_df['Outcome'] = y_train.values

    test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    test_df['Outcome'] = y_test.values

    train_df.to_csv(f'{output_dir}/train.csv', index=False)
    test_df.to_csv(f'{output_dir}/test.csv', index=False)
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')

    all_scaled = pd.DataFrame(
        StandardScaler().fit_transform(df_clean.drop('Outcome', axis=1)),
        columns=feature_names
    )
    all_scaled['Outcome'] = df_clean['Outcome'].values
    all_scaled.to_csv(f'{output_dir}/diabetes_clean.csv', index=False)

    print(f"\n[INFO] Preprocessing complete!")
    print(f"[INFO] Output directory: {output_dir}/")
    for f in os.listdir(output_dir):
        fpath = os.path.join(output_dir, f)
        size = os.path.getsize(fpath)
        print(f"  - {f:30s} ({size:,} bytes)")
    print(f"\n[INFO] Train size: {len(train_df)} | Test size: {len(test_df)}")
    print(f"[INFO] Features: {feature_names}")
    print(f"[INFO] Scaling: StandardScaler")


def main():
    parser = argparse.ArgumentParser(description="Pima Indians Diabetes Preprocessing Pipeline")
    parser.add_argument(
        "--input", type=str, default="diabetes_raw/diabetes_with_headers.csv",
        help="Path to raw diabetes CSV file"
    )
    parser.add_argument(
        "--output", type=str, default="diabetes_preprocessing",
        help="Output directory for preprocessed data"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Test split ratio (default: 0.2)"
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    preprocessing(
        filepath=args.input,
        output_dir=args.output,
        test_size=args.test_size,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main()
