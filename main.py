import pandas as pd
import joblib

# Paths
DATA_PATH = "C:/Users/Dell/OneDrive/Desktop/Unsupervised Learning/Data/raw/CC GENERAL.csv"
MODEL_PATH = "C:/Users/Dell/OneDrive/Desktop/Unsupervised Learning/models/customer_segmentation_pipeline.pkl"
OUTPUT_PATH = "C:/Users/Dell/OneDrive/Desktop/Unsupervised Learning/Data/processed"


def main():

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    print("Loading pipeline...")
    pipeline = joblib.load(MODEL_PATH)

    print("Generating customer segments...")
    clusters = pipeline.predict(df)

    df["CLUSTER"] = clusters

    print("Saving results...")
    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("Segmentation completed successfully!")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()