from data_loader import load_and_prepare_dataset

def print_dataset_info(name, train_df, test_df):
    print(f"\n{'='*50}")
    print(f"Dataset: {name}")
    print(f"{'='*50}")
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    print("\nSample from training set:")
    print(train_df.head(1))
    print("\nColumns:", train_df.columns.tolist())

def main():
    # Load both datasets
    print("Loading datasets...")
    
    # Load Bitext
    bitext_train, bitext_test = load_and_prepare_dataset("bitext")
    print_dataset_info("Bitext", bitext_train, bitext_test)
    
    # Load BiToD
    bitod_train, bitod_test = load_and_prepare_dataset("bitod")
    print_dataset_info("BiToD", bitod_train, bitod_test)

if __name__ == "__main__":
    main() 