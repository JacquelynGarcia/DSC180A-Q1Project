from code import load_datasets

def main():
    train_path = "data/train2.tsv"
    val_path   = "data/val2.tsv"
    test_path  = "data/test2.tsv"

    df_train, df_val, df_test = load_datasets(train_path, val_path, test_path)
    
    

if __name__ == "__main__":
    main()