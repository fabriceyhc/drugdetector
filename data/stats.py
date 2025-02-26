
if __name__ == "__main__":

    # python -m data.stats

    from ydata_profiling import ProfileReport, compare
    import pandas as pd

    # Load your datasets
    train_df = pd.read_csv("./data/multilabel/train.csv")
    val_df = pd.read_csv("./data/multilabel/val.csv")
    test_df = pd.read_csv("./data/multilabel/test.csv")

    # Create individual profile reports with descriptive titles
    train_report = ProfileReport(train_df, title="Train")
    val_report = ProfileReport(val_df, title="Validation")
    test_report = ProfileReport(test_df, title="Test")

    # Compare the reports in a single, merged report
    comparison_report = compare([train_report, val_report, test_report])

    # Save the combined report to an HTML file
    comparison_report.to_file("./data/multilabel/comparison.html")

    
    # Load your datasets
    train_df = pd.read_csv("./data/singlelabel/train.csv")
    val_df = pd.read_csv("./data/singlelabel/val.csv")
    test_df = pd.read_csv("./data/singlelabel/test.csv")

    # Create individual profile reports with descriptive titles
    train_report = ProfileReport(train_df, title="Train")
    val_report = ProfileReport(val_df, title="Validation")
    test_report = ProfileReport(test_df, title="Test")

    # Compare the reports in a single, merged report
    comparison_report = compare([train_report, val_report, test_report])

    # Save the combined report to an HTML file
    comparison_report.to_file("./data/singlelabel/comparison.html")