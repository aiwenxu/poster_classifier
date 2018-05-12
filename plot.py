from helper import plot, load_stats

OUTPUT_FOLDER_NAME = "plots"
INPUT_FOLDER_NAME = "results"

for i in range(6):
    val_acc = load_stats("{}/{}/val_accs.pkl".format(INPUT_FOLDER_NAME, i))
    val_loss = load_stats("{}/{}/val_losses.pkl".format(INPUT_FOLDER_NAME, i))
    plot(val_acc, "{}/{}_val_acc.pdf".format(OUTPUT_FOLDER_NAME, i))
    plot(val_loss, "{}/{}_val_loss.pdf".format(OUTPUT_FOLDER_NAME, i))