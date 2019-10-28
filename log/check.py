import argparse

parser = argparse.ArgumentParser(description='checker')

parser.add_argument('--text_file', default='', type=str, help="")
args = parser.parse_args()
# train

with open(args.text_file,"r") as f:
    is_train = False
    max_epoch = 0
    train_losses = []
    valid_losses = []

    for i in range(200):
        train_losses.append(0)
        valid_losses.append(0)

    for line in f:
        arr = line.split()
        if len(arr) == 9:
            if not is_train:
                if arr[3] == "(Epoch":
                    epoch_id = int(arr[4].replace(")",""))
                    train_loss = float(arr[6].replace("LOSS:",""))
                    is_train = True
        elif len(arr) == 8:
                if arr[3] == "VALID":
                    valid_loss = float(arr[6].replace("LOSS:",""))
                    # print(arr[6])
                    # print(train_loss, valid_loss, epoch_id)
                    train_losses[epoch_id] = train_loss
                    valid_losses[epoch_id] = valid_loss
                    is_train = False
                    max_epoch = max(max_epoch, epoch_id)

    print("train_losses=", train_losses[:max_epoch])
    print("valid_losses=", valid_losses[:max_epoch])