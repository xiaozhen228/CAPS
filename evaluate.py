import torch
import torch.nn.functional as F
from utils.dice_score import dice_loss
from utils.dice_score import multiclass_dice_coeff, dice_coeff


def evaluate(net, dataloader,dataloader_test, device,class_num=2):
    net.eval()
    num_val_batches = len(dataloader)
    print("需要验证的num_val_batches:%f"%num_val_batches)
    dice_score = 0
    loss = 0
    dataloader = iter(dataloader)
    

    # iterate over the validation set
    for i in range(num_val_batches):
        data = next(dataloader)
        image, mask_true = data['image'], data['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true1 = F.one_hot(mask_true.clone().detach(), class_num).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            loss += torch.nn.CrossEntropyLoss()(mask_pred, mask_true) \
                    + dice_loss(F.softmax(mask_pred, dim=1).float(),
                                F.one_hot(mask_true, class_num).permute(0, 3, 1, 2).float(),
                                multiclass=True)
            # convert to one-hot format
            if class_num == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true1, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), class_num).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true1[:, 1:, ...], reduce_batch_first=False)

           

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches, loss/ dataloader_test.num_of_samples()
