import os
import numpy as np
import yaml
import argparse
import torch
import torchmetrics
import pandas as pd
from time import time, sleep
from tqdm import tqdm
from utils import load_configs, test_gpu_cuda, prepare_tensorboard, prepare_optimizer, save_checkpoint, \
    get_logging, load_checkpoints, prepare_saving_dir
from data import prepare_dataloaders_nonfouling
from model import prepare_models
from accelerate import Accelerator
from focal_loss import FocalLoss

# nonfouling is a binary classification QShao 2-2-2024
def train(epoch, accelerator, dataloader, tools, global_step, tensorboard_log, configs):
    # Initialize metrics
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=tools['num_classes'])
    f1_score = torchmetrics.F1Score(num_classes=tools['num_classes'], average='macro', task="multiclass")

    accuracy.to(accelerator.device)
    f1_score.to(accelerator.device)

    tools["optimizer"].zero_grad()

    epoch_loss = 0
    train_loss = 0
    counter = 0

    progress_bar = tqdm(range(global_step, int(np.ceil(len(dataloader) / tools['accum_iter']))),
                        disable=not accelerator.is_local_main_process, leave=False)
    progress_bar.set_description("Steps")

    for i, data in enumerate(dataloader):
        with accelerator.accumulate(tools['net']):
            sequence, labels, sample_weight = data

            outputs = tools['net'](sequence)
            losses = tools['loss_function'](outputs, labels)

            if configs.train_settings.sample_weight:
                # Multiply each sequence loss by the sample weight
                losses = losses * sample_weight.unsqueeze(1)

            # classification loss
            loss = torch.mean(losses)

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(tools["train_batch_size"])).mean()
            train_loss += avg_loss.item() / tools['accum_iter']

            preds = torch.argmax(outputs, dim=1)

            accuracy.update(accelerator.gather(preds).detach(), accelerator.gather(labels).detach())
            f1_score.update(accelerator.gather(preds).detach(), accelerator.gather(labels).detach())

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(tools['net'].parameters(), tools['grad_clip'])

            tools['optimizer'].step()
            tools['scheduler'].step()
            tools['optimizer'].zero_grad()

        if accelerator.sync_gradients:
            if tensorboard_log:
                tools['train_writer'].add_scalar('step loss', train_loss, global_step)
                tools['train_writer'].add_scalar('learning rate', tools['optimizer'].param_groups[0]['lr'], global_step)

            progress_bar.update(1)
            global_step += 1

            counter += 1
            epoch_loss += train_loss
            train_loss = 0

        logs = {"step_loss": loss.detach().item(),
                "lr": tools['optimizer'].param_groups[0]['lr']}
        progress_bar.set_postfix(**logs)

    epoch_acc = accuracy.compute().cpu().item()
    epoch_f1 = f1_score.compute().cpu().item()

    accelerator.log({"train_f1": epoch_f1, "train_acc": epoch_acc}, step=epoch)
    if tensorboard_log:
        tools['train_writer'].add_scalar('accuracy', epoch_acc, epoch)
        tools['train_writer'].add_scalar('f1', epoch_f1, epoch)

    # Reset metrics at the end of epoch
    accuracy.reset()
    f1_score.reset()

    return epoch_loss / counter, epoch_acc, epoch_f1


def valid(epoch, accelerator, dataloader, tools, tensorboard_log):
    tools['net'].eval()

    # Initialize metrics
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=tools['num_classes'])
    macro_f1_score = torchmetrics.F1Score(num_classes=tools['num_classes'], average='macro', task="multiclass")
    f1_score = torchmetrics.F1Score(num_classes=tools['num_classes'], average=None, task="multiclass")

    accuracy.to(accelerator.device)
    macro_f1_score.to(accelerator.device)
    f1_score.to(accelerator.device)

    counter = 0

    progress_bar = tqdm(range(len(dataloader)),
                        disable=not accelerator.is_local_main_process, leave=False)
    progress_bar.set_description("Steps")

    valid_loss = 0
    for i, data in enumerate(dataloader):
        sequence, labels, sample_weight = data

        with torch.inference_mode():
            outputs = tools['net'](sequence)

            losses = tools['loss_function'](accelerator.gather(outputs), accelerator.gather(labels))

            loss = torch.mean(losses)

            preds = torch.argmax(outputs, dim=1)

            accuracy.update(accelerator.gather(preds).detach(), accelerator.gather(labels).detach())
            macro_f1_score.update(accelerator.gather(preds).detach(), accelerator.gather(labels).detach())
            f1_score.update(accelerator.gather(preds).detach(), accelerator.gather(labels).detach())

        counter += 1
        valid_loss += loss.data.item()

        progress_bar.update(1)
        logs = {"step_loss": loss.detach().item(),
                "lr": tools['optimizer'].param_groups[0]['lr']}

        progress_bar.set_postfix(**logs)

    valid_loss = valid_loss / counter

    epoch_acc = accuracy.compute().cpu().item()
    epoch_macro_f1 = macro_f1_score.compute().cpu().item()
    epoch_f1 = f1_score.compute().cpu()

    accelerator.log({"valid_f1": epoch_macro_f1, "valid_acc": epoch_acc}, step=epoch)
    if tensorboard_log:
        tools['valid_writer'].add_scalar('accuracy', epoch_acc, epoch)
        tools['valid_writer'].add_scalar('f1', epoch_macro_f1, epoch)

    # Reset metrics at the end of epoch
    accuracy.reset()
    macro_f1_score.reset()
    f1_score.reset()

    return valid_loss, epoch_acc, epoch_macro_f1, epoch_f1


def test(epoch, accelerator, dataloader, tools, tensorboard_log):
    tools['net'].eval()

    # Initialize metrics
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=tools['num_classes'])
    macro_f1_score = torchmetrics.F1Score(num_classes=tools['num_classes'], average='macro', task="multiclass")
    f1_score = torchmetrics.F1Score(num_classes=tools['num_classes'], average=None, task="multiclass")
    confusion_matrix=torchmetrics.classification.MulticlassConfusionMatrix(num_classes=tools['num_classes'])

    accuracy.to(accelerator.device)
    macro_f1_score.to(accelerator.device)
    f1_score.to(accelerator.device)
    confusion_matrix.to(accelerator.device)

    counter = 0

    progress_bar = tqdm(range(len(dataloader)),
                        disable=not accelerator.is_local_main_process, leave=False)
    progress_bar.set_description("Steps")

    test_loss = 0
    all_predictions=[]
    all_true_values=[]
    for i, data in enumerate(dataloader):
        sequence, labels, sample_weight = data

        with torch.inference_mode():
            outputs = tools['net'](sequence)

            losses = tools['loss_function'](accelerator.gather(outputs), accelerator.gather(labels))

            loss = torch.mean(losses)

            preds = torch.argmax(outputs, dim=1)
            all_predictions.extend(preds.cpu().numpy())
            all_true_values.extend(labels.cpu().numpy())

            accuracy.update(accelerator.gather(preds).detach(), accelerator.gather(labels).detach())
            macro_f1_score.update(accelerator.gather(preds).detach(), accelerator.gather(labels).detach())
            f1_score.update(accelerator.gather(preds).detach(), accelerator.gather(labels).detach())

        counter += 1
        test_loss += loss.data.item()

        progress_bar.update(1)
        logs = {"step_loss": loss.detach().item(),
                "lr": tools['optimizer'].param_groups[0]['lr']}

        progress_bar.set_postfix(**logs)

    test_loss = test_loss / counter

    epoch_acc = accuracy.compute().cpu().item()
    epoch_macro_f1 = macro_f1_score.compute().cpu().item()
    epoch_f1 = f1_score.compute().cpu()

    accelerator.log({"test_f1": epoch_macro_f1, "test_acc": epoch_acc}, step=epoch)
    if tensorboard_log:
        tools['test_writer'].add_scalar('accuracy', epoch_acc, epoch)
        tools['test_writer'].add_scalar('f1', epoch_macro_f1, epoch)
        tools['test_writer'].add_tensor('confusion_matrix', epoch_confusion_matrix, epoch)

    # Reset metrics at the end of epoch
    accuracy.reset()
    macro_f1_score.reset()
    f1_score.reset()
    confusion_matrix.reset()

    #output prediction and target at the end of epoch
    df = pd.DataFrame({
    'True Values': all_true_values,
    'Predictions': all_predictions
    })

    return test_loss, epoch_acc, epoch_macro_f1, epoch_f1, df



def main(args, dict_config, config_file_path):
    configs = load_configs(dict_config, args)

    if type(configs.fix_seed) == int:
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    torch.cuda.empty_cache()
    test_gpu_cuda()

    result_path, checkpoint_path = prepare_saving_dir(configs, config_file_path)

    logging = get_logging(result_path)

    accelerator = Accelerator(
        mixed_precision=configs.train_settings.mixed_precision,
        # split_batches=True,
        gradient_accumulation_steps=configs.train_settings.grad_accumulation,
        dispatch_batches=False
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("accelerator_tracker", config=None)

    dataloaders_dict = prepare_dataloaders_nonfouling(configs)
    logging.info('preparing dataloaders are done')

    net = prepare_models(configs, logging)
    logging.info('preparing model is done')

    optimizer, scheduler = prepare_optimizer(net, configs, len(dataloaders_dict["train"]), logging)
    logging.info('preparing optimizer is done')

    net, start_epoch = load_checkpoints(configs, optimizer, scheduler, logging, net)

    dataloaders_dict["train"], dataloaders_dict["valid"] = accelerator.prepare(
        dataloaders_dict["train"],
        dataloaders_dict["valid"],
    )

    dataloaders_dict["test"] = accelerator.prepare(
        dataloaders_dict["test"],

    )

    net, optimizer, scheduler = accelerator.prepare(
        net, optimizer, scheduler
    )

    # initialize tensorboards
    train_writer, valid_writer, test_writer = prepare_tensorboard(result_path)

    # prepare loss function
    if configs.train_settings.loss == 'crossentropy':
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
    elif configs.train_settings.loss == 'focal':
        alpha_value = torch.full((configs.encoder.num_classes,), 0.8)
        criterion = FocalLoss(
            alpha=alpha_value,
            gamma=2.0,
            reduction='none')
        criterion = accelerator.prepare(criterion)
    else:
        print('wrong optimizer!')
        exit()

    tools = {
        'net': net,
        'train_device': configs.train_settings.device,
        'valid_device': configs.valid_settings.device,
        'train_batch_size': configs.train_settings.batch_size,
        'valid_batch_size': configs.valid_settings.batch_size,
        'test_batch_size': configs.test_settings.batch_size,
        'optimizer': optimizer,
        'mixed_precision': configs.train_settings.mixed_precision,
        'train_writer': train_writer,
        'valid_writer': valid_writer,
        'test_writer': test_writer,
        'accum_iter': configs.train_settings.grad_accumulation,
        'loss_function': criterion,
        'grad_clip': configs.optimizer.grad_clip_norm,
        'checkpoints_every': configs.checkpoints_every,
        'scheduler': scheduler,
        'result_path': result_path,
        'checkpoint_path': checkpoint_path,
        'logging': logging,
        'num_classes': configs.encoder.num_classes
    }

    logging.info(f'number of train steps per epoch: {np.ceil(len(dataloaders_dict["train"]) / tools["accum_iter"])}')
    logging.info(f'number of valid steps per epoch: {len(dataloaders_dict["valid"])}')
    logging.info(f'number of test steps per epoch: {len(dataloaders_dict["test"])}')

    best_valid_acc = 0
    best_valid_f1 = {}
    best_valid_macro_f1 = 0
    global_step = 0
    for epoch in range(start_epoch, configs.train_settings.num_epochs + 1):
        tools['epoch'] = epoch
        start_time = time()
        train_loss, train_acc, train_f1 = train(epoch, accelerator, dataloaders_dict["train"],
                                                tools, global_step,
                                                configs.tensorboard_log, configs)
        end_time = time()
        if accelerator.is_main_process:
            logging.info(f'epoch {epoch} - time {np.round(end_time - start_time, 2)}s, '
                         f'train loss {np.round(train_loss, 4)}, train acc {np.round(train_acc, 4)}, '
                         f'train f1 {np.round(train_f1, 4)}')

        if epoch % configs.valid_settings.do_every == 0 and epoch != 0:
            start_time = time()
            valid_loss, valid_acc, valid_macro_f1, valid_f1 = valid(epoch, accelerator,
                                                                    dataloaders_dict["valid"],
                                                                    tools,
                                                                    tensorboard_log=False)
            end_time = time()
            if accelerator.is_main_process:
                logging.info(
                    f'evaluation - time {np.round(end_time - start_time, 2)}s, '
                    f'valid loss {np.round(valid_loss, 6)}, valid acc {np.round(valid_acc, 6)}, '
                    f'valid f1 {np.round(valid_macro_f1, 6)}'
                )

            if valid_macro_f1 > best_valid_macro_f1:
                # if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_valid_macro_f1 = valid_macro_f1
                best_valid_f1 = valid_f1
                # Set the path to save the model checkpoint.
                model_path = os.path.join(tools['result_path'], 'checkpoints', f'best_model.pth')
                accelerator.wait_for_everyone()
                save_checkpoint(epoch, model_path, tools, accelerator)

        if epoch % configs.checkpoints_every == 0:
            # Set the path to save the model checkpoint.
            model_path = os.path.join(tools['result_path'], 'checkpoints', f'checkpoint_{epoch}.pth')
            accelerator.wait_for_everyone()
            save_checkpoint(epoch, model_path, tools, accelerator)

    if accelerator.is_main_process:
        logging.info(f'best valid acc: {np.round(best_valid_acc, 6)}')
        logging.info(f'best valid macro f1: {np.round(best_valid_macro_f1, 6)}')
        # best_valid_f1 = [np.round(v, 4) for v in best_valid_f1.tolist()]
        # logging.info(f'best valid f1: {best_valid_f1}')

    train_writer.close()
    valid_writer.close()

    # pause 20 second to make sure the best validation checkpoint is ready on the disk
    sleep(20)

    model_path = os.path.join(tools['result_path'], 'checkpoints', 'best_model.pth')
    model_checkpoint = torch.load(model_path, map_location='cpu')
    net.load_state_dict(model_checkpoint['model_state_dict'])

    start_time = time()
    test_loss, test_acc, test_macro_f1, test_f1, pred_true = test(0, accelerator, dataloaders_dict["test"],
                                                        tools, tensorboard_log=False)

    # added by QShao June-10-2024
    csv_path=os.path.join(tools['result_path'], f'pred_true.csv')
    pred_true.to_csv(csv_path, index=False)
    print("Predictions and true values have been saved to pred_true.csv")

    end_time = time()

    if accelerator.is_main_process:
        logging.info(
            f'\ntest - time {np.round(end_time - start_time, 2)}s, '
            f'test loss {np.round(test_loss, 4)}')
        logging.info(f'test acc: {np.round(test_acc, 4)}')
        logging.info(f'test macro f1: {np.round(test_macro_f1, 4)}')
        # test_f1 = [np.round(v, 4) for v in test_f1.tolist()]
        # logging.info(f'test f1: {test_f1}')


    accelerator.end_training()
    accelerator.free_memory()
    del tools, net, dataloaders_dict, accelerator, optimizer, scheduler
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a classification model using esm")
    parser.add_argument("--config_path", "-c", help="The location of config file", default='./config_fold.yaml')
    parser.add_argument("--result_path", default=None,
                        help="result_path, if setted by command line, "
                             "overwrite the one in config.yaml, by default is None")
    parser.add_argument("--resume_path", default=None,
                        help="if set, overwrite the one in config.yaml, by default is None")
    parser.add_argument("--num_end_adapter_layers", default=None, help="num_end_adapter_layers")
    parser.add_argument("--module_type", default=None, help="module_type for adapterh")
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(args, config_file, config_path)
    print('done!')
