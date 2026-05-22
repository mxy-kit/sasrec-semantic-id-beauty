import copy
import os

import torch

from modeling.trainer import MetricCallback, InferenceCallback
from modeling.utils import create_logger, TensorboardWriter, DEVICE

LOGGER = create_logger(name=__name__)


class Trainer:
    def __init__(
            self,
            experiment_name,
            train_dataloader,
            validation_dataloader,
            eval_dataloader,
            model,
            optimizer,
            loss_function,
            ranking_metrics,
            epoch_cnt=None,
            step_cnt=None,
            best_metric=None,
            epochs_threshold=40,
            valid_step=256,
            eval_step=256,
            checkpoint_dir='../checkpoints',
            eval_during_train=False
    ):
        self._experiment_name = experiment_name
        self._train_dataloader = train_dataloader
        self._validation_dataloader = validation_dataloader
        self._eval_dataloader = eval_dataloader
        self._model = model
        self._optimizer = optimizer
        self._loss_function = loss_function
        self._epoch_cnt = epoch_cnt
        self._step_cnt = step_cnt
        self._best_metric = best_metric
        self._epochs_threshold = epochs_threshold
        self._ranking_metrics = ranking_metrics
        self._checkpoint_dir = checkpoint_dir
        self._valid_step = valid_step
        self._eval_step = eval_step
        self._eval_during_train = eval_during_train

        os.makedirs(self._checkpoint_dir, exist_ok=True)

        tensorboard_writer = TensorboardWriter(self._experiment_name)

        self._metric_callback = MetricCallback(
            tensorboard_writer=tensorboard_writer,
            on_step=1
        )

        self._validation_callback = InferenceCallback(
            tensorboard_writer=tensorboard_writer,
            step_name='validation',
            model=model,
            dataloader=validation_dataloader,
            on_step=valid_step,
            metrics=ranking_metrics,
            pred_prefix='predictions',
            labels_prefix='labels'
        )

        self._eval_callback = InferenceCallback(
            tensorboard_writer=tensorboard_writer,
            step_name='eval',
            model=model,
            dataloader=eval_dataloader,
            on_step=eval_step,
            metrics=ranking_metrics,
            pred_prefix='predictions',
            labels_prefix='labels'
        )

    def _save_state_dict(self, state_dict, name):
        path = os.path.join(self._checkpoint_dir, name)
        torch.save(state_dict, path)
        LOGGER.debug(f'Saved checkpoint: {path}')
        return path

    def train(self):
        step_num = 0
        epoch_num = 0
        current_metric = float('-inf')
        best_epoch = 0
        best_checkpoint = None

        max_steps = self._step_cnt if self._step_cnt is not None else 200_000
        max_epochs = self._epoch_cnt if self._epoch_cnt is not None else 10**9

        LOGGER.debug('Start training...')
        LOGGER.debug(f'Max steps: {max_steps}, max epochs: {max_epochs}')

        while step_num < max_steps and epoch_num < max_epochs:
            if best_checkpoint is not None and best_epoch + self._epochs_threshold < epoch_num:
                LOGGER.debug(
                    'There is no progress during {} epochs. Finish training'.format(self._epochs_threshold)
                )
                break

            LOGGER.debug(f'Start epoch {epoch_num}')

            for batch in self._train_dataloader:
                if step_num >= max_steps:
                    break

                self._model.train()

                for key, values in batch.items():
                    batch[key] = values.to(DEVICE)

                batch.update(self._model(batch))
                loss = self._loss_function(batch)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                self._metric_callback(
                    key='loss',
                    value=loss.item(),
                    step_num=step_num,
                    prefix='train'
                )

                validation_metrics = {}
                if self._valid_step is not None and self._valid_step > 0:
                    if step_num > 0 and step_num % self._valid_step == 0:
                        LOGGER.debug(f'Running validation on step {step_num}...')
                        validation_metrics = self._validation_callback(step_num)

                        for key, value in validation_metrics.items():
                            self._metric_callback(
                                key=key,
                                value=value,
                                step_num=step_num,
                                prefix='validation'
                            )
                            print(f'validation/{key}: {value}')

                        if self._best_metric is not None and self._best_metric in validation_metrics:
                            metric_value = validation_metrics[self._best_metric]

                            if best_checkpoint is None or metric_value >= current_metric:
                                LOGGER.debug(
                                    f'New best checkpoint on {self._best_metric}: '
                                    f'{metric_value} > {current_metric}'
                                )
                                print(
                                    f'BEST step={step_num} epoch={epoch_num} '
                                    f'{self._best_metric}={metric_value} prev={current_metric}'
                                )

                                current_metric = metric_value
                                best_epoch = epoch_num
                                best_checkpoint = copy.deepcopy(self._model.state_dict())

                                self._save_state_dict(
                                    best_checkpoint,
                                    f'{self._experiment_name}_best_step_{step_num}.pth'
                                )

                if self._eval_during_train:
                    if self._eval_step is not None and self._eval_step > 0:
                        if step_num > 0 and step_num % self._eval_step == 0:
                            LOGGER.debug(f'Running eval on step {step_num}...')
                            evaluation_metrics = self._eval_callback(step_num)
                            for key, value in evaluation_metrics.items():
                                self._metric_callback(
                                    key=key,
                                    value=value,
                                    step_num=step_num,
                                    prefix='eval'
                                )
                                print(f'eval/{key}: {value}')

                step_num += 1

            epoch_num += 1

        LOGGER.debug('Training procedure has been finished!')

        final_path = self._save_state_dict(
            self._model.state_dict(),
            f'{self._experiment_name}_last_state.pth'
        )
        print(f'LAST_CHECKPOINT: {final_path}')

        if best_checkpoint is None:
            LOGGER.debug('No validation checkpoint was selected. Returning last checkpoint.')
            best_checkpoint = copy.deepcopy(self._model.state_dict())
            self._save_state_dict(
                best_checkpoint,
                f'{self._experiment_name}_best_fallback_last.pth'
            )

        return best_checkpoint

    def eval(self):
        evaluation_metrics = self._eval_callback(0)
        for key, value in evaluation_metrics.items():
            print(key, value)

    def save(self):
        LOGGER.debug('Saving model...')
        checkpoint_path = f'{self._checkpoint_dir}/{self._experiment_name}_final_state.pth'
        torch.save(self._model.state_dict(), checkpoint_path)
        LOGGER.debug('Saved model as {}'.format(checkpoint_path))

    def load(self, checkpoint):
        self._model.load_state_dict(checkpoint)
