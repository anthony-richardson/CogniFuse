import os
import json

from utils.model_util import count_parameters, create_multimodal_deformer, create_unimodal_deformer
from load.load_data import get_data_loader
import utils.tasks as tasks
from train.training_loop import TrainingLoop
from utils.train_platforms import TensorboardPlatform
from utils.fixseed import fixseed
from utils.parser_util import multimodal_deformer_train_args, unimodal_deformer_train_args, is_multimodal


def main():
    multimodal = is_multimodal()
    if multimodal:
        args = multimodal_deformer_train_args()
        deformer_model = create_multimodal_deformer(args)
    else:
        args = unimodal_deformer_train_args()
        deformer_model = create_unimodal_deformer(args)

    fixseed(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        args_path = os.path.join(args.save_dir, 'args.json')
        with open(args_path, 'w') as fw:
            json.dump(vars(args), fw, indent=4)

    save_dir = os.path.join(args.save_dir, args.fold)
    data_dir = os.path.join(args.data_dir, args.fold)

    num_parameters = count_parameters(deformer_model)
    print(f'Number of parameters: {num_parameters}')

    task_tools = getattr(tasks, args.task)

    train_data = get_data_loader(
        batch_size=args.batch_size,
        tasks=task_tools.get_mapper().keys(),
        data_dir=data_dir,
        split='train'
    )

    validation_data = get_data_loader(
        batch_size=args.batch_size,
        tasks=task_tools.get_mapper().keys(),
        data_dir=data_dir,
        split='validation'
    )

    train_platform = TensorboardPlatform(save_dir)
    train_platform.report_args(args, name='Args')

    training_loop = TrainingLoop(
        args=args,
        train_platform=train_platform,
        model=deformer_model,
        task_tools=task_tools,
        train_data=train_data,
        validation_data=validation_data,
        save_dir=save_dir#,
        #modality_name=args.modality
    )
    training_loop.run_loop()

    train_platform.close()


if __name__ == '__main__':
    main()
