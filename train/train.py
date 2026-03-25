import os

from utils.model_util import count_parameters, create_model
from load.load_data import get_data_loader
import utils.tasks as tasks
from train.training_loop import TrainingLoop
from utils.train_platforms import TensorboardPlatform
from utils.fixseed import fixseed
from utils.parser_util import train_args


def main():
    args = train_args()
    deformer_model = create_model(args)

    fixseed(args.seed)

    base_dir = args.save_dir
    save_dir = os.path.join(base_dir, args.fold)
    data_dir = os.path.join(args.data_dir, args.fold)

    num_parameters = count_parameters(deformer_model)
    print(f'Number of parameters: {num_parameters}')

    task_tools = getattr(tasks, args.task)

    train_data = get_data_loader(
        batch_size=args.batch_size,
        tasks=task_tools.get_mapper().keys(),
        data_dir=data_dir,
        split='train',
        drop_last=bool(args.drop_last)
    )

    validation_data = get_data_loader(
        batch_size=args.batch_size,
        tasks=task_tools.get_mapper().keys(),
        data_dir=data_dir,
        split='validation',
        drop_last=bool(args.drop_last)
    )

    train_platform = TensorboardPlatform(log_dir=save_dir, args_dir=base_dir)
    train_platform.report_args(args=args, name='args')

    training_loop = TrainingLoop(
        args=args,
        train_platform=train_platform,
        model=deformer_model,
        task_tools=task_tools,
        train_data=train_data,
        validation_data=validation_data,
        save_dir=save_dir
    )
    training_loop.run_loop()

    train_platform.close()


if __name__ == '__main__':
    main()
