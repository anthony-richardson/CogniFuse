from abc import ABC, abstractmethod
import torch

''' Scenario definitions
    'Relax_before_LCT'          # pause before LCT task (controlled, no LCT)
    'Relax_during_LCT',         # pause in between LCT task (controlled, no LCT)
    'Relax_after_LCT',          # pause after LCT task (controlled, no LCT)
    'SwitchingTask_1',          # Switching paradigm at difficulty 1 (controlled, no LCT)
    'SwitchingTask_2',          # Switching paradigm at difficulty 2 (controlled, no LCT)
    'SwitchingTask_3',          # Switching paradigm at difficulty 3 (controlled, no LCT)
    'LCT_Baseline',             # LCT without additional task (uncontrolled, LCT)
    'SwitchBackAuditive_1',     # LCT with added auditive task at difficulty 1 (uncontrolled, LCT + auditive)
    'SwitchBackAuditive_2',     # LCT with added auditive task at difficulty 2 (uncontrolled, LCT + auditive)
    'SwitchBackAuditive_3',     # LCT with added auditive task at difficulty 3 (uncontrolled, LCT + auditive)
    'VisualSearchTask_1',       # LCT with added visual task at difficulty 1 (uncontrolled, LCT + visual)
    'VisualSearchTask_2',       # LCT with added visual task at difficulty 2 (uncontrolled, LCT + visual)
    'VisualSearchTask_3'        # LCT with added visual task at difficulty 3 (uncontrolled, LCT + visual)
'''


'''def get_task_tools(name):
    if name == 'ControlledSwitchingLowHigh':
        return ControlledSwitchingLowHigh()
    else:
        raise ValueError(f'Unsupported task name [{name}]')'''


class Task(ABC):
    # Returns a dictionary that maps
    # from task name to class number.
    @staticmethod
    @abstractmethod
    def get_mapper():
        pass

    # Creates the class labels based on the task
    # and difficulty.
    def map_meta_info_to_class(self, meta_info):
        mapper = self.get_mapper()

        class_labels = []
        for n in meta_info['scenario']:
            cls = mapper[n]
            class_labels.append(cls)

        class_labels = torch.as_tensor(class_labels)

        return class_labels


'''class ControlledSwitchingLowHigh(Task):
    @staticmethod
    def get_mapper():
        mapper = {
            'Relax_before_LCT': 0,
            'Relax_during_LCT': 0,
            'Relax_after_LCT': 0,
            'SwitchingTask_1': 1,
            'SwitchingTask_2': 1,
            'SwitchingTask_3': 1
        }
        return mapper'''


# c0 from https://dl.acm.org/doi/pdf/10.1145/2070481.2070516
class SwitchingTaskPresence(Task):
    @staticmethod
    def get_mapper():
        mapper = {
            'Relax_before_LCT': 0,
            'Relax_during_LCT': 0,
            'Relax_after_LCT': 0,
            'SwitchingTask_1': 1,
            'SwitchingTask_2': 1,
            'SwitchingTask_3': 1
        }
        return mapper


class SwitchingTaskDifficulty3(Task):
    @staticmethod
    def get_mapper():
        mapper = {
            'SwitchingTask_1': 0,
            'SwitchingTask_2': 1,
            'SwitchingTask_3': 2
        }
        return mapper


class SwitchingTaskDifficulty2(Task):
    @staticmethod
    def get_mapper():
        mapper = {
            'SwitchingTask_1': 0,
            'SwitchingTask_3': 1
        }
        return mapper


# u0 from https://dl.acm.org/doi/pdf/10.1145/2070481.2070516
class SwitchBackAuditivePresenceRelax(Task):
    @staticmethod
    def get_mapper():
        mapper = {
            'Relax_before_LCT': 0,
            'Relax_during_LCT': 0,
            'Relax_after_LCT': 0,
            'SwitchBackAuditive_1': 1,
            'SwitchBackAuditive_2': 1,
            'SwitchBackAuditive_3': 1
        }
        return mapper


# u1 from https://dl.acm.org/doi/pdf/10.1145/2070481.2070516
class SwitchBackAuditivePresence(Task):
    @staticmethod
    def get_mapper():
        mapper = {
            'LCT_Baseline': 0,
            'SwitchBackAuditive_1': 1,
            'SwitchBackAuditive_2': 1,
            'SwitchBackAuditive_3': 1
        }
        return mapper


class SwitchBackAuditiveDifficulty3(Task):
    @staticmethod
    def get_mapper():
        mapper = {
            'SwitchBackAuditive_1': 0,
            'SwitchBackAuditive_2': 1,
            'SwitchBackAuditive_3': 2
        }
        return mapper


class SwitchBackAuditiveDifficulty2(Task):
    @staticmethod
    def get_mapper():
        mapper = {
            'SwitchBackAuditive_1': 0,
            'SwitchBackAuditive_3': 1
        }
        return mapper


# u2 from https://dl.acm.org/doi/pdf/10.1145/2070481.2070516
class VisualSearchTaskPresence(Task):
    @staticmethod
    def get_mapper():
        mapper = {
            'LCT_Baseline': 0,
            'VisualSearchTask_1': 1,
            'VisualSearchTask_2': 1,
            'VisualSearchTask_3': 1
        }
        return mapper


class VisualSearchTaskDifficulty3(Task):
    @staticmethod
    def get_mapper():
        mapper = {
            'VisualSearchTask_1': 0,
            'VisualSearchTask_2': 1,
            'VisualSearchTask_3': 2
        }
        return mapper


class VisualSearchTaskDifficulty2(Task):
    @staticmethod
    def get_mapper():
        mapper = {
            'VisualSearchTask_1': 0,
            'VisualSearchTask_3': 1
        }
        return mapper


class TaskDiscrimination(Task):
    @staticmethod
    def get_mapper():
        mapper = {
            'SwitchingTask_1': 0,
            'SwitchingTask_2': 0,
            'SwitchingTask_3': 0,
            'SwitchBackAuditive_1': 1,
            'SwitchBackAuditive_2': 1,
            'SwitchBackAuditive_3': 1,
            'VisualSearchTask_1': 2,
            'VisualSearchTask_2': 2,
            'VisualSearchTask_3': 2
        }
        return mapper


# TODO
class UserDiscrimination(Task):
    @staticmethod
    def get_mapper():
        mapper = {
            '': 0,
        }
        return mapper
