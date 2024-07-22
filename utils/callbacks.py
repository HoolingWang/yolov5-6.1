# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Callback utils."""

import threading


class Callbacks:
    """Handles all registered callbacks for YOLOv5 Hooks."""

    def __init__(self):
        """Initializes a Callbacks object to manage registered YOLOv5 training event hooks."""
        # self._callbacks 是一个字典，存储了所有可以注册回调的钩子事件。
        self._callbacks = {
            "on_pretrain_routine_start": [],
            "on_pretrain_routine_end": [],
            "on_train_start": [],
            "on_train_epoch_start": [],
            "on_train_batch_start": [],
            "optimizer_step": [],
            "on_before_zero_grad": [],
            "on_train_batch_end": [],
            "on_train_epoch_end": [],
            "on_val_start": [],
            "on_val_batch_start": [],
            "on_val_image_end": [],
            "on_val_batch_end": [],
            "on_val_end": [],
            "on_fit_epoch_end": [],  # fit = train + val
            "on_model_save": [],
            "on_train_end": [],
            "on_params_update": [],
            "teardown": [],
        }
        # self.stop_training 是一个布尔变量，用于控制是否停止训练。
        self.stop_training = False  # set True to interrupt training

    # 这个方法用于将新的回调函数注册到指定的钩子事件上。
    def register_action(self, hook, name="", callback=None):
        """
        Register a new action to a callback hook.

        Args:
            hook: The callback hook name to register the action to
            name: The name of the action for later reference
            callback: The callback to fire
        """
        '''
        hook 是要注册的钩子事件名称。
        name 是回调函数的名称，便于后续引用。
        callback 是要注册的回调函数，必须是可调用的。
        '''
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        assert callable(callback), f"callback '{callback}' is not callable"
        self._callbacks[hook].append({"name": name, "callback": callback})

    def get_registered_actions(self, hook=None):
        """
        Returns all the registered actions by callback hook.
        这个方法返回指定钩子事件的所有已注册回调函数。

        Args:
            hook: The name of the hook to check, defaults to all
            如果不指定 hook，则返回所有钩子事件的回调函数。
        """
        return self._callbacks[hook] if hook else self._callbacks

    def run(self, hook, *args, thread=False, **kwargs):
        """
        Loop through the registered actions and fire all callbacks on main thread.
        这个方法遍历指定钩子事件的所有已注册回调函数，并依次执行。

        Args:
            hook: The name of the hook to check, defaults to all
            args: Arguments to receive from YOLOv5
            thread: (boolean) Run callbacks in daemon thread
            kwargs: Keyword Arguments to receive from YOLOv5
            hook 是要触发的钩子事件名称。
            args 和 kwargs 是传递给回调函数的参数。
            thread 参数决定是否在新线程中运行回调函数。
        """

        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        for logger in self._callbacks[hook]:
            if thread:
                threading.Thread(target=logger["callback"], args=args, kwargs=kwargs, daemon=True).start()
            else:
                logger["callback"](*args, **kwargs)
