The configuration arguments are used to instantiate class `TopKCheckpointManager`.
```yaml
checkpoint:
  topk:
    monitor_key: test_mean_score
    mode: max
    k: 5
    format_str: 'epoch={epoch:04d}-test_mean_score={test_mean_score:.3f}.ckpt'
  save_last_ckpt: True
  save_last_snapshot: False
```
The information above illustrates that here uses `test_mean_score` as a standard to decide whether a model have a better performance.

Obviously, Once `test_mean_score` is higher, the model can achieve a better performance, corresponding to `mode: max`.

`k=5` means we just save 5 sets of parameters which ranks in the top 5, and you will only get 5 sets in the end no matter how many epoch you've trained. Below is the implementation of this :
```python
def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
    if self.k == 0:
        return None

    value = data[self.monitor_key]
    ckpt_path = os.path.join(
        self.save_dir, self.format_str.format(**data))
    
    if len(self.path_value_map) < self.k:
        # under-capacity
        self.path_value_map[ckpt_path] = value
        return ckpt_path
    
    # at capacity
    sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
    min_path, min_value = sorted_map[0]
    max_path, max_value = sorted_map[-1]

    delete_path = None
    if self.mode == 'max':
        if value > min_value:
            delete_path = min_path
    else:
        if value < max_value:
            delete_path = max_path

    if delete_path is None:
        return None
    else:
        del self.path_value_map[delete_path]
        self.path_value_map[ckpt_path] = value

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        if os.path.exists(delete_path):
            os.remove(delete_path)
        return ckpt_path
```