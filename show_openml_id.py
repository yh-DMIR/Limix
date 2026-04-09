import openml

task = openml.tasks.get_task(999)
dataset = task.get_dataset()

print(dataset.dataset_id)
print(dataset.name)
