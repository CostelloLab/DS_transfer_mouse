def get_task_data(task_data):
    if isinstance(task_data, (tuple, list)):
        nb_path = task_data[0]
        task_markers = task_data[1]

        if not isinstance(task_markers, (tuple, list)):
            task_markers = [task_markers]
    else:
        nb_path = task_data
        task_markers = []

    return nb_path, task_markers
