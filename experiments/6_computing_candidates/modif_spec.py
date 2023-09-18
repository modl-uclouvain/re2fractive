def get_fws_and_tasks(workflow, fw_name_constraint=None, task_name_constraint=None):
    """
    Helper method: given a workflow, returns back the fw_ids and task_ids that match name
    constraints. Used in developing multiple powerups.
    Args:
	workflow (Workflow): Workflow
        fw_name_constraint (str): a constraint on the FW name
        task_name_constraint (str): a constraint on the task name
    Returns:
       a list of tuples of the form (fw_id, task_id) of the RunVasp-type tasks
    """
    fws_and_tasks = []
    for idx_fw, fw in enumerate(workflow.fws):
        if fw_name_constraint is None or fw_name_constraint in fw.name:
            for idx_t, t in enumerate(fw.tasks):
                if task_name_constraint is None or task_name_constraint in str(t):
                    fws_and_tasks.append((idx_fw, idx_t))
    return fws_and_tasks

def preserve_fworker(original_wf, fw_name_constraint=None):
    """
    set _preserve_fworker spec of Fireworker(s) of a Workflow. Can be used to
    pin a workflow to the first fworker it is run with. Very useful when running
    on multiple machines that can't share files. fw_name_constraint can be used
    to only preserve fworker after a certain point where file passing becomes
    important
    Args:
	original_wf (Workflow):
        fw_name_constraint (str): name of the Fireworks to be tagged (all if
        None is passed)
    Returns:
	Workflow: modified workflow with specified Fireworkers tagged
    """
    idx_list = get_fws_and_tasks(original_wf, fw_name_constraint=fw_name_constraint)
    for idx_fw, idx_t in idx_list:
        original_wf.fws[idx_fw].spec["_preserve_fworker"] = True
    return original_wf

if __name__=='__main__':
    pass
