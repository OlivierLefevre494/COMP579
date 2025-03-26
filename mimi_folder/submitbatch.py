import azure.batch as batch
import azure.batch.batch_auth as batch_auth
import azure.batch.models as batch_models
from azure.storage.blob import BlobServiceClient


credentials = batch_auth.SharedKeyCredentials(BATCH_ACCOUNT_NAME, BATCH_ACCOUNT_KEY)
batch_client = batch.BatchServiceClient(credentials, batch_url=BATCH_ACCOUNT_URL)
epsilons = [0.25, 0.1, 0.01]
lrs = [0.01, 0.001, 0.0001]
replay_buffers = [(128, 1000000), (1, 1)]
# Define hyperparameters (18 hyperparams) and seeds (10 seeds)
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
hyperparams = [(epsilon, lr, buffer_size, buffer_length)
               for epsilon in epsilons
               for lr in lrs
               for buffer_size, buffer_length in replay_buffers]

# Create Batch Job
job = batch_models.JobAddParameter(
    id=JOB_ID,
    pool_info=batch_models.PoolInformation(pool_id=POOL_ID),
)
batch_client.job.add(job)

# Submit 180 Tasks
i = 0
for seed in seeds:
    for hparam in hyperparams:
        command = f"python assault1.py --seed {seed} --lr {hparam[1]} --batch_size {hparam[2]} --replay_buffer_size {hparam[3]} --epsilon {hparam[0]} --output results/seed{seed}_lr{hparam[1]}_buffer{hparam[2]}_epsilon{hparam[0]}_batch{hparam[3]}.pkl"
        task_id = f"task-{seed}-{hparam[1]}-{hparam[2]}-{hparam[0]}-{hparam[3]}"
        task = batch_models.TaskAddParameter(
            id=str(i),
            command_line=command,
            resource_files=[],  # Upload assault1.py separately
        )
        batch_client.task.add(job_id=JOB_ID, task=task)
        i += 1
