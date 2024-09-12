from serverless import StaticCredentials
from serverless import ServerlessClient
from serverless import RayJobTask
from serverless import JobStatus
from serverless.exceptions import QuerySdkError

import time
import sys

# Access key && Secret key for VolcEngine
ak = "xxx"
sk = "xxx"

def init():
    credentials = StaticCredentials(ak, sk)

    return ServerlessClient(credentials,
                            endpoint='open.volcengineapi.com',
                            service='emr_serverless',
                            region='cn-beijing',
                            connection_timeout=60,
                            socket_timeout=60)


def submit_clustering(delta, epsilon, iter_T):
    client = init()
    job = client.execute(task=RayJobTask(name="datasculpt-clustering-ray",
                                         conf={
                                             "serverless.ray.version": "2.33.0",
                                             "serverless.customized.image.enabled": "true",
                                             "serverless.ray.image": "your_docker_image", # your docker image
                                             "serverless.ray.enable.autoscaling": "true",
                                             "serverless.ray.worker.min.replicas": "1",
                                             "serverless.ray.worker.max.replicas": "120",
                                         },
                                         head_cpu='8',
                                         head_memory='64Gi',
                                         worker_cpu='32',
                                         worker_memory='256Gi',
                                         worker_replicas=1,
                                         entrypoint_cmd=f'python /home/ray/workdir/semantic_clustering/isodata_varient_volcano.py {delta} {epsilon} {iter_T}',
                                         entrypoint_resource='tos://operator-data/keerlu/DataSculpt/semantic_clustering.zip',
                                         queue='data_serverless',
                                        #  runtime_env={
                                        #      "pip": ["ffmpeg<5"],
                                        #     #  "env_vars": {
                                        #     #      "PROTONFS_LOGGING_LEVEL": "DEBUG"
                                        #     #  }
                                        #  },
                                         ),
                         is_sync=False)

    # time.sleep(20)
    # print('RayJob UI: %s' % job.get_tracking_url())

    while not JobStatus.is_finished(job.status):
        job = client.get_job(job.id)
        print('Id: %s, Status: %s' % (job.id, job.status))
        try:
            print('Tracking ui: %s' % job.get_tracking_url())
        except QuerySdkError:
            pass
        time.sleep(3)

    print('The task executed successfully!!!')
    print('Tracking ui: %s' % job.get_tracking_url())
    # print('Result ui: %s' % job.get_result_url())

    log_cursor = client.get_submission_log(job)
    while log_cursor.has_next():
        log_cursor.fetch_next_page()
        current_rows = log_cursor.current_rows
        for log_entry in current_rows:
            print(log_entry)


def submit_MOCO(context_window):
    client = init()
    job = client.execute(task=RayJobTask(name="datasculpt-MOCO-ray",
                                         conf={
                                             "serverless.ray.version": "2.33.0",
                                             "serverless.customized.image.enabled": "true",
                                             "serverless.ray.image": "your_docker_image", # your docker image
                                             "serverless.ray.enable.autoscaling": "true",
                                             "serverless.ray.worker.min.replicas": "1",
                                             "serverless.ray.worker.max.replicas": "120",
                                         },
                                         head_cpu='8',
                                         head_memory='64Gi',
                                         worker_cpu='32',
                                         worker_memory='256Gi',
                                         worker_replicas=1,
                                         entrypoint_cmd=f'python /home/ray/workdir/MOCO_greedy/construct_datasculpt.py {context_window}',
                                         entrypoint_resource='tos://operator-data/keerlu/DataSculpt/MOCO_greedy.zip',
                                         queue='data_serverless',
                                        #  runtime_env={
                                        #      "pip": ["ffmpeg<5"],
                                        #     #  "env_vars": {
                                        #     #      "PROTONFS_LOGGING_LEVEL": "DEBUG"
                                        #     #  }
                                        #  },
                                         ),
                         is_sync=False)

    # time.sleep(20)
    # print('RayJob UI: %s' % job.get_tracking_url())

    while not JobStatus.is_finished(job.status):
        job = client.get_job(job.id)
        print('Id: %s, Status: %s' % (job.id, job.status))
        try:
            print('Tracking ui: %s' % job.get_tracking_url())
        except QuerySdkError:
            pass
        time.sleep(3)

    print('The task executed successfully!!!')
    print('Tracking ui: %s' % job.get_tracking_url())
    # print('Result ui: %s' % job.get_result_url())

    log_cursor = client.get_submission_log(job)
    while log_cursor.has_next():
        log_cursor.fetch_next_page()
        current_rows = log_cursor.current_rows
        for log_entry in current_rows:
            print(log_entry)


def get_ray_ui():
    client = init()
    job = client.get_job(280218916)
    print('start_time: %s' % job.__getattribute__("start_time"))
    print('end_time: %s' % job.__getattribute__("end_time"))
    print('status: %s' % job.__getattribute__("status"))
    print('queue_name: %s' % job.__getattribute__("queue_name"))
    print('conf: %s' % job.__getattribute__("conf"))
    print('tracking_url: %s' % job.get_tracking_url())


def cancel_job(job_id: str):
    client = init()
    client.cancel_job(client.get_job(job_id))


def query_log(job_id: str):
    client = init()
    job = client.get_job(job_id)
    log_cursor = client.get_driver_log(job)
    while log_cursor.has_next():
        log_cursor.fetch_next_page()
        current_rows = log_cursor.current_rows
        for log_entry in current_rows:
            print(log_entry)


if __name__ == '__main__':
    context_window, delta, epsilon, iter_T = sys.argv[1:]
    # query_log("283397020")
    submit_clustering(delta, epsilon, iter_T) # step 1
    submit_MOCO(context_window) # step 2
    # cancel_job("284074059")
