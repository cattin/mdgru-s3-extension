#!/bin/python3

import os
import sys
import stat
import time
from datetime import datetime
import shutil
import warnings

from omegaconf import OmegaConf
from omegaconf import DictConfig, ListConfig
import docker
from copy import deepcopy
import numpy as np

MDGRU_PARAMETER_DONT_LIST = ["save_validation_results",
                             "correct_orientation",
                             "whiten",
                             "perform_one_hot_encoding",
                             "validate_same",
                             "use_tensorboard",
                             "perform_full_image_validation",
                             "save_validation_results"]

MDGRU_PARAMETER_STORE_TRUE_LIST = ["use_pytorch",
                                   "use_s3",
                                   "nonthreaded",
                                   "dice_autoweighted",
                                   "dice_generalized",
                                   "dice_cc"]




class Job():

    STATUS_RUNNING = "running"
    STATUS_PLANNED = "planned"
    STATUS_FINISHED = "finished"

    COLOR_RUNNING = '\033[93m'
    COLOR_PLANNED = '\033[91m'
    COLOR_FINISHED = '\033[92m'

    def __init__(self, job_name, session_token, cfg, status=STATUS_PLANNED):
        self.job_name = job_name
        self.gpu_id = None
        self.status = status
        self.session_token = session_token
        self.cfg = deepcopy(cfg)
        self.runtime = None
        self.container_name = None

        self.animation_state = 0
        self.animation_elements = ["|", "/", "-", "\\"]

    def is_job(self, container_name):
        container_name_token = container_name.split(".")

        # valid container names have exact three entries token.job_id.gpu_id
        if len(container_name_token) != 3:
            return False
        # only container names with the correct session token are processed
        if container_name_token[0] != self.session_token:
            return False

        # looks like we found a container with the correct name pattern
        return self.job_name == container_name_token[1]

    def get_container_name(self):
        return self.container_name

    def update_status(self, status):
        self.status = status

    def update_gpu_id(self, gpu_id):
        self.gpu_id = gpu_id

    def is_running(self):
        return self.status == Job.STATUS_RUNNING

    def __str__(self):
        if self.status == Job.STATUS_PLANNED:
            color = Job.COLOR_PLANNED
        elif self.status == Job.STATUS_RUNNING:
            color = Job.COLOR_RUNNING
        elif self.status == Job.STATUS_FINISHED:
            color = Job.COLOR_FINISHED
        else:
            print("Error: status corrupt")
            exit(-1)
        if self.status == Job.STATUS_RUNNING or self.status == Job.STATUS_PLANNED:
            runtime = "NA"
        else:
            runtime = self.runtime

        if self.status == Job.STATUS_RUNNING:
            self.animation_state = (self.animation_state + 1) % len(self.animation_elements)
            animation = self.animation_elements[self.animation_state]
        else:
            animation = ""

        return f"{color} Job: {self.job_name} GPU: {str(self.gpu_id)}  Status: {self.status} Runtime: {runtime} " \
               f"{animation}\033[0m"

    def start(self):
        self.status = self.STATUS_RUNNING
        # build the container name
        self.container_name = ".".join([self.session_token, self.job_name, self.gpu_id])

        # correct the location for training, test, and validation
        self.cfg.mdgru.locationtraining = os.path.join(cfg.mdgru.datapath, self.job_name, "train",
                                                       self.cfg.mdgru.locationtraining)
        self.cfg.mdgru.locationvalidation = os.path.join(cfg.mdgru.datapath, self.job_name, "val",
                                                       self.cfg.mdgru.locationvalidation)
        self.cfg.mdgru.locationtesting = os.path.join(cfg.mdgru.datapath, self.job_name, "test",
                                                         self.cfg.mdgru.locationtesting)

        # reset the data path so that the model checkpoints will be stored in the correct fold folder
        self.cfg.mdgru.datapath = os.path.join(self.cfg.mdgru.datapath, self.job_name, "checkpoints")
        if not os.path.isdir(self.cfg.mdgru.datapath):
            os.makedirs(self.cfg.mdgru.datapath)

        # run the new job
        run(self.cfg, self.gpu_id, self.container_name)
        self.runtime = datetime.now()

    def finished(self):
        self.status = Job.STATUS_FINISHED
        self.runtime = datetime.now() - self.runtime


class DockerDispatcher():
    def __init__(self, cfg, config_filename):
        self.cfg = cfg

        # get the docker client
        self.docker_client = docker.from_env()

        self.folds = [folder for folder in os.listdir(cfg.mdgru.datapath)
                       if os.path.isdir(os.path.join(cfg.mdgru.datapath, folder))]

        if len(self.folds) == 0:
            warnings.warn(" ".join(["No fold found in folder: ", cfg.mdgru.datapath]))

        self.session_token = str(np.random.randint(100000, 999999))
        self.gpu_pool = [str(gpu_id) for gpu_id in cfg.run.gpu_pool]
        self.gpu_free = self.gpu_pool
        self.jobs_list = [Job(job_id, self.session_token, self.cfg) for job_id in self.folds]
        self.config_filename = config_filename
        self.update_rate = 1
        self.first_run = True

    def update_GPU_pool(self):
        cfg = OmegaConf.load(self.config_filename)
        new_gpu_pool = [str(gpu_id) for gpu_id in cfg.run.gpu_pool]
        new_gpus = list(set(new_gpu_pool).difference(self.gpu_pool))

        # add the new GPUs to the list of free GPUs
        self.gpu_free = self.gpu_free + new_gpus

        self.gpu_pool = new_gpu_pool

    def all_jobs_done(self):
        for job in self.jobs_list:
            if job.status == Job.STATUS_PLANNED or job.status == Job.STATUS_RUNNING:
                return False
        # if there is no more job planned all jobs have been scheduled
        return True

    def container_is_running(self, container_name):
        docker_container = self.docker_client.containers.list()
        for container in docker_container:
            if container.name == container_name:
                return True

        return False

    def start_job(self, job):
        job.start()
        while not self.container_is_running(job.container_name):
            # wait until the new container has started
            pass
        return

    def print_job_overview(self):
        if not self.first_run:
            go_back_string = "\033[" + str(len(self.jobs_list) + 2) + "A"
            print(go_back_string)
            print("\033[J")
        else:
            self.first_run = False


        for job in self.jobs_list:
            print(job)

    def update_job_scheduling(self):

        if len(self.gpu_free) == 0:
            return

        for job in self.jobs_list:
            if job.status == Job.STATUS_PLANNED:
                if len(self.gpu_free) > 0:
                    if not job.is_running():
                        job.update_gpu_id(self.gpu_free.pop())
                        # start the new job
                        self.start_job(job)
                else:
                    return

    def run(self):
        # run unitl all jobs are done
        while not self.all_jobs_done():
            self.update_GPU_pool()
            # update the job status of each job based on the running docker container
            self.update_job_status()
            # reassign the GPU and start the next jobs
            self.update_job_scheduling()
            self.print_job_overview()
            time.sleep(self.update_rate)

        print("Done !")

    def update_job_status(self):
        docker_container = self.docker_client.containers.list()
        # docker_container = []
        for job in self.jobs_list:
            job_found = False
            for container in docker_container:
                # access the container name needs to be in a try an catch block as it might be the case that the
                # container has finished before the call
                try:
                    container_name = container.name
                    job_found = job.is_job(container_name)
                    # if we found the job it is still running
                    if job_found:
                        break
                except:
                    pass
            # if the job was not found and has the status running it has finished and we can reuse the GPU
            if not job_found and job.is_running():
                job.finished()
                # if the GPU is still in the GPU pool
                if job.gpu_id in self.gpu_pool:
                    self.gpu_free.append(job.gpu_id)


def cfg_2_mdgru(cfg):
    parameter_list = []

    for element in cfg:
        space = " "
        # check if the element is a boolean
        if str(cfg[element]).lower() == "true" or str(cfg[element]).lower() == "false":
            if element in MDGRU_PARAMETER_DONT_LIST:
                if str(cfg[element]).lower() == "false":
                    parameter_list.append("".join(("--", "dont_", element)))
                    continue
                else:
                    #parameter_list.append("".join(("--", element)))
                    continue
            elif element in MDGRU_PARAMETER_STORE_TRUE_LIST:
                if str(cfg[element]).lower() == "false":
                    continue
                else:
                    parameter_list.append("".join(("--", element)))
                    continue
            else:
                 #if str(cfg[element]).lower() == "true":
                 #    parameter_list.append("".join(("--", element)))
                 #else:
                  #   parameter_list.append("".join(("--", "no_", element)))
                 #continue
                #space = " "
                if str(cfg[element]).lower() == "false":
                    continue
                else:
                    cfg[element] = " "

        if len(element) == 1:
            dashes = "-"
        else:
            dashes = "--"

        if isinstance(cfg[element], ListConfig):
            str_list = [str(entry) for entry in cfg[element]]
            values = " ".join(str_list)
        else:
            values = str(cfg[element])

        parameter_list.append("".join((dashes, element, space, values)))
    print(parameter_list)
    return parameter_list

def build_docker_image(cfg) -> None:

    # print("BUILD DOCKER CONTAINER")

    path_docker_file = os.path.abspath("docker")
    command = " ".join(("docker build", path_docker_file, "--quiet", "--tag", cfg.run.container_name))
    # print(command)
    os.system(command)
    print("\033[2A")
    # print("\033[J")

def run_docker(command):
    # print("START DOCKER CONTAINER")
    final_command = " ".join(command)
    # print(final_command)
    os.system(final_command)
    print("\033[2A")
    # print("\033[J")


def run(cfg: DictConfig, gpu_id, container_name) -> None:

    # build the docker image
    build_docker_image(cfg)

    detach = ""
    if cfg.run.detach_docker:
        detach = "-d"


    # compose the final docker run command
    docker_run_command = ["docker run", detach, "--shm-size 8G", "--init", "--rm", "--name", container_name]

    if cfg.mdgru.gpu >= 0:
        device_command = [" --gpus='\"device="]
        device_command.append(str(gpu_id))
        device_command.append("\"'")
        docker_run_command.append("".join(device_command))

        # reindex the GPU IDs as it starts with 0 in the docker
        cfg.mdgru.gpu = 0

    if not cfg.run.docker_rootless:
        docker_run_command.append("--user=$(id -u):$(id -g)")

    docker_run_command.append("--volume=$PWD:/app")

    # mount the output path
    if not os.path.isdir(cfg.mdgru.datapath):
        os.makedirs(cfg.mdgru.datapath)

    data_path_docker = "/app/results/datapath"
    docker_run_command.append("".join(("-v ", cfg.mdgru.datapath, ":", data_path_docker)))

    # update the result path to run in the docker container
    cfg.mdgru.datapath = data_path_docker

    # set the training location of the location in the docker
    training_path_docker = "/app/results/training_path"
    # get the dirname as the path needs to be mounted not the image
    dirname_training = os.path.dirname(cfg.mdgru.locationtraining)
    docker_run_command.append("".join(("-v ", dirname_training, ":", training_path_docker)))
    # update the result path to run in the docker container
    cfg.mdgru.locationtraining = os.path.join(training_path_docker, os.path.basename(cfg.mdgru.locationtraining))

    # set the validation location of the location in the docker
    validation_path_docker = "/app/results/validation_path"
    # get the dirname as the path needs to be mounted not the image
    dirname_validation = os.path.dirname(cfg.mdgru.locationvalidation)
    docker_run_command.append("".join(("-v ", dirname_validation, ":", validation_path_docker)))
    # update the result path to run in the docker container
    cfg.mdgru.locationvalidation = os.path.join(validation_path_docker, os.path.basename(cfg.mdgru.locationvalidation))

    # set the test location of the location in the docker
    test_path_docker = "/app/results/test_path"
    # get the dirname as the path needs to be mounted not the image
    dirname_test = os.path.dirname(cfg.mdgru.locationtesting)
    docker_run_command.append("".join(("-v ", dirname_test, ":", test_path_docker)))
    # update the result path to run in the docker container
    cfg.mdgru.locationtesting = os.path.join(test_path_docker, os.path.basename(cfg.mdgru.locationtesting))

    # create the cache if not found
    if not os.path.isdir(cfg.run.cache_path):
        os.makedirs(cfg.run.cache_path)

    # change the access rights of the cache to the current user only
    os.chmod(cfg.run.cache_path, stat.S_IRWXU | stat.S_IRWXG)

    # set the  location of the cache in the docker
    cache_path_docker = "/app/cache"
    docker_run_command.append("".join(("-v ", cfg.run.cache_path, ":", cache_path_docker)))

    # set the docker tag
    docker_run_command.append(cfg.run.container_name + ":latest")

    # transforms the arguments from OmegaConf to a list of strings
    program_arguments = cfg_2_mdgru(cfg.mdgru)
    # arguments to access lakeFS
    lakefs_arguments = ["--s3_endpoint", cfg.lakefs.s3_endpoint, "--access_key", cfg.lakefs.access_key,
                        "--secret_key", cfg.lakefs.secret_key, "--data_repository", cfg.lakefs.data_repository, "--branch", cfg.lakefs.branch]
    cache_arguments = ["--cache_path", cache_path_docker]
    print(cache_arguments)

    command = docker_run_command + program_arguments + lakefs_arguments + cache_arguments

    # run the docker image
    run_docker(command)



if __name__ == "__main__":
    # get the cli commands
    cli_conf = OmegaConf.from_cli()
    cfg = OmegaConf.load(cli_conf.config)
    cfg.datasplit.foldspath = os.path.expanduser(cfg.datasplit.foldspath)
    cfg.mdgru.datapath = os.path.expanduser(cfg.mdgru.datapath)
    cfg.mdgru.cache_path = os.path.expanduser(cfg.mdgru.cache_path)
    cfg.run.cache_path = os.path.expanduser(cfg.run.cache_path)

    # Process the additional argv-config parameters


    print("START MDGRU")
    docker_dispatcher = DockerDispatcher(cfg, cli_conf.config)
    docker_dispatcher.run()

    # delete the cache after the training
    if cfg.run.cache_delete:
        print("Delete cache ...")
        shutil.rmtree(cfg.run.cache_path)


