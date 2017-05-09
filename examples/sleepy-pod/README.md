## sleepy-pod example

This example shows how to run a pod named 'sleepy-pod' using Kubernetes and Docker.

sleepy-pod runs a process called sleepy that has a SIGINT and SIGTERM handler.
If it receives either of those signals, sleepy writes to /dev/termination-log. This path can be replaced by passing in a parameter to sleepy which contains the desired path.

### Step Zero: Prerequisites

This example assumes that you have forked the repository and [turned up a Kubernetes cluster](../../docs/getting-started-guides):

```shell
$ cd kubernetes
$ hack/dev-build-and-up.sh
```

### Step One: Launch sleepy-pod

Use the file `examples/sleepy-pod/sleepy-controller.json` to create a replication controller which manages a single pod. The pod runs sleepy in a container. Using a replication controller is the preferred way to launch long-running pods, even for 1 replica, so the pod will benefit from self-healing mechanism in kubernetes.

Create the sleepy replication controller in your Kubernetes cluster using the `kubectl` CLI:

```shell
$ cluster/kubectl.sh create -f examples/sleepy-pod/sleepy-controller.json
```

Once that's up you can list the replication controllers in the cluster:
```shell
$ cluster/kubectl.sh get rc
CONTROLLER                             CONTAINER(S)            IMAGE(S)                            SELECTOR                     REPLICAS
sleepy-controller                sleepy            kubernetes/sleepy                     name=sleepy       3
```

List pods in cluster to verify the master is running. You'll see three sleepy pods. It will also display the machine that the pod is running on once it gets placed (may take up to thirty seconds).

```shell
$ cluster/kubectl.sh get pods
POD                      IP           CONTAINER(S)        IMAGE(S)            HOST                                                             LABELS                   STATUS
sleepy-controller-hh2gd  10.244.3.5   sleepy              kubernetes/sleepy                   kubernetes-minion-npzh.c.abshah-kubernetes-001.internal/104.154.64.95    name=sleepy                                           Running
sleepy-controller-i7hvs  10.244.1.6   sleepy              kubernetes/sleepy                   kubernetes-minion-bugr.c.abshah-kubernetes-001.internal/130.211.176.68   name=sleepy                                           Running
sleepy-controller-nyxxv  10.244.3.6   sleepy                 kubernetes/sleepy                   kubernetes-minion-npzh.c.abshah-kubernetes-001.internal/104.154.64.95    name=sleepy                                           Running

```

If you ssh to that machine, you can run `docker ps` to see the actual pod:

```shell
me@workstation$ gcloud compute ssh --zone us-central1-b kubernetes-minion-4

me@kubernetes-minion-3:~$ sudo docker ps
CONTAINER ID        IMAGE                                  COMMAND                CREATED             STATUS
d39fab876dc3        kubernetes/sleepy:latest               "/app/sleepy"          4 minutes ago       Up 4 minutes                
2fbd489995de        kubernetes/sleepy:latest               "/app/sleepy"          4 minutes ago       Up 4 minutes                                      
```


### Step Two: Stop a container
If you stop a sleepy container, sleepy writes a termination message in /dev/sleepy-log. The log path is configurable in sleepy-controller.json.

```shell
me@kubernetes-minion-3:~$ sudo docker stop d39fab876dc3
me@kubernetes-minion-3:~$ cat /dev/sleepy-log
sleepy was killed. sob..
```

Shortly, if you check the number of pods, it will be back to 3, since the replication controller will recreate a sleepy pod to compensate for the one that you killed.


### Step Three: Cleanup

To turn down a Kubernetes cluster:

```shell
$ cluster/kube-down.sh
```

