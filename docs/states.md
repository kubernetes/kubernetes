# Pod State in Kubernetes

When one gets or lists Pods in Kubernetes, one of the fields returned is the PodStatus.  PodStatus is a string with the following possible values and semantics:

## All Restart policies
The following three status apply to all pods regardless of restart policy.

### PodWaiting
The Pod exists in the system, but hasn't been scheduled onto a host machine yet.

### PodScheduled
The Pod has been scheduled onto a host machine, but not all containers in the Pod have been started. This stage includes downloading images over the network, which may take a while.

### PodTerminated
The Pod exists in the system, but the system has decided to terminate it for some reason, for example a machine failure, or resource constraints.

### PodStopped
The Pod was stopped by an API call to delete the pod.

## RestartAlways
The following status definition is possible for pods with a ```RestartAlways``` policy.

### PodRunning
The Pod exists in the system and has been scheduled onto a machine and all containers have been started.


## RestartNever
The following definitions are true for pods with a ```RestartNever``` policy.

### PodRunning
The pod exists, has been scheduled, all containers have been started, at least one container is still running, and all other containers are either running or have exited with a zero status.

### PodFailed
The pod exists, has been scheduled, and at least one container has exited with a non-zero exit status.

### PodSucceeded
The pod exists, has been scheduled, and all containers have exited with a status of zero.

## RestartOnFailure
The following definitions are true for pods with a ```RestartOnFailure``` policy.

### PodRunning
The pod exists, has been scheduled, and all containers have been started, or they have exited with a status of zero.

### PodSucceeded
The pod exists, has been scheduled, and all containers have exited with a status of zero.
