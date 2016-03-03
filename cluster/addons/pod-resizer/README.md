# pod-resizer

This container image watches over another container, and vertically scales the
dependent container up and down. Currently the only option is to scale it
linearly based on the number of nodes, and it only works for a singleton.

## Nanny program and arguments

The nanny scales resources linearly with the number of nodes in the cluster. The base and marginal resource requirements are given as command line arguments, but you cannot give a marginal requirement without a base requirement.

The cluster size is periodically checked, and used to calculate the expected resources. If the expected and actual resources differ by more than the threshold (given as a +/- percent), then the deployment is updated (updating a deployment stops the old pod, and starts a new pod).

```
Usage of pod_nanny:
      --container="pod-nanny": The name of the container to watch. This defaults to the nanny itself.
      --cpu="MISSING": The base CPU resource requirement.
      --deployment="": The name of the deployment being monitored. This is required.
      --extra_cpu="0": The amount of CPU to add per node.
      --extra_memory="0Mi": The amount of memory to add per node.
      --extra_storage="0Gi": The amount of storage to add per node.
      --log-flush-frequency=5s: Maximum number of seconds between log flushes
      --memory="MISSING": The base memory resource requirement.
      --namespace="default": The namespace of the ward. This defaults to the nanny's own pod.
      --pod="nanny-v1-523901499-f3dm3": The name of the pod to watch. This defaults to the nanny's own pod.
      --poll_period=10000: The time, in milliseconds, to poll the dependent container.
      --storage="MISSING": The base storage resource requirement.
      --threshold=0: A number between 0-100. The dependent's resources are rewritten when they deviate from expected by more than threshold.
```
