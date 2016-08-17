# Add NVIDIA GPU discovery and assignment capabilities to Kubelet.

## Background:

So far we only support 1 GPU(```/dev/nvidia0```) and the GPU device need to be configured as a Kubelet's parameter manually. But in production environment, there could have multiple GPUs(```/dev/nvidia0, /dev/nvidia1, /dev/nvidia2 ...```) installed on 1 Minion, also 1 micro service could needs more than 1 GPU, and there could have thousands of Minions which means if Kubelet could discover GPU by itself would be great.


## Goal:

 - Have the capabilities of GPU discovery to let Kubelet could aware more than 1 NVIDIA GPU and make the deployment easier.
 - Assign more than 1 GPU to 1 container to make the micro service with multiple GPUs requirement run on Kubernetes possible.
 - Add a general GPU interface for all the GPU vendors for future use.


## Implementation:

### Limitation:

In this implementation, we still only schedule dedicated NVIDIA GPU and NVIDIA GPU supported only, and will not be NVIDIA GPU details(familiies, cores, memory) sensitive.

### Features:

We need to add below features to enhance GPU(especial NVIDIA GPU) capabilities. 

 1. Discover all the NVIDIA GPUs on a Minion. Without the NVIDIA GPU details information(families, cores, memory) needed, we can simply use regex to discover all the ```/dev/nvidiaN``` devices, so the NVML(NVIDIA Management Library) won't be necessary, but in the future if we want to schedule/assign GPU cores/memories the NVML libs is unavoidable.
 
 2. Assign more than 1 NVIDIA GPU to a container. The devices host/container mapping also need to be added to the runtime information, just as the volume did.
 
 3. Provide general GPU interface for all the GPU vendors, like initialization, assign/free GPUs to container, available GPUs.


**Relative issues**: #28216, #29509, #19047, #17035, #25557

#### Pod yaml example:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nvidia-gpu-test
spec:
  containers:
  - name: nvidia-gpu
    image: gpu-image
    resources:
      limits:
        alpha.kubernetes.io/nvidia-gpu: 2
```

