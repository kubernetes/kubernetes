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
  1.1 Kubelet needs a list of GPU information, also need to report the number of free and total GPUs. Each general GPU device could be like:
    ```go
    type GPUDevice struct {
	   ContainerName string		 // Which container this GPU attached. If it's NULL, the Status can't be Free.
	   Path          string		 // The GPU path, like /dev/nvidia0, /dev/nvidia1, /dev/nvidia2...
	   Status        string    // InUse/Free/Unknow. If it's InUse, must has a container name.
    }
    ```
  1.2 Once Kubelet restart, the GPU information needs to be recovered from containers' imformation.

 2. Assign more than 1 NVIDIA GPU to 1 container, and each container could have multiple NVIDIA GPUs. The devices host/container mapping also need to be added to the runtime information, just as the volume did.
  2.1 Kubelet need to find a the free GPU from the ```GPUDevice``` list, and asssign them to container. The device path on host and container are the same, like if we assgin ```/dev/nvidia1``` and ```/dev/nvidia2``` to a container, inside the container, we can see ```/dev/nvidia1``` and ```/dev/nvidia2``` as the same as on the host.
  2.2 Once a container dead/be killed, the GPUs attached to this container need to be set free.

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

