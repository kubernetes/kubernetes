# Add NVIDIA GPU discovery and assignment capabilities to Kubelet.

## Background:

So far we only support 1 GPU(```/dev/nvidia0```) and the GPU device need to be configured as a Kubelet's parameter manually. But in production environment, there could have multiple GPUs(```/dev/nvidia0, /dev/nvidia1, /dev/nvidia2 ...```) installed on 1 Minion, currently, we only support a single GPU, dev/nvidia0, on a node. In addition, using this GPU require manual configuration of the Kubelet.

This document proposes implementing support for auto-discovery of GPUs in an extensible manner, support for identifying multiple GPUs, and support for applications which require multiple GPUs.


## Goal:

 - Have the capabilities of GPU discovery to let Kubelet could aware more than 1 NVIDIA GPU and make the deployment easier.
 - Allow a single Kubernetes application to consume multiple GPUs.
 - Add a general GPU interface for all the GPU vendors for future use.
 - Add device support interface in runtime, so it could support both Docker and rkt.


## Implementation:

### Limitation:

In this implementation, an application may only request a number of GPUs. Additional resource dimensions, such as GPU model, memory available, or number of cores, will not be able to be specified by an application. But in the future, we could support GPU model, memory, cores, reference: https://developer.nvidia.com/nvidia-management-library-nvml.

The implementation will also only support NVIDIA GPUs.

### Features:

We need to add below features to enhance GPU(especial NVIDIA GPU) capabilities. The discovery and assignment need to be tracked by separate modules in Kubelet.

 1. Discover all the NVIDIA GPUs on a Minion, but do not need all the NVIDIA GPU details information(families, cores, memory). To discover all the NVIDIA GPUs on a node, we can simply identify all devices on the host matching /dev/nvidiaX. By directly mapping through entire devices, we can avoid using the NVIDIA Management Library (NVML). In the future, we will likely want to add this library in order to monitor per-core utilization and take advantage of other features that will allow GPUs to be shared between containers.
 
  1.1 Kubelet needs a list of GPU information, also need to report the number of free and total GPUs. Each general GPU device could be like:
    ```go
    type GPUDevice struct {
      ID   string	// GPU ID.
      Path string		 // The GPU path, like /dev/nvidia0, /dev/nvidia1, /dev/nvidia2...
    }
    ```

  1.2 Once Kubelet restart, the GPU information needs to be recovered from containers' imformation.

 2. Assign more than 1 NVIDIA GPU to 1 container, and each container could have multiple NVIDIA GPUs. A GPU can belong to at most one container. The devices host/container mapping also need to be added to the runtime information, just as the volume did. The container GPU mapping list could show the GPU assignment.
	
    ```go
    type ContainerGPUMapping struct {
      ContainerName string	// Which container this GPU attached. If it's NULL, the status can't be Free.
      GPUID         string	// The GPU assigned to container.
      Status        string	// InUse/Free/Unknow. If it's InUse, must has a container name.
    }
    ```
  
  2.1 Kubelet need to find free GPUs from the current ```GPUDevice``` list and ```ContainerGPUMapping``` list, and asssign them to container. The device path mapping on host and container are the same, like if we assgin ```/dev/nvidia1``` and ```/dev/nvidia2``` to a container, inside the container, we can see ```/dev/nvidia1``` and ```/dev/nvidia2``` as the same as on the host.
  
  2.2 Once a container dead/be killed, the GPUs attached to this container need to be freed. If the container be restarted, the GPUs attached will be freed and reassigned, if a pod is evicted, all the GPUs assigned to the containers in this pod will be immediately released.

 3. Provide general GPU interface for all the GPU vendors, like initialization, assign/free GPUs to container, available GPUs.


**Relevant issues**: #28216, #29509, #19047, #17035, #25557

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

