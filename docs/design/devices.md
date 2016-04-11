# Kubernetes support "GPU"
This document suggest a way for support GPU in kubernetes. And kubernetes could has the below abilities after this be implemented. (In the current version, we only implement NVIDIA GPU, others will come soon!)

- Kubelet could provide GPU information and update them to Scheduler.
- Kubelet could assign GPU to container.
- Scheduler could scheduler if there has free GPUs on a Minion.
- Kubectl could create Pod with containers have GPU requreiment.

Please notice this:
The Kubernetes environment has no responsibility for GPU environment setup on host. All the GPU relative componments works fine based on the host configured correctlly.
Basically, the below configuration on host should be ready before use GPU inside container:
- modprobe nvidia and nvidia-uvm kernel modules. (or raedeon or whatever)(If the driver and CUDA installed correct, the GPUs should could be mounted once the system boot up)
- the nvidia-dmi could detect what pci devices exist and their characteristics (e.g. Vendor = Nvidia, Model = "GeForce GTX 480")
- mknod /dev/nvidia0, /dev/nvidiactl, /dev/nvidia-uvm, etc. (If the driver and CUDA installed correct, the GPUs should could be mounted once the system boot up)
- install the CUDA driver (not the kernel module).
- The container image that will use the GPU should have the GPU detail information labeled, like below, but finally we will move it to pod's yaml file.
            "Labels": {
                "com.nvidia.cuda.version": "7.5",
                "com.nvidia.volumes.needed": "nvidia_driver"
            },


## Background
GPU is important to lots of workloads, such as machine learning, video stream decoding/coding, etc... But Kubernetes doesn't support schedule/tracking it which need to be.


## Implementation
Related discussion:

https://github.com/kubernetes/kubernetes/issues/19049

Related implementation: 

All the codes below could found at below branch, and the codes has been verified, it could scheduler the GPU resources and assign GPU to container, the app inside container runs success.
https://github.com/Hui-Zhi/kubernetes/tree/try_nvidia_docker_merge_devices_support

The GPU should be known by both Scheduler and Kubelet, but for Scheduler, only need to know the attributes which could be a condition when scheduling/tracking, for Kubelet, need to report the GPU to scheduler and have the ability to assign these GPUs to containers, which means Kubelet need to know more than it give to Scheduler. 

### Extend the Kubelet
Kubelet should responsible for the GPU information collection/update/allocation/release.

#### 1. Provide GPU information.
Need to has the ability to send the GPU status to Scheduler for scheduling/tracking. The GPU need to be added as the volume or network,like: func ProbeXXXPlugins(pluginDir string) []XXX.XXXPlugin. All the scheduling/tracking information should be provided in this plugin. The information of GPU for container allocation/release also should be provided in this plugin. (like GPU device path "/dev/nvidia0")

The responsibility of the GPU plugin in Kubelet:

	1). Implement different subplugin for each of vendor. But all the GPU resources should be in 1 plugin. Like:
		func ProbeGPUPlugins() []gpuTypes.GPUPlugin {
          glog.Infof("GPUPlugin: ProbeGPUPlugins")
          allPlugins := []gpuTypes.GPUPlugin{}

          // add cuda plugin
          cudaPlugin := cuda.ProbeGPUPlugin()
          ...
          // add other plugins
          xxxxPlugin := xxxx.ProbeGPUPlugin()
        }
		
	2). Discover the GPU information on host. (The host need to have CUDA installed) The way we discover GPU and its information by nvidia-smi, the same way used in NVIDIA-Docker.
	
	3). Collect/Update the GPU status to Scheduler, like:
	    func IsGPUAvailable(pods []*api.Pod, gpuCapacity int) bool {
          glog.Infof("GPUPlugin: IsGPUAvaiable()")
          totalGPU := gpuCapacity
          totalGPURequest := int(0)
  
          for _, pod := range pods {
            totalGPURequest += getGPUResourceRequest(pod)
          }
  
          glog.Infof("Kubelet: IsGPUAvailable: totalGPU: %d, totalGPURequest: %d", totalGPU, totalGPURequest)
          return totalGPURequest == 0 || (totalGPU-totalGPURequest) >= 0
        }

	4). Allocate/Release GPU to/from container. The Plugin should could mount the GPU relative libraries to the container, like build the list of devices to be exposed in the container (e.g. /dev/nvidia0, /dev/nvidiactl, /dev/nvidia-uvm). Do it as NVIDIA-Docker. 

#### 2. Container with GPU requirement.
	1). "--device" ability to add GPU to container. The dockertools need to be extended to support the "--device", then we can assign GPU to container, also other hardware could benifit from it if we want to add more hardware such as GPU.
	2). Allocate/Release GPU to/from container.
	
### Extend the Scheduler
Scheduler has the ability to schedule and track the hardware supported in Kubelet. The algorithm need to add the ability.
#### 1. GPU information.
	Scheduler only can get the number of GPU information. we can call it like NVIDIA-GPU to separate it from other vendors' GPU. But in the future, we need to redesign it, like:
	type Devices struct {
		Name string
		Vendor string
		Number uint
	}
	then we can add more devices without change the general struct.
	
#### 2. The scheduler should treat GPU as dedicate resource, so can't like CPU and memory, it should as below:
	fitsCPU := totalMilliCPU == 0 || (totalMilliCPU-milliCPURequested) >= podRequest.milliCPU
    fitsMemory := totalMemory == 0 || (totalMemory-memoryRequested) >= podRequest.memory
    fitGPU := (totalGPU - GPURequested) >= podRequest.GPU


### Extend the resource model

Resource quantities
Should include number to describe how many GPUs on a machine.


### yaml file example:

apiVersion: v1
kind: Pod
metadata:
  name: nvidia-cuda
spec:
  containers:
  - name: nvidia-cuda
    image: nvidia_cuda:v1
    command: [ "sh", "/test.sh"]
    resources:
      requests:
        cpu: 100m
        memory: 100Mi
        nvidiagpu: 1




