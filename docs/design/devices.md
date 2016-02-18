# K8s support "/sys/fs/cgroup/devices"
This document suggest a way for support hardware which could under "/sys/fs/cgroup/devices". So far, we could try to add GPU as a sample, then extend the others.

## Background
GPU is important to lots of workloads, such as machine learning, video stream decoding/coding, etc... But Kubernetes doesn't support schedule/tracking it which need to be.
Also, we can think to add all the devices could be assigned to container to k8s, it could make some hardware more efficient.

## Implementation
Related discussion:
https://github.com/kubernetes/kubernetes/issues/19049

The hardware resources should be known by both Scheduler and Kubelet, but for Scheduler, only need to know the attributes which could be a condition when scheduling/tracking, for Kubelet, need to report the devices to scheduler and have the ability to assign these devices to containers, which means Kubelet need to know more than it give to Scheduler. 

### Extend the Kubelet
The dockertools need to be extended to support the "--device", then we can assign devices to container.
Need to has the ability to send the devices status to Scheduler for schedule/track.
The devices need to be added as the volume or network,like: func ProbeXXXPlugins(pluginDir string) []XXX.XXXPlugin. All the schedule/track information should be provided in this plugin. The information of hardware for container assignment also should be provided in this plugin. (like GPU device path "/dev/nvidia0")

### Extend the Scheduler
Scheduler has the ability to schedule and track the hardware supported in Kubelet. The algorithm need to add the ability.
All the hardware should be described in 1 struct, so we can have expansibility for every hardware enabled in Kubelet.

### Extend the resource model
The resource for all the hardware under "/sys/fs/cgroup/devices" should be unified, and also should include the device name, then can scheduler/track different devices in 1 struct.

Resource types
Should add "devices" type for all the hardware under "/sys/fs/cgroup/devices", and also need to add the device name to distinguish each device type. The type could not be like "type ResourceList map[ResourceName]resource.Quantity" anymore, because the devices also need a name for each one. 


Resource quantities
For GPU, should include number to describe how many GPUs on a machine.



