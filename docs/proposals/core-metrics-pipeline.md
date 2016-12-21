# Core Metrics Pipeline in kubelet

**Author**: David Ashpole (@dashpole)

**Last Updated**: 12/21/2016

**Status**: Draft Proposal (WIP)

This document proposes a design for an internal Core Metrics Pipeline.

<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Core Metrics Pipeline in kubelet](#core-metrics-pipeline-in-kubelet)
  - [Introduction](#introduction)
    - [Background](#background)
    - [Motivations](#motivations)
    - [Proposal](#proposal)
    - [Non Goals](#non-goals)
  - [Design](#design)
    - [Proposed Core Metrics API:](#proposed-core-metrics-api)
    - [Core Machine Info:](#core-machine-info)
  - [Implementation Plan](#implementation-plan)
  - [Rollout Plan](#rollout-plan)
  - [Implementation Status](#implementation-status)

<!-- END MUNGE: GENERATED_TOC -->

## Introduction

### Background

CAdvisor is an open source container monitoring solution which only monitors containers, and has no concept of k8s constructs like pods or volumes.  Kubernetes vendors cAdvisor into its codebase, and uses cAdvisor as a library with functions that enable it to collect metrics on containers.  The kubelet can then combine container-level metrics from cAdvisor with the kubelet's knowledge of k8s constructs like pods to produce the kubelet Summary statistics, which provides metrics for use by the kubelet, or by users through the Summary API.  cAdvisor works by collecting metrics at an interval (10 seconds), and the kubelet then simply querries these cached metrics whenever it has a need for them.

Currently, cAdvisor collects a large number of metrics related to system and container performance. However, only some of these metrics are consumed by the kubelet summary API, and many are not used.  The kubelet summary API is published to the kubelet summary API endpoint.  Some of the metrics provided by the summary API are consumed internally, but most are not used internally.

### Motivations

Giving the kubelet the role of both providing metrics for its own use, and providing metrics for users has a couple problems
 - First, it is clearly inefficent to collect metrics and not use them.  The kubelet uses only a small portion of the metrics it collects, and thus presents considerable extra overhead to any users who do not use them, or prefer a third party monitoring solution.
 - Second, as the number of metrics collected grows over time, the kubelet will gain more and more overhead for collecting, processing, and publishing these metrics.  Since the metrics users may want is unbounded, the kubelet's resource overhead could easily grow to unreasonable levels.

It is very cumbersome to make changes or bugfixes in cAdvisor, because that then needs to be vendored back into kubernetes.

CAdvisor is structured to collect metrics on an interval, which is appropriate for a stand-alone metrics collector.  However, many functions in the kubelet are latency-sensitive (eviction, for example), and would benifit from a more "On-Demand" metrics collection design.

### Proposal

I propose to have a two seperate monitoring pipelines.
 - The first is an internal, "Core Metrics" pipeline, which provides only those metrics consumed by kubernetes system components.
 - The second is an external, independent "Third Party Metrics" pipeline, which provides metrics for users of kubernetes.

The proposed metrics pipelines would bound the resources that the kubelet consumes, and create a seperation of concerns for kubelet metrics and user metrics.  This proposal also paves the way for simpler third party metrics integration, although that is not covered in this proposal.

### Non Goals

Third Party Metric Provider integration will not be tackled in this proposal. This includes both possibly providing a kubelet metadata (pod-container mappings, volume-pod mappings, etc) API, and a possible standardized metrics APIs for metrics providers to conform to.

Integration with CRI will not be covered in this proposal.  In future proposals, integrating with CRI may provide a better abstraction of information required by the core metrics pipeline to collect metrics.

## Design

This design covers only the internal Core Metrics Pipeline.

High level requirements for the design are as follows:
 - Do not break existing users.  We should continue to provide the full summary API by default.
 - The kubelet collects the minimum possible number of metrics for full kubernetes functionality.
 - Code for collecting core metrics resides in the kubernetes codebase.
 - Metrics can be fetched "On Demand", giving the kubelet more up-to-date stats.

More details on how I intend to achieve these high level goals can be found in the Implementation Plan.

In order to continue to provide the full summary API, either the kubelet or a stand-alone version of cAdvisor will need to publish these metrics.

I propose that this Core Metrics API be an unversioned, internal API so that making changes is simpler.  However, if scheduling decisions are made based on these metrics, it will have to be versioned to handle version skew between the API server and Kubelet.

This proposal purposefully omits many metrics that may eventually become core metrics.  This is by design.  Once metrics are needed for an internal use case, they can be added to the core metrics API.

### Proposed Core Metrics API:

An important difference between the current summary api and the proposed core metrics api is that per-pod stats in the core metrics api contain only usage data, and not capacity-related statistics.  This is more accurate since a pod's resource capacity is really defined by its "requests" and "limits", and it is a better reflection of how the kubelet uses the data.  The kubelet finds which resources are constrained using node-level capacity and availability data, and then chooses which pods to take action on based on the pod's usage of the constrained resource.  If neccessary, capacity for resources a pod consumes can still be correlated with node-level resources using this format of stats.

    // CoreStats is a top-level container for holding NodeStats and PodStats.  
    type CoreStats struct {  
      // Overall node stats.  
      Node NodeStats `json:"node"`  
      // Per-pod stats.  
      Pods []PodStats `json:"pods"`  
    }  

    // NodeStats holds node-level stats.  NodeStats contains capacity and availibility for Node Resources.  
    type NodeStats struct {  
      // Stats pertaining to memory (RAM) resources.  
      // +optional  
      Memory *MemoryStats `json:"memory,omitempty"`  
      // Stats pertaining to total usage of filesystem resources on the filesystem used by node k8s components.  
      // +optional  
      KubeletFs *FsStats `json:"kubeletfs,omitempty"`  
      // Stats pertaining to total usage of filesystem resources on the filesystem used by the runtime.  
      // +optional  
      RuntimeFs *FsStats `json:"runtimefs,omitempty"`  
    }  

    // MemoryStats contains data about memory usage.  
    type MemoryStats struct {  
      // The memory capacity, in bytes  
      CapacityBytes *uint64  
      // The available memory, in bytes  
      AvailableBytes *uint64  
    }  

    // FsStats contains data about filesystem usage.  
    type FsStats struct {  
      // The device that this filesystem is on  
      Device string `json:"device"`  
      // AvailableBytes represents the storage space available (bytes) for the filesystem.  
      // +optional  
      AvailableBytes *uint64 `json:"availableBytes,omitempty"`  
      // CapacityBytes represents the total capacity (bytes) of the filesystems underlying storage.  
      // +optional  
      CapacityBytes *uint64 `json:"capacityBytes,omitempty"`  
      // InodesFree represents the free inodes in the filesystem.  
      // +optional  
      InodesFree *uint64 `json:"inodesFree,omitempty"`  
      // Inodes represents the total inodes in the filesystem.  
      // +optional  
      Inodes *uint64 `json:"inodes,omitempty"`  
    }  

    // PodStats holds pod-level unprocessed sample stats.  
    type PodStats struct {  
      // UID of the pod  
      PodUID string `json:"uid"`  
      // Stats of containers in the measured pod.  
      Containers []ContainerStats `json:"containers" patchStrategy:"merge" patchMergeKey:"name"`  
      // Stats pertaining to volume usage of filesystem resources.  
      // VolumeStats.UsedBytes is the number of bytes used by the Volume  
      // +optional  
      VolumeStats []DiskUsageStats `json:"volume,omitempty" patchStrategy:"merge" patchMergeKey:"name"`  
    }  

    // ContainerStats holds container-level unprocessed sample stats.  
    type ContainerStats struct {  
      // Stats pertaining to contianer usage of system memory  
      // +optional  
      Memory *MemoryUsageStats `json:"memory,omitempty"`  
      // Stats pertaining to container rootfs usage of disk.  
      // Rootfs.UsedBytes is the number of bytes used for the container write layer.  
      // +optional  
      Rootfs *DiskUsageStats `json:"rootfs,omitempty"`  
      // Stats pertaining to container logs usage of Disk.  
      // +optional  
      Logs *DiskUsageStats `json:"logs,omitempty"`  
    }  

    type MemoryUsageStats struct {  
      // The amount of working set memory. This includes recently accessed memory,  
      // dirty memory, and kernel memory.  
      // +optional  
      WorkingSetBytes *uint64 `json:"workingSetBytes,omitempty"`  
    }  

    type DiskUsageStats struct {  
      // The device on which resources are consumed  
      Device string `json:"device"`  
      // UsedBytes represents the disk space consumed on the device, in bytes.  
      // +optional  
      UsedBytes uint64 `json:"usedBytes,omitempty"`  
      // InodesUsed represents the inodes consumed on the device  
      // +optional  
      InodesUsed *uint64 `json:"inodesUsed,omitempty"`  
    }  

### Core Machine Info:

In addition to providing metrics, cAdvisor also provides machine info.  While it is not neccessary to use this structure for reporting Machine Info, the following contains all of the information provided to the kubelet by cAdvisor, generally used at startup.

The code that provides this data currently resides in cAdvisor.  I propose moving this to the kubelet as well.

    type CoreInfo struct {  
      // MachineID reported by the node. For unique machine identification  
      // in the cluster this field is prefered. Learn more from man(5)  
      // machine-id: http://man7.org/linux/man-pages/man5/machine-id.5.html  
      MachineID string `json:"machineID" protobuf:"bytes,1,opt,name=machineID"`  
      // SystemUUID reported by the node. For unique machine identification  
      // MachineID is prefered. This field is specific to Red Hat hosts  
      // https://access.redhat.com/documentation/en-US/Red_Hat_Subscription_Management/1/html/RHSM/getting-system-uuid.html  
      SystemUUID string `json:"systemUUID" protobuf:"bytes,2,opt,name=systemUUID"`  
      // Boot ID reported by the node.  
      BootID string `json:"bootID" protobuf:"bytes,3,opt,name=bootID"`  
      // Kernel Version reported by the node from 'uname -r' (e.g. 3.16.0-0.bpo.4-amd64).  
      KernelVersion string `json:"kernelVersion" protobuf:"bytes,4,opt,name=kernelVersion"`  
      // OS Image reported by the node from /etc/os-release (e.g. Debian GNU/Linux 7 (wheezy)).  
      OSImage string `json:"osImage" protobuf:"bytes,5,opt,name=osImage"`  
      // Capacity represents the total resources of a node.  
      // More info: http://kubernetes.io/docs/user-guide/persistent-volumes#capacity for more details.  
      // +optional  
      // ContainerRuntime Version reported by the node through runtime remote API (e.g. docker://1.5.0).  
      ContainerRuntimeVersion string `json:"containerRuntimeVersion" protobuf:"bytes,6,opt,name=containerRuntimeVersion"`  
      // Cloud provider the machine belongs to.  
      CloudProvider CloudProvider `json:"cloud_provider"`  
      // ID of cloud instance (e.g. instance-1) given to it by the cloud provider.  
      InstanceID InstanceID `json:"instance_id"`  
      // The number of cores in this machine.  
      NumCores int `json:"num_cores"`  
      // The amount of memory (in bytes) in this machine  
      MemoryCapacity uint64 `json:"memory_capacity"`  
    }  

    type CloudProvider string  

    const (  
      GCE             CloudProvider = "GCE"  
      AWS                           = "AWS"  
      Azure                         = "Azure"  
      Baremetal                     = "Baremetal"  
      UnknownProvider               = "Unknown"  
    )  

## Implementation Plan

I will move all code pertaining to collection and processing of core metrics from cAdvisor into kubernetes.  
I will vendor the new core metrics code back into cAdvisor. 
I will modify volume stats collection so that it relies on this code.  
I will modify the structure of stats collection code to be "On-Demand"   

Tenative future work, not included in this proposal:  
Obtain all runtime-specific information needed to collect metrics from the CRI.  
Create a third party metadata API, whose function is to provide third party monitoring solutions with kubernetes-specific data (pod-container relationships, for example).  
Modify cAdvisor to be "stand alone", and run in a seperate binary from the kubelet.  It will consume the above metadata API, and provide the summary API.  
The kubelet no longer provides the summary API, and starts cAdvisor stand-alone by default.  Include flag to not start cAdvisor.  

## Rollout Plan

TBD

## Implementation Status

The implementation goals of the first milestone are outlined below.
- [ ] Create the proposal
- [ ] Move all code relevant for the collection and processing of core metrics from cAdvisor into kubernetes.
- [ ] Vendor the new core metrics code back into cAdvisor.
- [ ] Modify volume stats collection so that it relies on this code.
- [ ] Modify the structure of stats collection code to be "On-Demand"



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/core-metrics-pipeline.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
