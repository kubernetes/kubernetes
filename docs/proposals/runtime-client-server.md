<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Client/Server container runtime

## Abstract

A proposal of client/server implementation of kubelet container runtime interface.

## Motivation

Currently, any container runtime has to be linked into the kubelet. This makes
experimentation difficult, and prevents users from landing an alternate
container runtime without landing code in core kubernetes.

To facilitate experimentation and to enable user choice, this proposal adds a
client/server implementation of the [new container runtime interface](https://github.com/kubernetes/kubernetes/pull/25899). The main goal
of this proposal is:

- make it easy to integrate new container runtimes
- improve code maintainability

## Proposed design

**Design of client/server container runtime**

The main idea of client/server container runtime is to keep main control logic in kubelet while letting remote runtime only do dedicated actions. An alpha [container runtime API](../../pkg/kubelet/api/v1alpha1/runtime/api.proto) is introduced for integrating new container runtimes. The API is based on [protobuf](https://developers.google.com/protocol-buffers/) and [gRPC](http://www.grpc.io) for a number of benefits:

- Perform faster than json
- Get client bindings for free: gRPC supports ten languages
- No encoding/decoding codes needed
- Manage api interfaces easily: server and client interfaces are generated automatically

A new container runtime manager `KubeletGenericRuntimeManager` will be introduced to kubelet, which will

- conforms to kubelet's [Runtime](../../pkg/kubelet/container/runtime.go#L58) interface
- manage Pods and Containers lifecycle according to kubelet policies
- call remote runtime's API to perform specific pod, container or image operations

A simple workflow of invoking remote runtime API on starting a Pod with two containers can be shown:

```
Kubelet                  KubeletGenericRuntimeManager       RemoteRuntime
   +                              +                               +
   |                              |                               |
   +---------SyncPod------------->+                               |
   |                              |                               |
   |                              +---- Create PodSandbox ------->+
   |                              +<------------------------------+
   |                              |                               |
   |                              XXXXXXXXXXXX                    |
   |                              |          X                    |
   |                              |    NetworkPlugin.             |
   |                              |       SetupPod                |
   |                              |          X                    |
   |                              XXXXXXXXXXXX                    |
   |                              |                               |
   |                              +<------------------------------+
   |                              +----    Pull image1   -------->+
   |                              +<------------------------------+
   |                              +---- Create container1 ------->+
   |                              +<------------------------------+
   |                              +---- Start container1 -------->+
   |                              +<------------------------------+
   |                              |                               |
   |                              +<------------------------------+
   |                              +----    Pull image2   -------->+
   |                              +<------------------------------+
   |                              +---- Create container2 ------->+
   |                              +<------------------------------+
   |                              +---- Start container2 -------->+
   |                              +<------------------------------+
   |                              |                               |
   | <-------Success--------------+                               |
   |                              |                               |
   +                              +                               +
```

And deleting a pod can be shown:

```
Kubelet                  KubeletGenericRuntimeManager      RemoteRuntime
   +                              +                               +
   |                              |                               |
   +---------SyncPod------------->+                               |
   |                              |                               |
   |                              +----   Stop container1   ----->+
   |                              +<------------------------------+
   |                              +----  Delete container1  ----->+
   |                              +<------------------------------+
   |                              |                               |
   |                              +----   Stop container2   ------>+
   |                              +<------------------------------+
   |                              +----  Delete container2  ------>+
   |                              +<------------------------------+
   |                              |                               |
   |                              XXXXXXXXXXXX                    |
   |                              |          X                    |
   |                              |    NetworkPlugin.             |
   |                              |       TeardownPod             |
   |                              |          X                    |
   |                              XXXXXXXXXXXX                    |
   |                              |                               |
   |                              |                               |
   |                              +---- Delete PodSandbox  ------>+
   |                              +<------------------------------+
   |                              |                               |
   | <-------Success--------------+                               |
   |                              |                               |
   +                              +                               +
```

**API definition**

Since we are going to introduce more image formats and want to separate image management from containers and pods, this proposal introduces two services `RuntimeService` and `ImageService`. Both services are defined at [pkg/kubelet/api/v1alpha1/runtime/api.proto](../../pkg/kubelet/api/v1alpha1/runtime/api.proto):

```proto
// Runtime service defines the public APIs for remote container runtimes
service RuntimeService {
    // Version returns the runtime name, runtime version and runtime API version
    rpc Version(VersionRequest) returns (VersionResponse) {}

    // CreatePodSandbox creates a pod-level sandbox.
    // The definition of PodSandbox is at https://github.com/kubernetes/kubernetes/pull/25899
    rpc CreatePodSandbox(CreatePodSandboxRequest) returns (CreatePodSandboxResponse) {}
    // StopPodSandbox stops the sandbox. If there are any running containers in the
    // sandbox, they should be force terminated.
    rpc StopPodSandbox(StopPodSandboxRequest) returns (StopPodSandboxResponse) {}
    // DeletePodSandbox deletes the sandbox. If there are any running containers in the
    // sandbox, they should be force deleted.
    rpc DeletePodSandbox(DeletePodSandboxRequest) returns (DeletePodSandboxResponse) {}
    // PodSandboxStatus returns the Status of the PodSandbox.
    rpc PodSandboxStatus(PodSandboxStatusRequest) returns (PodSandboxStatusResponse) {}
    // ListPodSandbox returns a list of SandBox.
    rpc ListPodSandbox(ListPodSandboxRequest) returns (ListPodSandboxResponse) {}

    // CreateContainer creates a new container in specified PodSandbox
    rpc CreateContainer(CreateContainerRequest) returns (CreateContainerResponse) {}
    // StartContainer starts the container.
    rpc StartContainer(StartContainerRequest) returns (StartContainerResponse) {}
    // StopContainer stops a running container with a grace period (i.e., timeout).
    rpc StopContainer(StopContainerRequest) returns (StopContainerResponse) {}
    // RemoveContainer removes the container. If the container is running, the container
    // should be force removed.
    rpc RemoveContainer(RemoveContainerRequest) returns (RemoveContainerResponse) {}
    // ListContainers lists all containers by filters.
    rpc ListContainers(ListContainersRequest) returns (ListContainersResponse) {}
    // ContainerStatus returns status of the container.
    rpc ContainerStatus(ContainerStatusRequest) returns (ContainerStatusResponse) {}

    // Exec executes the command in the container.
    rpc Exec(stream ExecRequest) returns (stream ExecResponse) {}
}

// Image service defines the public APIs for managing images
service ImageService {
    // ListImages lists existing images.
    rpc ListImages(ListImagesRequest) returns (ListImagesResponse) {}
    // ImageStatus returns the status of the image.
    rpc ImageStatus(ImageStatusRequest) returns (ImageStatusResponse) {}
    // PullImage pulls a image with authentication config.
    rpc PullImage(PullImageRequest) returns (PullImageResponse) {}
    // RemoveImage removes the image.
    rpc RemoveImage(RemoveImageRequest) returns (RemoveImageResponse) {}
}
```

Note that some types in [pkg/kubelet/api/v1alpha1/runtime/api.proto](../../pkg/kubelet/api/v1alpha1/runtime/api.proto) are already defined at [Container runtime interface/integration](https://github.com/kubernetes/kubernetes/pull/25899).
We should decide how to integrate the types in [#25899](https://github.com/kubernetes/kubernetes/pull/25899) with gRPC services:

* Auto-generate those types into protobuf by [go2idl](../../cmd/libs/go2idl/)
  - Pros:
    - trace type changes automatically, all type changes in Go will be automatically generated into proto files
  - Cons:
    - type change may break existing API implementations, e.g. new fields added automatically may not noticed by remote runtime
    - needs to convert Go types to gRPC generated types, and vise versa
    - needs processing attributes order carefully so as not to break generated protobufs (this could be done by using [protobuf tag](https://developers.google.com/protocol-buffers/docs/gotutorial))
    - go2idl doesn't support gRPC, [protoc-gen-gogo](https://github.com/gogo/protobuf) is still required for generating gRPC client
* Embed those types as raw protobuf definitions and generate Go files by [protoc-gen-gogo](https://github.com/gogo/protobuf)
  - Pros:
    - decouple type definitions, all type changes in Go will be added to proto manually, so it's easier to track gRPC API version changes
    - Kubelet could reuse Go types generated by `protoc-gen-gogo` to avoid type conversions
  - Cons:
    - duplicate definition of same types
    - hard to track type changes automatically
    - need to manage proto files manually

For better version controlling and fast iterations, this proposal embeds all those types in `api.proto` directly.

## Implementation

Each new runtime should implement the [gRPC](http://www.grpc.io) server based on [pkg/kubelet/api/v1alpha1/runtime/api.proto](../../pkg/kubelet/api/v1alpha1/runtime/api.proto). For version controlling, `KubeletGenericRuntimeManager` will request `RemoteRuntime`'s `Version()` interface with the runtime api version. To keep backward compatibility, the API follows standard [protobuf guide](https://developers.google.com/protocol-buffers/docs/proto) to deprecate or add new interfaces.

A new flag `--container-runtime-endpoint` (overrides `--container-runtime`) will be introduced to kubelet which identifies the unix socket file of the remote runtime service. And new flag `--image-service-endpoint` will be introduced to kubelet which identifies the unix socket file of the image service.

To facilitate switching current container runtime (e.g. `docker` or `rkt`) to new runtime API, `KubeletGenericRuntimeManager` will provide a plugin mechanism allowing to specify local implementation or gRPC implementation.

## Community Discussion

This proposal is first filed by [@brendandburns](https://github.com/brendandburns) at [kubernetes/13768](https://github.com/kubernetes/kubernetes/issues/13768):

* [kubernetes/13768](https://github.com/kubernetes/kubernetes/issues/13768)
* [kubernetes/13709](https://github.com/kubernetes/kubernetes/pull/13079)
* [New container runtime interface](https://github.com/kubernetes/kubernetes/pull/25899)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/runtime-client-server.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
