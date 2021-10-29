/*
Copyright 2021 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package remote

import (
	"unsafe"

	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
)

func fromV1alpha2VersionResponse(from *v1alpha2.VersionResponse) *runtimeapi.VersionResponse {
	return (*runtimeapi.VersionResponse)(unsafe.Pointer(from))
}

func fromV1alpha2PodSandboxStatus(from *v1alpha2.PodSandboxStatus) *runtimeapi.PodSandboxStatus {
	return (*runtimeapi.PodSandboxStatus)(unsafe.Pointer(from))
}

func fromV1alpha2ListPodSandboxResponse(from *v1alpha2.ListPodSandboxResponse) *runtimeapi.ListPodSandboxResponse {
	return (*runtimeapi.ListPodSandboxResponse)(unsafe.Pointer(from))
}

func fromV1alpha2ListContainersResponse(from *v1alpha2.ListContainersResponse) *runtimeapi.ListContainersResponse {
	return (*runtimeapi.ListContainersResponse)(unsafe.Pointer(from))
}

func fromV1alpha2ContainerStatus(from *v1alpha2.ContainerStatus) *runtimeapi.ContainerStatus {
	return (*runtimeapi.ContainerStatus)(unsafe.Pointer(from))
}

func fromV1alpha2ExecResponse(from *v1alpha2.ExecResponse) *runtimeapi.ExecResponse {
	return (*runtimeapi.ExecResponse)(unsafe.Pointer(from))
}

func fromV1alpha2AttachResponse(from *v1alpha2.AttachResponse) *runtimeapi.AttachResponse {
	return (*runtimeapi.AttachResponse)(unsafe.Pointer(from))
}

func fromV1alpha2PortForwardResponse(from *v1alpha2.PortForwardResponse) *runtimeapi.PortForwardResponse {
	return (*runtimeapi.PortForwardResponse)(unsafe.Pointer(from))
}

func fromV1alpha2RuntimeStatus(from *v1alpha2.RuntimeStatus) *runtimeapi.RuntimeStatus {
	return (*runtimeapi.RuntimeStatus)(unsafe.Pointer(from))
}

func fromV1alpha2ContainerStats(from *v1alpha2.ContainerStats) *runtimeapi.ContainerStats {
	return (*runtimeapi.ContainerStats)(unsafe.Pointer(from))
}

func fromV1alpha2ImageFsInfoResponse(from *v1alpha2.ImageFsInfoResponse) *runtimeapi.ImageFsInfoResponse {
	return (*runtimeapi.ImageFsInfoResponse)(unsafe.Pointer(from))
}

func fromV1alpha2ListContainerStatsResponse(from *v1alpha2.ListContainerStatsResponse) *runtimeapi.ListContainerStatsResponse {
	return (*runtimeapi.ListContainerStatsResponse)(unsafe.Pointer(from))
}

func fromV1alpha2PodSandboxStats(from *v1alpha2.PodSandboxStats) *runtimeapi.PodSandboxStats {
	return (*runtimeapi.PodSandboxStats)(unsafe.Pointer(from))
}

func fromV1alpha2ListPodSandboxStatsResponse(from *v1alpha2.ListPodSandboxStatsResponse) *runtimeapi.ListPodSandboxStatsResponse {
	return (*runtimeapi.ListPodSandboxStatsResponse)(unsafe.Pointer(from))
}

func fromV1alpha2Image(from *v1alpha2.Image) *runtimeapi.Image {
	return (*runtimeapi.Image)(unsafe.Pointer(from))
}

func fromV1alpha2ListImagesResponse(from *v1alpha2.ListImagesResponse) *runtimeapi.ListImagesResponse {
	return (*runtimeapi.ListImagesResponse)(unsafe.Pointer(from))
}

func v1alpha2PodSandboxConfig(from *runtimeapi.PodSandboxConfig) *v1alpha2.PodSandboxConfig {
	return (*v1alpha2.PodSandboxConfig)(unsafe.Pointer(from))
}

func v1alpha2PodSandboxFilter(from *runtimeapi.PodSandboxFilter) *v1alpha2.PodSandboxFilter {
	return (*v1alpha2.PodSandboxFilter)(unsafe.Pointer(from))
}

func v1alpha2ContainerConfig(from *runtimeapi.ContainerConfig) *v1alpha2.ContainerConfig {
	return (*v1alpha2.ContainerConfig)(unsafe.Pointer(from))
}

func v1alpha2ContainerFilter(from *runtimeapi.ContainerFilter) *v1alpha2.ContainerFilter {
	return (*v1alpha2.ContainerFilter)(unsafe.Pointer(from))
}

func v1alpha2LinuxContainerResources(from *runtimeapi.LinuxContainerResources) *v1alpha2.LinuxContainerResources {
	return (*v1alpha2.LinuxContainerResources)(unsafe.Pointer(from))
}

func v1alpha2ExecRequest(from *runtimeapi.ExecRequest) *v1alpha2.ExecRequest {
	return (*v1alpha2.ExecRequest)(unsafe.Pointer(from))
}

func v1alpha2AttachRequest(from *runtimeapi.AttachRequest) *v1alpha2.AttachRequest {
	return (*v1alpha2.AttachRequest)(unsafe.Pointer(from))
}

func v1alpha2PortForwardRequest(from *runtimeapi.PortForwardRequest) *v1alpha2.PortForwardRequest {
	return (*v1alpha2.PortForwardRequest)(unsafe.Pointer(from))
}

func v1alpha2RuntimeConfig(from *runtimeapi.RuntimeConfig) *v1alpha2.RuntimeConfig {
	return (*v1alpha2.RuntimeConfig)(unsafe.Pointer(from))
}

func v1alpha2ContainerStatsFilter(from *runtimeapi.ContainerStatsFilter) *v1alpha2.ContainerStatsFilter {
	return (*v1alpha2.ContainerStatsFilter)(unsafe.Pointer(from))
}

func v1alpha2PodSandboxStatsFilter(from *runtimeapi.PodSandboxStatsFilter) *v1alpha2.PodSandboxStatsFilter {
	return (*v1alpha2.PodSandboxStatsFilter)(unsafe.Pointer(from))
}

func v1alpha2ImageFilter(from *runtimeapi.ImageFilter) *v1alpha2.ImageFilter {
	return (*v1alpha2.ImageFilter)(unsafe.Pointer(from))
}

func v1alpha2ImageSpec(from *runtimeapi.ImageSpec) *v1alpha2.ImageSpec {
	return (*v1alpha2.ImageSpec)(unsafe.Pointer(from))
}

func v1alpha2AuthConfig(from *runtimeapi.AuthConfig) *v1alpha2.AuthConfig {
	return (*v1alpha2.AuthConfig)(unsafe.Pointer(from))
}
