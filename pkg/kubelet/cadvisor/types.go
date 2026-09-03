/*
Copyright 2015 The Kubernetes Authors.

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

//go:generate mockery
package cadvisor

import (
	"context"

	cadvisorapi "github.com/google/cadvisor/lib/model"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
)

// Interface is an abstract interface for testability.  It abstracts the interface to cAdvisor.
type Interface interface {
	Start() error
	ContainerInfoV2(name string, options cadvisorapi.RequestOptions) (map[string]cadvisorapi.ContainerInfo, error)
	GetRequestedContainersInfo(containerName string, options cadvisorapi.RequestOptions) (map[string]*cadvisorapi.ContainerInfo, error)
	MachineInfo(logger klog.Logger) (*cadvisorapi.MachineInfo, error)

	VersionInfo() (*cadvisorapi.VersionInfo, error)

	// Returns usage information about the filesystem holding container images.
	ImagesFsInfo(context.Context) (cadvisorapi.FsInfo, error)

	// Returns usage information about the root filesystem.
	RootFsInfo() (cadvisorapi.FsInfo, error)

	// Returns usage information about the writeable layer.
	// KEP 4191 can separate the image filesystem
	ContainerFsInfo(context.Context) (cadvisorapi.FsInfo, error)

	// Get filesystem information for the filesystem that contains the given file.
	GetDirFsInfo(path string) (cadvisorapi.FsInfo, error)
}

// ContainerEnumerator lists live CRI containers so a cAdvisor client can build
// per-container info where the underlying cAdvisor container discovery does not
// (e.g. Windows). The kubelet's internalapi.RuntimeService satisfies it.
type ContainerEnumerator interface {
	ListContainers(ctx context.Context, filter *runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error)
}

// ContainerEnumeratorSetter is implemented by cAdvisor clients that can accept
// a CRI runtime to populate per-container info for the metrics collector (e.g.
// container_oom_events_total on Windows). It is optional; Linux cAdvisor does
// not implement it, and the kubelet skips the wiring when the interface is
// absent.
type ContainerEnumeratorSetter interface {
	SetContainerEnumerator(ContainerEnumerator)
}

// ImageFsInfoProvider informs cAdvisor how to find imagefs for container images.
type ImageFsInfoProvider interface {
	// ImageFsInfoLabel returns the label cAdvisor should use to find the filesystem holding container images.
	ImageFsInfoLabel() (string, error)
	// In split image filesystem this will be different from ImageFsInfoLabel
	ContainerFsInfoLabel() (string, error)
}
