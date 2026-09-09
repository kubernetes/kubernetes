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

// ImageFsInfoProvider informs cAdvisor how to find imagefs for container images.
type ImageFsInfoProvider interface {
	// ImageFsInfoLabel returns the label cAdvisor should use to find the filesystem holding container images.
	ImageFsInfoLabel() (string, error)
	// In split image filesystem this will be different from ImageFsInfoLabel
	ContainerFsInfoLabel() (string, error)
}
