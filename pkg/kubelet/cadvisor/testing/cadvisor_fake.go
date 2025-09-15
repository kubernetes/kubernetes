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

package testing

import (
	"context"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
)

// Fake cadvisor.Interface implementation.
type Fake struct {
	NodeName string
}

const (
	// FakeKernelVersion is a fake kernel version for testing.
	FakeKernelVersion = "3.16.0-0.bpo.4-amd64"
	// FakeContainerOSVersion is a fake OS version for testing.
	FakeContainerOSVersion = "Debian GNU/Linux 7 (wheezy)"

	fakeNumCores       = 1
	fakeMemoryCapacity = 4026531840
	fakeDockerVersion  = "1.13.1"
)

var _ cadvisor.Interface = new(Fake)

// Start is a fake implementation of Interface.Start.
func (c *Fake) Start() error {
	return nil
}

// ContainerInfoV2 is a fake implementation of Interface.ContainerInfoV2.
func (c *Fake) ContainerInfoV2(name string, options cadvisorapiv2.RequestOptions) (map[string]cadvisorapiv2.ContainerInfo, error) {
	return map[string]cadvisorapiv2.ContainerInfo{}, nil
}

// GetRequestedContainersInfo is a fake implementation if Interface.GetRequestedContainersInfo
func (c *Fake) GetRequestedContainersInfo(containerName string, options cadvisorapiv2.RequestOptions) (map[string]*cadvisorapi.ContainerInfo, error) {
	return map[string]*cadvisorapi.ContainerInfo{}, nil
}

// MachineInfo is a fake implementation of Interface.MachineInfo.
func (c *Fake) MachineInfo() (*cadvisorapi.MachineInfo, error) {
	// Simulate a machine with 1 core and 3.75GB of memory.
	// We set it to non-zero values to make non-zero-capacity machines in Kubemark.
	return &cadvisorapi.MachineInfo{
		NumCores:       fakeNumCores,
		InstanceID:     cadvisorapi.InstanceID(c.NodeName),
		MemoryCapacity: fakeMemoryCapacity,
	}, nil
}

// VersionInfo is a fake implementation of Interface.VersionInfo.
func (c *Fake) VersionInfo() (*cadvisorapi.VersionInfo, error) {
	return &cadvisorapi.VersionInfo{
		KernelVersion:      FakeKernelVersion,
		ContainerOsVersion: FakeContainerOSVersion,
		DockerVersion:      fakeDockerVersion,
	}, nil
}

// ImagesFsInfo is a fake implementation of Interface.ImagesFsInfo.
func (c *Fake) ImagesFsInfo(context.Context) (cadvisorapiv2.FsInfo, error) {
	return cadvisorapiv2.FsInfo{}, nil
}

// RootFsInfo is a fake implementation of Interface.RootFsInfo.
func (c *Fake) RootFsInfo() (cadvisorapiv2.FsInfo, error) {
	return cadvisorapiv2.FsInfo{}, nil
}

// ContainerFsInfo is a fake implementation of Interface.ContainerFsInfo.
func (c *Fake) ContainerFsInfo(context.Context) (cadvisorapiv2.FsInfo, error) {
	return cadvisorapiv2.FsInfo{}, nil
}

// GetDirFsInfo is a fake implementation of Interface.GetDirFsInfo.
func (c *Fake) GetDirFsInfo(path string) (cadvisorapiv2.FsInfo, error) {
	return cadvisorapiv2.FsInfo{}, nil
}
