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
	"github.com/google/cadvisor/events"
	cadvisorapi "github.com/google/cadvisor/info/v1"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
)

// Fake cAdvisor implementation.
type Fake struct {
	NodeName string
}

const (
	FakeNumCores           = 1
	FakeMemoryCapacity     = 4026531840
	FakeKernelVersion      = "3.16.0-0.bpo.4-amd64"
	FakeContainerOsVersion = "Debian GNU/Linux 7 (wheezy)"
	FakeDockerVersion      = "1.13.1"
)

var _ cadvisor.Interface = new(Fake)

func (c *Fake) Start() error {
	return nil
}

func (c *Fake) ContainerInfo(name string, req *cadvisorapi.ContainerInfoRequest) (*cadvisorapi.ContainerInfo, error) {
	return new(cadvisorapi.ContainerInfo), nil
}

func (c *Fake) ContainerInfoV2(name string, options cadvisorapiv2.RequestOptions) (map[string]cadvisorapiv2.ContainerInfo, error) {
	return map[string]cadvisorapiv2.ContainerInfo{}, nil
}

func (c *Fake) SubcontainerInfo(name string, req *cadvisorapi.ContainerInfoRequest) (map[string]*cadvisorapi.ContainerInfo, error) {
	return map[string]*cadvisorapi.ContainerInfo{}, nil
}

func (c *Fake) DockerContainer(name string, req *cadvisorapi.ContainerInfoRequest) (cadvisorapi.ContainerInfo, error) {
	return cadvisorapi.ContainerInfo{}, nil
}

func (c *Fake) MachineInfo() (*cadvisorapi.MachineInfo, error) {
	// Simulate a machine with 1 core and 3.75GB of memory.
	// We set it to non-zero values to make non-zero-capacity machines in Kubemark.
	return &cadvisorapi.MachineInfo{
		NumCores:       FakeNumCores,
		InstanceID:     cadvisorapi.InstanceID(c.NodeName),
		MemoryCapacity: FakeMemoryCapacity,
	}, nil
}

func (c *Fake) VersionInfo() (*cadvisorapi.VersionInfo, error) {
	return &cadvisorapi.VersionInfo{
		KernelVersion:      FakeKernelVersion,
		ContainerOsVersion: FakeContainerOsVersion,
		DockerVersion:      FakeDockerVersion,
	}, nil
}

func (c *Fake) ImagesFsInfo() (cadvisorapiv2.FsInfo, error) {
	return cadvisorapiv2.FsInfo{}, nil
}

func (c *Fake) RootFsInfo() (cadvisorapiv2.FsInfo, error) {
	return cadvisorapiv2.FsInfo{}, nil
}

func (c *Fake) WatchEvents(request *events.Request) (*events.EventChannel, error) {
	return new(events.EventChannel), nil
}

func (c *Fake) GetDirFsInfo(path string) (cadvisorapiv2.FsInfo, error) {
	return cadvisorapiv2.FsInfo{}, nil
}
