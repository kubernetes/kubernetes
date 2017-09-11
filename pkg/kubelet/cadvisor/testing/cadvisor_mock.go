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
	"github.com/stretchr/testify/mock"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
)

type Mock struct {
	mock.Mock
}

var _ cadvisor.Interface = new(Mock)

func (c *Mock) Start() error {
	args := c.Called()
	return args.Error(0)
}

// ContainerInfo is a mock implementation of Interface.ContainerInfo.
func (c *Mock) ContainerInfo(name string, req *cadvisorapi.ContainerInfoRequest) (*cadvisorapi.ContainerInfo, error) {
	args := c.Called(name, req)
	return args.Get(0).(*cadvisorapi.ContainerInfo), args.Error(1)
}

// ContainerInfoV2 is a mock implementation of Interface.ContainerInfoV2.
func (c *Mock) ContainerInfoV2(name string, options cadvisorapiv2.RequestOptions) (map[string]cadvisorapiv2.ContainerInfo, error) {
	args := c.Called(name, options)
	return args.Get(0).(map[string]cadvisorapiv2.ContainerInfo), args.Error(1)
}

func (c *Mock) SubcontainerInfo(name string, req *cadvisorapi.ContainerInfoRequest) (map[string]*cadvisorapi.ContainerInfo, error) {
	args := c.Called(name, req)
	return args.Get(0).(map[string]*cadvisorapi.ContainerInfo), args.Error(1)
}

// DockerContainer is a mock implementation of Interface.DockerContainer.
func (c *Mock) DockerContainer(name string, req *cadvisorapi.ContainerInfoRequest) (cadvisorapi.ContainerInfo, error) {
	args := c.Called(name, req)
	return args.Get(0).(cadvisorapi.ContainerInfo), args.Error(1)
}

// MachineInfo is a mock implementation of Interface.MachineInfo.
func (c *Mock) MachineInfo() (*cadvisorapi.MachineInfo, error) {
	args := c.Called()
	return args.Get(0).(*cadvisorapi.MachineInfo), args.Error(1)
}

func (c *Mock) VersionInfo() (*cadvisorapi.VersionInfo, error) {
	args := c.Called()
	return args.Get(0).(*cadvisorapi.VersionInfo), args.Error(1)
}

func (c *Mock) ImagesFsInfo() (cadvisorapiv2.FsInfo, error) {
	args := c.Called()
	return args.Get(0).(cadvisorapiv2.FsInfo), args.Error(1)
}

func (c *Mock) RootFsInfo() (cadvisorapiv2.FsInfo, error) {
	args := c.Called()
	return args.Get(0).(cadvisorapiv2.FsInfo), args.Error(1)
}

func (c *Mock) WatchEvents(request *events.Request) (*events.EventChannel, error) {
	args := c.Called()
	return args.Get(0).(*events.EventChannel), args.Error(1)
}

func (c *Mock) HasDedicatedImageFs() (bool, error) {
	args := c.Called()
	return args.Get(0).(bool), args.Error(1)
}

func (c *Mock) GetFsInfoByFsUUID(uuid string) (cadvisorapiv2.FsInfo, error) {
	args := c.Called(uuid)
	return args.Get(0).(cadvisorapiv2.FsInfo), args.Error(1)
}
