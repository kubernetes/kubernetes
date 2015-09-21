/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package cadvisor

import (
	"github.com/google/cadvisor/events"
	cadvisorApi "github.com/google/cadvisor/info/v1"
	cadvisorApiV2 "github.com/google/cadvisor/info/v2"
	"github.com/stretchr/testify/mock"
)

type Mock struct {
	mock.Mock
}

var _ Interface = new(Mock)

func (c *Mock) Start() error {
	args := c.Called()
	return args.Error(1)
}

// ContainerInfo is a mock implementation of Interface.ContainerInfo.
func (c *Mock) ContainerInfo(name string, req *cadvisorApi.ContainerInfoRequest) (*cadvisorApi.ContainerInfo, error) {
	args := c.Called(name, req)
	return args.Get(0).(*cadvisorApi.ContainerInfo), args.Error(1)
}

func (c *Mock) SubcontainerInfo(name string, req *cadvisorApi.ContainerInfoRequest) (map[string]*cadvisorApi.ContainerInfo, error) {
	args := c.Called(name, req)
	return args.Get(0).(map[string]*cadvisorApi.ContainerInfo), args.Error(1)
}

// DockerContainer is a mock implementation of Interface.DockerContainer.
func (c *Mock) DockerContainer(name string, req *cadvisorApi.ContainerInfoRequest) (cadvisorApi.ContainerInfo, error) {
	args := c.Called(name, req)
	return args.Get(0).(cadvisorApi.ContainerInfo), args.Error(1)
}

// MachineInfo is a mock implementation of Interface.MachineInfo.
func (c *Mock) MachineInfo() (*cadvisorApi.MachineInfo, error) {
	args := c.Called()
	return args.Get(0).(*cadvisorApi.MachineInfo), args.Error(1)
}

func (c *Mock) VersionInfo() (*cadvisorApi.VersionInfo, error) {
	args := c.Called()
	return args.Get(0).(*cadvisorApi.VersionInfo), args.Error(1)
}

func (c *Mock) DockerImagesFsInfo() (cadvisorApiV2.FsInfo, error) {
	args := c.Called()
	return args.Get(0).(cadvisorApiV2.FsInfo), args.Error(1)
}

func (c *Mock) RootFsInfo() (cadvisorApiV2.FsInfo, error) {
	args := c.Called()
	return args.Get(0).(cadvisorApiV2.FsInfo), args.Error(1)
}

func (c *Mock) WatchEvents(request *events.Request) (*events.EventChannel, error) {
	args := c.Called()
	return args.Get(0).(*events.EventChannel), args.Error(1)
}
