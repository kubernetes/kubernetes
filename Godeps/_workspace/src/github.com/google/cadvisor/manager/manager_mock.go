// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package manager

import (
	"github.com/google/cadvisor/events"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/info/v2"
	"github.com/stretchr/testify/mock"
)

type ManagerMock struct {
	mock.Mock
}

func (c *ManagerMock) Start() error {
	args := c.Called()
	return args.Error(0)
}

func (c *ManagerMock) Stop() error {
	args := c.Called()
	return args.Error(0)
}

func (c *ManagerMock) GetContainerInfo(name string, query *info.ContainerInfoRequest) (*info.ContainerInfo, error) {
	args := c.Called(name, query)
	return args.Get(0).(*info.ContainerInfo), args.Error(1)
}

func (c *ManagerMock) SubcontainersInfo(containerName string, query *info.ContainerInfoRequest) ([]*info.ContainerInfo, error) {
	args := c.Called(containerName, query)
	return args.Get(0).([]*info.ContainerInfo), args.Error(1)
}

func (c *ManagerMock) AllDockerContainers(query *info.ContainerInfoRequest) (map[string]info.ContainerInfo, error) {
	args := c.Called(query)
	return args.Get(0).(map[string]info.ContainerInfo), args.Error(1)
}

func (c *ManagerMock) DockerContainer(name string, query *info.ContainerInfoRequest) (info.ContainerInfo, error) {
	args := c.Called(name, query)
	return args.Get(0).(info.ContainerInfo), args.Error(1)
}

func (c *ManagerMock) GetContainerSpec(containerName string) (info.ContainerSpec, error) {
	args := c.Called(containerName)
	return args.Get(0).(info.ContainerSpec), args.Error(1)
}

func (c *ManagerMock) GetContainerDerivedStats(containerName string) (v2.DerivedStats, error) {
	args := c.Called(containerName)
	return args.Get(0).(v2.DerivedStats), args.Error(1)
}

func (c *ManagerMock) WatchForEvents(queryuest *events.Request, passedChannel chan *events.Event) error {
	args := c.Called(queryuest, passedChannel)
	return args.Error(0)
}

func (c *ManagerMock) GetPastEvents(queryuest *events.Request) (events.EventSlice, error) {
	args := c.Called(queryuest)
	return args.Get(0).(events.EventSlice), args.Error(1)
}

func (c *ManagerMock) GetMachineInfo() (*info.MachineInfo, error) {
	args := c.Called()
	return args.Get(0).(*info.MachineInfo), args.Error(1)
}

func (c *ManagerMock) GetVersionInfo() (*info.VersionInfo, error) {
	args := c.Called()
	return args.Get(0).(*info.VersionInfo), args.Error(1)
}

func (c *ManagerMock) GetFsInfo() ([]v2.FsInfo, error) {
	args := c.Called()
	return args.Get(0).([]v2.FsInfo), args.Error(1)
}
