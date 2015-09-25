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

package container

import (
	info "github.com/google/cadvisor/info/v1"
	"github.com/stretchr/testify/mock"
)

// This struct mocks a container handler.
type MockContainerHandler struct {
	mock.Mock
	Name    string
	Aliases []string
}

func NewMockContainerHandler(containerName string) *MockContainerHandler {
	return &MockContainerHandler{
		Name: containerName,
	}
}

// If self.Name is not empty, then ContainerReference() will return self.Name and self.Aliases.
// Otherwise, it will use the value provided by .On().Return().
func (self *MockContainerHandler) ContainerReference() (info.ContainerReference, error) {
	if len(self.Name) > 0 {
		var aliases []string
		if len(self.Aliases) > 0 {
			aliases = make([]string, len(self.Aliases))
			copy(aliases, self.Aliases)
		}
		return info.ContainerReference{
			Name:    self.Name,
			Aliases: aliases,
		}, nil
	}
	args := self.Called()
	return args.Get(0).(info.ContainerReference), args.Error(1)
}

func (self *MockContainerHandler) GetSpec() (info.ContainerSpec, error) {
	args := self.Called()
	return args.Get(0).(info.ContainerSpec), args.Error(1)
}

func (self *MockContainerHandler) GetStats() (*info.ContainerStats, error) {
	args := self.Called()
	return args.Get(0).(*info.ContainerStats), args.Error(1)
}

func (self *MockContainerHandler) ListContainers(listType ListType) ([]info.ContainerReference, error) {
	args := self.Called(listType)
	return args.Get(0).([]info.ContainerReference), args.Error(1)
}

func (self *MockContainerHandler) ListThreads(listType ListType) ([]int, error) {
	args := self.Called(listType)
	return args.Get(0).([]int), args.Error(1)
}

func (self *MockContainerHandler) ListProcesses(listType ListType) ([]int, error) {
	args := self.Called(listType)
	return args.Get(0).([]int), args.Error(1)
}

func (self *MockContainerHandler) WatchSubcontainers(events chan SubcontainerEvent) error {
	args := self.Called(events)
	return args.Error(0)
}

func (self *MockContainerHandler) StopWatchingSubcontainers() error {
	args := self.Called()
	return args.Error(0)
}

func (self *MockContainerHandler) Exists() bool {
	args := self.Called()
	return args.Get(0).(bool)
}

func (self *MockContainerHandler) GetCgroupPath(path string) (string, error) {
	args := self.Called(path)
	return args.Get(0).(string), args.Error(1)
}

func (self *MockContainerHandler) GetContainerLabels() map[string]string {
	args := self.Called()
	return args.Get(0).(map[string]string)
}

type FactoryForMockContainerHandler struct {
	Name                        string
	PrepareContainerHandlerFunc func(name string, handler *MockContainerHandler)
}

func (self *FactoryForMockContainerHandler) String() string {
	return self.Name
}

func (self *FactoryForMockContainerHandler) NewContainerHandler(name string, inHostNamespace bool) (ContainerHandler, error) {
	handler := &MockContainerHandler{}
	if self.PrepareContainerHandlerFunc != nil {
		self.PrepareContainerHandlerFunc(name, handler)
	}
	return handler, nil
}

func (self *FactoryForMockContainerHandler) CanHandle(name string) bool {
	return true
}
