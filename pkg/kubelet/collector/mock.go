// +build cgo,linux

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

package collector

import (
	"github.com/stretchr/testify/mock"
)

type Mock struct {
	mock.Mock
}

var _ Interface = new(Mock)

func (c *Mock) Start() error {
	args := c.Called()
	return args.Error(0)
}

func (c *Mock) MachineInfo() (*MachineInfo, error) {
	args := c.Called()
	return args.Get(0).(*MachineInfo), args.Error(1)
}

func (c *Mock) VersionInfo() (*VersionInfo, error) {
	args := c.Called()
	return args.Get(0).(*VersionInfo), args.Error(1)
}

func (c *Mock) FsInfo(fsLabel string) (*FsInfo, error) {
	args := c.Called(fsLabel)
	return args.Get(0).(*FsInfo), args.Error(1)
}

func (c *Mock) WatchEvents(request *Request) (chan *Event, error) {
	args := c.Called()
	return args.Get(0).(chan *Event), args.Error(1)
}

func (c *Mock) ContainerInfo(containerName string, req *ContainerInfoRequest, subcontainers bool, isRawContainer bool) (map[string]interface{}, error) {
	args := c.Called(containerName, req, subcontainers, isRawContainer)
	return args.Get(0).(map[string]interface{}), args.Error(1)
}
