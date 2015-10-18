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
	cadvisorapi "github.com/google/cadvisor/info/v1"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
)

// Fake cAdvisor implementation.
type Fake struct {
}

var _ Interface = new(Fake)

func (c *Fake) Start() error {
	return nil
}

func (c *Fake) ContainerInfo(name string, req *cadvisorapi.ContainerInfoRequest) (*cadvisorapi.ContainerInfo, error) {
	return new(cadvisorapi.ContainerInfo), nil
}

func (c *Fake) SubcontainerInfo(name string, req *cadvisorapi.ContainerInfoRequest) (map[string]*cadvisorapi.ContainerInfo, error) {
	return map[string]*cadvisorapi.ContainerInfo{}, nil
}

func (c *Fake) DockerContainer(name string, req *cadvisorapi.ContainerInfoRequest) (cadvisorapi.ContainerInfo, error) {
	return cadvisorapi.ContainerInfo{}, nil
}

func (c *Fake) MachineInfo() (*cadvisorapi.MachineInfo, error) {
	return new(cadvisorapi.MachineInfo), nil
}

func (c *Fake) VersionInfo() (*cadvisorapi.VersionInfo, error) {
	return new(cadvisorapi.VersionInfo), nil
}

func (c *Fake) DockerImagesFsInfo() (cadvisorapiv2.FsInfo, error) {
	return cadvisorapiv2.FsInfo{}, nil
}

func (c *Fake) RootFsInfo() (cadvisorapiv2.FsInfo, error) {
	return cadvisorapiv2.FsInfo{}, nil
}

func (c *Fake) WatchEvents(request *events.Request) (*events.EventChannel, error) {
	return new(events.EventChannel), nil
}
