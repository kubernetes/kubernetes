/*
Copyright 2015 Google Inc. All rights reserved.

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
	cadvisorApi "github.com/google/cadvisor/info/v1"
	cadvisorApi2 "github.com/google/cadvisor/info/v2"
)

// Fake cAdvisor implementation.
type Fake struct {
}

var _ Interface = new(Fake)

func (c *Fake) ContainerInfo(name string, req *cadvisorApi.ContainerInfoRequest) (*cadvisorApi.ContainerInfo, error) {
	return new(cadvisorApi.ContainerInfo), nil
}

func (c *Fake) DockerContainer(name string, req *cadvisorApi.ContainerInfoRequest) (cadvisorApi.ContainerInfo, error) {
	return cadvisorApi.ContainerInfo{}, nil
}

func (c *Fake) MachineInfo() (*cadvisorApi.MachineInfo, error) {
	return new(cadvisorApi.MachineInfo), nil
}

func (c *Fake) DockerImagesFsInfo() (cadvisorApi2.FsInfo, error) {
	return cadvisorApi2.FsInfo{}, nil
}
