/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package unversioned

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"

	cadvisorApi "github.com/google/cadvisor/info/v1"
)

type ContainerInfoGetter interface {
	// GetContainerInfo returns information about a container.
	GetContainerInfo(host, podID, containerID string, req *cadvisorApi.ContainerInfoRequest) (*cadvisorApi.ContainerInfo, error)
	// GetRootInfo returns information about the root container on a machine.
	GetRootInfo(host string, req *cadvisorApi.ContainerInfoRequest) (*cadvisorApi.ContainerInfo, error)
	// GetMachineInfo returns the machine's information like number of cores, memory capacity.
	GetMachineInfo(host string) (*cadvisorApi.MachineInfo, error)
}

type HTTPContainerInfoGetter struct {
	Client *http.Client
	Port   int
}

func (self *HTTPContainerInfoGetter) GetMachineInfo(host string) (*cadvisorApi.MachineInfo, error) {
	request, err := http.NewRequest(
		"GET",
		fmt.Sprintf("http://%v/spec",
			net.JoinHostPort(host, strconv.Itoa(self.Port)),
		),
		nil,
	)
	if err != nil {
		return nil, err
	}

	response, err := self.Client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("trying to get machine spec from %v; received status %v",
			host, response.Status)
	}
	var minfo cadvisorApi.MachineInfo
	err = json.NewDecoder(response.Body).Decode(&minfo)
	if err != nil {
		return nil, err
	}
	return &minfo, nil
}

func (self *HTTPContainerInfoGetter) getContainerInfo(host, path string, req *cadvisorApi.ContainerInfoRequest) (*cadvisorApi.ContainerInfo, error) {
	var body io.Reader
	if req != nil {
		content, err := json.Marshal(req)
		if err != nil {
			return nil, err
		}
		body = bytes.NewBuffer(content)
	}

	request, err := http.NewRequest(
		"GET",
		fmt.Sprintf("http://%v/stats/%v",
			net.JoinHostPort(host, strconv.Itoa(self.Port)),
			path,
		),
		body,
	)
	if err != nil {
		return nil, err
	}

	response, err := self.Client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("trying to get info for %v from %v; received status %v",
			path, host, response.Status)
	}
	var cinfo cadvisorApi.ContainerInfo
	err = json.NewDecoder(response.Body).Decode(&cinfo)
	if err != nil {
		return nil, err
	}
	return &cinfo, nil
}

func (self *HTTPContainerInfoGetter) GetContainerInfo(host, podID, containerID string, req *cadvisorApi.ContainerInfoRequest) (*cadvisorApi.ContainerInfo, error) {
	return self.getContainerInfo(
		host,
		fmt.Sprintf("%v/%v", podID, containerID),
		req,
	)
}

func (self *HTTPContainerInfoGetter) GetRootInfo(host string, req *cadvisorApi.ContainerInfoRequest) (*cadvisorApi.ContainerInfo, error) {
	return self.getContainerInfo(host, "", req)
}
