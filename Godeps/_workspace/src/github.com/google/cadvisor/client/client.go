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

// TODO(cAdvisor): Package comment.
package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"path"
	"strings"

	info "github.com/google/cadvisor/info/v1"
)

// Client represents the base URL for a cAdvisor client.
type Client struct {
	baseUrl string
}

// NewClient returns a new client with the specified base URL.
func NewClient(url string) (*Client, error) {
	if !strings.HasSuffix(url, "/") {
		url += "/"
	}

	return &Client{
		baseUrl: fmt.Sprintf("%sapi/v1.2/", url),
	}, nil
}

// MachineInfo returns the JSON machine information for this client.
// A non-nil error result indicates a problem with obtaining
// the JSON machine information data.
func (self *Client) MachineInfo() (minfo *info.MachineInfo, err error) {
	u := self.machineInfoUrl()
	ret := new(info.MachineInfo)
	if err = self.httpGetJsonData(ret, nil, u, "machine info"); err != nil {
		return
	}
	minfo = ret
	return
}

// ContainerInfo returns the JSON container information for the specified
// container and request.
func (self *Client) ContainerInfo(name string, query *info.ContainerInfoRequest) (cinfo *info.ContainerInfo, err error) {
	u := self.containerInfoUrl(name)
	ret := new(info.ContainerInfo)
	if err = self.httpGetJsonData(ret, query, u, fmt.Sprintf("container info for %q", name)); err != nil {
		return
	}
	cinfo = ret
	return
}

// Returns the information about all subcontainers (recursive) of the specified container (including itself).
func (self *Client) SubcontainersInfo(name string, query *info.ContainerInfoRequest) ([]info.ContainerInfo, error) {
	var response []info.ContainerInfo
	url := self.subcontainersInfoUrl(name)
	err := self.httpGetJsonData(&response, query, url, fmt.Sprintf("subcontainers container info for %q", name))
	if err != nil {
		return []info.ContainerInfo{}, err

	}
	return response, nil
}

// Returns the JSON container information for the specified
// Docker container and request.
func (self *Client) DockerContainer(name string, query *info.ContainerInfoRequest) (cinfo info.ContainerInfo, err error) {
	u := self.dockerInfoUrl(name)
	ret := make(map[string]info.ContainerInfo)
	if err = self.httpGetJsonData(&ret, query, u, fmt.Sprintf("Docker container info for %q", name)); err != nil {
		return
	}
	if len(ret) != 1 {
		err = fmt.Errorf("expected to only receive 1 Docker container: %+v", ret)
		return
	}
	for _, cont := range ret {
		cinfo = cont
	}
	return
}

// Returns the JSON container information for all Docker containers.
func (self *Client) AllDockerContainers(query *info.ContainerInfoRequest) (cinfo []info.ContainerInfo, err error) {
	u := self.dockerInfoUrl("/")
	ret := make(map[string]info.ContainerInfo)
	if err = self.httpGetJsonData(&ret, query, u, "all Docker containers info"); err != nil {
		return
	}
	cinfo = make([]info.ContainerInfo, 0, len(ret))
	for _, cont := range ret {
		cinfo = append(cinfo, cont)
	}
	return
}

func (self *Client) machineInfoUrl() string {
	return self.baseUrl + path.Join("machine")
}

func (self *Client) containerInfoUrl(name string) string {
	return self.baseUrl + path.Join("containers", name)
}

func (self *Client) subcontainersInfoUrl(name string) string {
	return self.baseUrl + path.Join("subcontainers", name)
}

func (self *Client) dockerInfoUrl(name string) string {
	return self.baseUrl + path.Join("docker", name)
}

func (self *Client) httpGetJsonData(data, postData interface{}, url, infoName string) error {
	var resp *http.Response
	var err error

	if postData != nil {
		data, err := json.Marshal(postData)
		if err != nil {
			return fmt.Errorf("unable to marshal data: %v", err)
		}
		resp, err = http.Post(url, "application/json", bytes.NewBuffer(data))
	} else {
		resp, err = http.Get(url)
	}
	if err != nil {
		return fmt.Errorf("unable to get %q from %q: %v", infoName, url, err)
	}
	if resp == nil {
		return fmt.Errorf("received empty response for %q from %q", infoName, url)
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		err = fmt.Errorf("unable to read all %q from %q: %v", infoName, url, err)
		return err
	}
	if resp.StatusCode != 200 {
		return fmt.Errorf("request %q failed with error: %q", url, strings.TrimSpace(string(body)))
	}
	if err = json.Unmarshal(body, data); err != nil {
		err = fmt.Errorf("unable to unmarshal %q (Body: %q) from %q with error: %v", infoName, string(body), url, err)
		return err
	}
	return nil
}
