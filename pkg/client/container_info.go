/*
Copyright 2014 Google Inc. All rights reserved.

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

package client

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
)

type ContainerInfo interface {
	GetContainerInfo(host, name string) (interface{}, error)
}

type HTTPContainerInfo struct {
	Client *http.Client
	Port   uint
}

func (c *HTTPContainerInfo) GetContainerInfo(host, name string) (interface{}, error) {
	request, err := http.NewRequest("GET", fmt.Sprintf("http://%s:%d/containerInfo?container=%s", host, c.Port, name), nil)
	if err != nil {
		return nil, err
	}
	response, err := c.Client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	body, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return nil, err
	}
	var data interface{}
	err = json.Unmarshal(body, &data)
	return data, err
}

// Useful for testing.
type FakeContainerInfo struct {
	data interface{}
	err  error
}

func (c *FakeContainerInfo) GetContainerInfo(host, name string) (interface{}, error) {
	return c.data, c.err
}
