// Copyright 2017 Google Inc. All Rights Reserved.
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

package crio

import "fmt"

type crioClientMock struct {
	info           Info
	containersInfo map[string]*ContainerInfo
	err            error
}

func (c *crioClientMock) Info() (Info, error) {
	if c.err != nil {
		return Info{}, c.err
	}
	return c.info, nil
}

func (c *crioClientMock) ContainerInfo(id string) (*ContainerInfo, error) {
	if c.err != nil {
		return nil, c.err
	}
	cInfo, ok := c.containersInfo[id]
	if !ok {
		return nil, fmt.Errorf("no container with id %s", id)
	}
	return cInfo, nil
}

func mockCrioClient(info Info, containersInfo map[string]*ContainerInfo, err error) crioClient {
	return &crioClientMock{
		err:            err,
		info:           info,
		containersInfo: containersInfo,
	}
}
