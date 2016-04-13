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

package sources

import (
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/heapster/sources/api"
	"k8s.io/heapster/sources/datasource"
	"k8s.io/heapster/sources/nodes"
)

type fakeDataSource struct {
	f func(host datasource.Host, start, end time.Time) (subcontainers []*api.Container, root *api.Container, err error)
}

func (self *fakeDataSource) GetAllContainers(host datasource.Host, start, end time.Time) (subcontainers []*api.Container, root *api.Container, err error) {
	return self.f(host, start, end)
}

func TestBasicSuccess(t *testing.T) {
	cadvisorApi := &fakeDataSource{
		f: func(host datasource.Host, start, end time.Time) (subcontainers []*api.Container, root *api.Container, err error) {
			return nil, nil, nil
		},
	}

	source := &cadvisorSource{
		nodesApi:     &fakeNodesApi{nodes.NodeList{}},
		cadvisorPort: 8080,
		cadvisorApi:  cadvisorApi,
	}
	data, err := source.GetInfo(time.Now(), time.Now().Add(time.Minute))
	require.NoError(t, err)
	require.Equal(t, api.AggregateData{Pods: nil, Containers: nil, Machine: nil}, data)
}

func TestWorkflowSuccess(t *testing.T) {
	type cadvisorData struct {
		subcontainers []*api.Container
		root          *api.Container
	}
	hostA := datasource.Host{IP: "1.1.1.1", Port: 8080, Resource: ""}
	hostB := datasource.Host{IP: "1.1.1.2", Port: 8080, Resource: ""}
	expectedData := map[datasource.Host]cadvisorData{
		hostA: {
			subcontainers: []*api.Container{{Name: "/a"}},
			root:          &api.Container{Name: "/"},
		},
		hostB: {
			subcontainers: []*api.Container{{Name: "/b"}},
			root:          &api.Container{Name: "/"},
		},
	}
	cadvisorApi := &fakeDataSource{
		f: func(host datasource.Host, start, end time.Time) (subcontainers []*api.Container, root *api.Container, err error) {
			data, exists := expectedData[host]
			if !exists {
				return nil, nil, fmt.Errorf("unexpected host: %+v", host)
			}
			return data.subcontainers, data.root, nil
		},
	}
	nodeList := nodes.NodeList{
		Items: map[nodes.Host]nodes.Info{
			nodes.Host("a"): {InternalIP: "1.1.1.1"},
			nodes.Host("b"): {InternalIP: "1.1.1.2"},
		},
	}
	source := &cadvisorSource{
		nodesApi:     &fakeNodesApi{nodeList},
		cadvisorPort: 8080,
		cadvisorApi:  cadvisorApi,
	}
	data, err := source.GetInfo(time.Now(), time.Now().Add(time.Minute))
	require.NoError(t, err)
	assert.Len(t, data.Containers, 2)
	assert.Len(t, data.Machine, 2)
	assert.Equal(t, data.Machine[0].Name, expectedData[hostA].root.Name)
	assert.Equal(t, data.Machine[1].Name, expectedData[hostB].root.Name)
	containerNames := []string{
		data.Containers[0].Name,
		data.Containers[1].Name,
	}
	assert.Contains(t, containerNames, expectedData[hostA].subcontainers[0].Name)
	assert.Contains(t, containerNames, expectedData[hostB].subcontainers[0].Name)
}
