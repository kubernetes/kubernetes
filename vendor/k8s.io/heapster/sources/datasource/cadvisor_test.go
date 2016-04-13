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

package datasource

import (
	"encoding/json"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/google/cadvisor/client"
	cadvisor_api "github.com/google/cadvisor/info/v1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/heapster/sources/api"
	"k8s.io/kubernetes/pkg/util"
)

func TestBasicCadvisor(t *testing.T) {
	response := []cadvisor_api.ContainerInfo{}
	data, err := json.Marshal(&response)
	require.NoError(t, err)
	handler := util.FakeHandler{
		StatusCode:   200,
		RequestBody:  "",
		ResponseBody: string(data),
		T:            t,
	}
	server := httptest.NewServer(&handler)
	defer server.Close()
	cadvisorClient, err := client.NewClient(server.URL)
	require.NoError(t, err)
	cadvisorSource := &cadvisorSource{}
	subcontainer, root, err := cadvisorSource.getAllContainers(cadvisorClient, time.Now(), time.Now().Add(time.Minute))
	require.NoError(t, err)
	assert.Len(t, subcontainer, 0)
	assert.Nil(t, root)
}

func TestDetailedCadvisor(t *testing.T) {
	rootContainer := api.Container{
		Name: "/",
		Spec: api.ContainerSpec{
			ContainerSpec: cadvisor_api.ContainerSpec{
				CreationTime: time.Now(),
				HasCpu:       true,
				HasMemory:    true,
			},
		},
		Stats: []*api.ContainerStats{
			{
				ContainerStats: cadvisor_api.ContainerStats{
					Timestamp: time.Now(),
				},
			},
		},
	}
	subContainers := []api.Container{
		{
			Name: "a",
			Spec: api.ContainerSpec{
				ContainerSpec: cadvisor_api.ContainerSpec{
					CreationTime: time.Now(),
					HasCpu:       true,
					HasMemory:    true,
				},
			},
			Stats: []*api.ContainerStats{
				{
					ContainerStats: cadvisor_api.ContainerStats{
						Timestamp: time.Now(),
					},
				},
			},
		},
		{
			Name: "b",
			Spec: api.ContainerSpec{
				ContainerSpec: cadvisor_api.ContainerSpec{
					CreationTime: time.Now(),
					HasCpu:       true,
					HasMemory:    true,
				},
			},
			Stats: []*api.ContainerStats{
				{
					ContainerStats: cadvisor_api.ContainerStats{
						Timestamp: time.Now(),
					},
				},
			},
		},
	}

	response := []cadvisor_api.ContainerInfo{
		{
			ContainerReference: cadvisor_api.ContainerReference{
				Name: rootContainer.Name,
			},
			Spec: rootContainer.Spec.ContainerSpec,
			Stats: []*cadvisor_api.ContainerStats{
				&rootContainer.Stats[0].ContainerStats,
			},
		},
	}
	for _, cont := range subContainers {
		response = append(response, cadvisor_api.ContainerInfo{
			ContainerReference: cadvisor_api.ContainerReference{
				Name: cont.Name,
			},
			Spec: cont.Spec.ContainerSpec,
			Stats: []*cadvisor_api.ContainerStats{
				&cont.Stats[0].ContainerStats,
			},
		})
	}
	data, err := json.Marshal(&response)
	require.NoError(t, err)
	handler := util.FakeHandler{
		StatusCode:   200,
		RequestBody:  "",
		ResponseBody: string(data),
		T:            t,
	}
	server := httptest.NewServer(&handler)
	defer server.Close()
	cadvisorClient, err := client.NewClient(server.URL)
	require.NoError(t, err)
	cadvisorSource := &cadvisorSource{}
	subcontainers, root, err := cadvisorSource.getAllContainers(cadvisorClient, time.Now(), time.Now().Add(time.Minute))
	require.NoError(t, err)
	assert.Len(t, subcontainers, len(subContainers))
	assert.NotNil(t, root)
	assert.True(t, root.Spec.Eq(&rootContainer.Spec.ContainerSpec))
	for i, stat := range root.Stats {
		assert.True(t, stat.Eq(&rootContainer.Stats[i].ContainerStats))
	}
	for i, cont := range subcontainers {
		assert.True(t, subContainers[i].Spec.Eq(&cont.Spec.ContainerSpec))
		for j, stat := range cont.Stats {
			assert.True(t, subContainers[i].Stats[j].Eq(&stat.ContainerStats))
		}
	}
}
