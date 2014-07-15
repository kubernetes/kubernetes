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

package registry

import (
	"fmt"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func TestFindPort(t *testing.T) {
	manifest := api.ContainerManifest{
		Containers: []api.Container{
			{
				Ports: []api.Port{
					{
						Name:          "foo",
						ContainerPort: 8080,
						HostPort:      9090,
					},
					{
						Name:          "bar",
						ContainerPort: 8000,
						HostPort:      9000,
					},
				},
			},
		},
	}
	port, err := findPort(&manifest, util.IntOrString{Kind: util.IntstrString, StrVal: "foo"})
	expectNoError(t, err)
	if port != 8080 {
		t.Errorf("Expected 8080, Got %d", port)
	}
	port, err = findPort(&manifest, util.IntOrString{Kind: util.IntstrString, StrVal: "bar"})
	expectNoError(t, err)
	if port != 8000 {
		t.Errorf("Expected 8000, Got %d", port)
	}
	port, err = findPort(&manifest, util.IntOrString{Kind: util.IntstrInt, IntVal: 8000})
	if port != 8000 {
		t.Errorf("Expected 8000, Got %d", port)
	}
	port, err = findPort(&manifest, util.IntOrString{Kind: util.IntstrInt, IntVal: 7000})
	if port != 7000 {
		t.Errorf("Expected 7000, Got %d", port)
	}
	port, err = findPort(&manifest, util.IntOrString{Kind: util.IntstrString, StrVal: "baz"})
	if err == nil {
		t.Error("unexpected non-error")
	}
	port, err = findPort(&manifest, util.IntOrString{Kind: util.IntstrString, StrVal: ""})
	expectNoError(t, err)
	if port != 8080 {
		t.Errorf("Expected 8080, Got %d", port)
	}
	port, err = findPort(&manifest, util.IntOrString{})
	expectNoError(t, err)
	if port != 8080 {
		t.Errorf("Expected 8080, Got %d", port)
	}
}

func TestSyncEndpointsEmpty(t *testing.T) {
	serviceRegistry := MockServiceRegistry{}
	podRegistry := MockPodRegistry{}

	endpoints := MakeEndpointController(&serviceRegistry, &podRegistry)
	err := endpoints.SyncServiceEndpoints()
	expectNoError(t, err)
}

func TestSyncEndpointsError(t *testing.T) {
	serviceRegistry := MockServiceRegistry{
		err: fmt.Errorf("test error"),
	}
	podRegistry := MockPodRegistry{}

	endpoints := MakeEndpointController(&serviceRegistry, &podRegistry)
	err := endpoints.SyncServiceEndpoints()
	if err != serviceRegistry.err {
		t.Errorf("Errors don't match: %#v %#v", err, serviceRegistry.err)
	}
}

func TestSyncEndpointsItems(t *testing.T) {
	serviceRegistry := MockServiceRegistry{
		list: api.ServiceList{
			Items: []api.Service{
				{
					Selector: map[string]string{
						"foo": "bar",
					},
				},
			},
		},
	}
	podRegistry := MockPodRegistry{
		pods: []api.Pod{
			{
				DesiredState: api.PodState{
					Manifest: api.ContainerManifest{
						Containers: []api.Container{
							{
								Ports: []api.Port{
									{
										HostPort: 8080,
									},
								},
							},
						},
					},
				},
				Labels: map[string]string{
					"foo": "bar",
				},
			},
		},
	}

	endpoints := MakeEndpointController(&serviceRegistry, &podRegistry)
	err := endpoints.SyncServiceEndpoints()
	expectNoError(t, err)
	if len(serviceRegistry.endpoints.Endpoints) != 1 {
		t.Errorf("Unexpected endpoints update: %#v", serviceRegistry.endpoints)
	}
}

func TestSyncEndpointsPodError(t *testing.T) {
	serviceRegistry := MockServiceRegistry{
		list: api.ServiceList{
			Items: []api.Service{
				{
					Selector: map[string]string{
						"foo": "bar",
					},
				},
			},
		},
	}
	podRegistry := MockPodRegistry{
		err: fmt.Errorf("test error."),
	}

	endpoints := MakeEndpointController(&serviceRegistry, &podRegistry)
	err := endpoints.SyncServiceEndpoints()
	if err == nil {
		t.Error("Unexpected non-error")
	}
}
