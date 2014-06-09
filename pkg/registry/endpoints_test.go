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

	. "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

func TestSyncEndpointsEmpty(t *testing.T) {
	serviceRegistry := MockServiceRegistry{}
	taskRegistry := MockPodRegistry{}

	endpoints := MakeEndpointController(&serviceRegistry, &taskRegistry)
	err := endpoints.SyncServiceEndpoints()
	expectNoError(t, err)
}

func TestSyncEndpointsError(t *testing.T) {
	serviceRegistry := MockServiceRegistry{
		err: fmt.Errorf("Test Error"),
	}
	taskRegistry := MockPodRegistry{}

	endpoints := MakeEndpointController(&serviceRegistry, &taskRegistry)
	err := endpoints.SyncServiceEndpoints()
	if err != serviceRegistry.err {
		t.Errorf("Errors don't match: %#v %#v", err, serviceRegistry.err)
	}
}

func TestSyncEndpointsItems(t *testing.T) {
	serviceRegistry := MockServiceRegistry{
		list: ServiceList{
			Items: []Service{
				Service{
					Labels: map[string]string{
						"foo": "bar",
					},
				},
			},
		},
	}
	taskRegistry := MockPodRegistry{
		pods: []Pod{
			Pod{
				DesiredState: PodState{
					Manifest: ContainerManifest{
						Containers: []Container{
							Container{
								Ports: []Port{
									Port{
										HostPort: 8080,
									},
								},
							},
						},
					},
				},
			},
		},
	}

	endpoints := MakeEndpointController(&serviceRegistry, &taskRegistry)
	err := endpoints.SyncServiceEndpoints()
	expectNoError(t, err)
	if len(serviceRegistry.endpoints.Endpoints) != 1 {
		t.Errorf("Unexpected endpoints update: %#v", serviceRegistry.endpoints)
	}
}

func TestSyncEndpointsTaskError(t *testing.T) {
	serviceRegistry := MockServiceRegistry{
		list: ServiceList{
			Items: []Service{
				Service{
					Labels: map[string]string{
						"foo": "bar",
					},
				},
			},
		},
	}
	taskRegistry := MockPodRegistry{
		err: fmt.Errorf("test error."),
	}

	endpoints := MakeEndpointController(&serviceRegistry, &taskRegistry)
	err := endpoints.SyncServiceEndpoints()
	if err == nil {
		t.Error("Unexpected non-error")
	}
}
