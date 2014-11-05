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

package pod

import (
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func TestMakeBoundPodNoServices(t *testing.T) {
	registry := registrytest.ServiceRegistry{}
	factory := &BasicBoundPodFactory{
		ServiceRegistry: &registry,
	}

	pod, err := factory.MakeBoundPod("machine", &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foobar"},
		DesiredState: api.PodState{
			Manifest: api.ContainerManifest{
				Containers: []api.Container{
					{
						Name: "foo",
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	container := pod.Spec.Containers[0]
	if len(container.Env) != 0 {
		t.Errorf("Expected zero env vars, got: %#v", pod)
	}
	if pod.Name != "foobar" {
		t.Errorf("Failed to assign ID to pod: %#v", pod.Name)
	}

	if _, err := api.GetReference(pod); err != nil {
		t.Errorf("Unable to get a reference to bound pod: %v", err)
	}
}

func TestMakeBoundPodServices(t *testing.T) {
	registry := registrytest.ServiceRegistry{
		List: api.ServiceList{
			Items: []api.Service{
				{
					ObjectMeta: api.ObjectMeta{Name: "test"},
					Spec: api.ServiceSpec{
						Port: 8080,
						ContainerPort: util.IntOrString{
							Kind:   util.IntstrInt,
							IntVal: 900,
						},
						PortalIP: "1.2.3.4",
					},
				},
			},
		},
	}
	factory := &BasicBoundPodFactory{
		ServiceRegistry: &registry,
	}

	pod, err := factory.MakeBoundPod("machine", &api.Pod{
		DesiredState: api.PodState{
			Manifest: api.ContainerManifest{
				Containers: []api.Container{
					{
						Name: "foo",
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	container := pod.Spec.Containers[0]
	envs := []api.EnvVar{
		{
			Name:  "TEST_SERVICE_HOST",
			Value: "1.2.3.4",
		},
		{
			Name:  "TEST_SERVICE_PORT",
			Value: "8080",
		},
		{
			Name:  "TEST_PORT",
			Value: "tcp://1.2.3.4:8080",
		},
		{
			Name:  "TEST_PORT_8080_TCP",
			Value: "tcp://1.2.3.4:8080",
		},
		{
			Name:  "TEST_PORT_8080_TCP_PROTO",
			Value: "tcp",
		},
		{
			Name:  "TEST_PORT_8080_TCP_PORT",
			Value: "8080",
		},
		{
			Name:  "TEST_PORT_8080_TCP_ADDR",
			Value: "1.2.3.4",
		},
	}
	if len(container.Env) != len(envs) {
		t.Fatalf("Expected %d env vars, got %d: %#v", len(envs), len(container.Env), pod)
	}
	for ix := range container.Env {
		if !reflect.DeepEqual(envs[ix], container.Env[ix]) {
			t.Errorf("expected %#v, got %#v", envs[ix], container.Env[ix])
		}
	}
}

func TestMakeBoundPodServicesExistingEnvVar(t *testing.T) {
	registry := registrytest.ServiceRegistry{
		List: api.ServiceList{
			Items: []api.Service{
				{
					ObjectMeta: api.ObjectMeta{Name: "test"},
					Spec: api.ServiceSpec{
						Port: 8080,
						ContainerPort: util.IntOrString{
							Kind:   util.IntstrInt,
							IntVal: 900,
						},
						PortalIP: "1.2.3.4",
					},
				},
			},
		},
	}
	factory := &BasicBoundPodFactory{
		ServiceRegistry: &registry,
	}

	pod, err := factory.MakeBoundPod("machine", &api.Pod{
		DesiredState: api.PodState{
			Manifest: api.ContainerManifest{
				Containers: []api.Container{
					{
						Env: []api.EnvVar{
							{
								Name:  "foo",
								Value: "bar",
							},
						},
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	container := pod.Spec.Containers[0]

	envs := []api.EnvVar{
		{
			Name:  "foo",
			Value: "bar",
		},
		{
			Name:  "TEST_SERVICE_HOST",
			Value: "1.2.3.4",
		},
		{
			Name:  "TEST_SERVICE_PORT",
			Value: "8080",
		},
		{
			Name:  "TEST_PORT",
			Value: "tcp://1.2.3.4:8080",
		},
		{
			Name:  "TEST_PORT_8080_TCP",
			Value: "tcp://1.2.3.4:8080",
		},
		{
			Name:  "TEST_PORT_8080_TCP_PROTO",
			Value: "tcp",
		},
		{
			Name:  "TEST_PORT_8080_TCP_PORT",
			Value: "8080",
		},
		{
			Name:  "TEST_PORT_8080_TCP_ADDR",
			Value: "1.2.3.4",
		},
	}
	if len(container.Env) != len(envs) {
		t.Fatalf("Expected %d env vars, got: %#v", len(envs), pod)
	}
	for ix := range container.Env {
		if !reflect.DeepEqual(envs[ix], container.Env[ix]) {
			t.Errorf("expected %#v, got %#v", envs[ix], container.Env[ix])
		}
	}
}
