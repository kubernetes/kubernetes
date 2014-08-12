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

package etcd

import (
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func TestMakeManifestNoServices(t *testing.T) {
	registry := registrytest.ServiceRegistry{}
	factory := &BasicManifestFactory{
		serviceRegistry: &registry,
	}

	manifest, err := factory.MakeManifest("machine", api.Pod{
		JSONBase: api.JSONBase{ID: "foobar"},
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
		t.Errorf("unexpected error: %v", err)
	}

	container := manifest.Containers[0]
	if len(container.Env) != 1 ||
		container.Env[0].Name != "SERVICE_HOST" ||
		container.Env[0].Value != "machine" {
		t.Errorf("Expected one env vars, got: %#v", manifest)
	}
	if manifest.ID != "foobar" {
		t.Errorf("Failed to assign ID to manifest: %#v", manifest.ID)
	}
}

func TestMakeManifestServices(t *testing.T) {
	registry := registrytest.ServiceRegistry{
		List: api.ServiceList{
			Items: []api.Service{
				{
					JSONBase: api.JSONBase{ID: "test"},
					Port:     8080,
					ContainerPort: util.IntOrString{
						Kind:   util.IntstrInt,
						IntVal: 900,
					},
				},
			},
		},
	}
	factory := &BasicManifestFactory{
		serviceRegistry: &registry,
	}

	manifest, err := factory.MakeManifest("machine", api.Pod{
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
		t.Errorf("unexpected error: %v", err)
	}

	container := manifest.Containers[0]
	envs := []api.EnvVar{
		{
			Name:  "TEST_SERVICE_PORT",
			Value: "8080",
		},
		{
			Name:  "TEST_PORT",
			Value: "tcp://machine:8080",
		},
		{
			Name:  "TEST_PORT_900_TCP",
			Value: "tcp://machine:8080",
		},
		{
			Name:  "TEST_PORT_900_TCP_PROTO",
			Value: "tcp",
		},
		{
			Name:  "TEST_PORT_900_TCP_PORT",
			Value: "8080",
		},
		{
			Name:  "TEST_PORT_900_TCP_ADDR",
			Value: "machine",
		},
		{
			Name:  "SERVICE_HOST",
			Value: "machine",
		},
	}
	if len(container.Env) != 7 {
		t.Errorf("Expected 7 env vars, got %d: %#v", len(container.Env), manifest)
		return
	}
	for ix := range container.Env {
		if !reflect.DeepEqual(envs[ix], container.Env[ix]) {
			t.Errorf("expected %#v, got %#v", envs[ix], container.Env[ix])
		}
	}
}

func TestMakeManifestServicesExistingEnvVar(t *testing.T) {
	registry := registrytest.ServiceRegistry{
		List: api.ServiceList{
			Items: []api.Service{
				{
					JSONBase: api.JSONBase{ID: "test"},
					Port:     8080,
					ContainerPort: util.IntOrString{
						Kind:   util.IntstrInt,
						IntVal: 900,
					},
				},
			},
		},
	}
	factory := &BasicManifestFactory{
		serviceRegistry: &registry,
	}

	manifest, err := factory.MakeManifest("machine", api.Pod{
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
		t.Errorf("unexpected error: %v", err)
	}

	container := manifest.Containers[0]

	envs := []api.EnvVar{
		{
			Name:  "foo",
			Value: "bar",
		},
		{
			Name:  "TEST_SERVICE_PORT",
			Value: "8080",
		},
		{
			Name:  "TEST_PORT",
			Value: "tcp://machine:8080",
		},
		{
			Name:  "TEST_PORT_900_TCP",
			Value: "tcp://machine:8080",
		},
		{
			Name:  "TEST_PORT_900_TCP_PROTO",
			Value: "tcp",
		},
		{
			Name:  "TEST_PORT_900_TCP_PORT",
			Value: "8080",
		},
		{
			Name:  "TEST_PORT_900_TCP_ADDR",
			Value: "machine",
		},
		{
			Name:  "SERVICE_HOST",
			Value: "machine",
		},
	}
	if len(container.Env) != 8 {
		t.Errorf("Expected 8 env vars, got: %#v", manifest)
		return
	}
	for ix := range container.Env {
		if !reflect.DeepEqual(envs[ix], container.Env[ix]) {
			t.Errorf("expected %#v, got %#v", envs[ix], container.Env[ix])
		}
	}
}
