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
	"testing"

	. "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

func TestMakeManifestNoServices(t *testing.T) {
	registry := MockServiceRegistry{}
	factory := &BasicManifestFactory{
		serviceRegistry: &registry,
	}

	manifest, err := factory.MakeManifest("machine", Task{
		JSONBase: JSONBase{ID: "foobar"},
		DesiredState: TaskState{
			Manifest: ContainerManifest{
				Containers: []Container{
					Container{
						Name: "foo",
					},
				},
			},
		},
	})
	expectNoError(t, err)
	container := manifest.Containers[0]
	if len(container.Env) != 1 ||
		container.Env[0].Name != "SERVICE_HOST" ||
		container.Env[0].Value != "machine" {
		t.Errorf("Expected one env vars, got: %#v", manifest)
	}
	if manifest.Id != "foobar" {
		t.Errorf("Failed to assign id to manifest: %#v")
	}
}

func TestMakeManifestServices(t *testing.T) {
	registry := MockServiceRegistry{
		list: ServiceList{
			Items: []Service{
				Service{
					JSONBase: JSONBase{ID: "test"},
					Port:     8080,
				},
			},
		},
	}
	factory := &BasicManifestFactory{
		serviceRegistry: &registry,
	}

	manifest, err := factory.MakeManifest("machine", Task{
		DesiredState: TaskState{
			Manifest: ContainerManifest{
				Containers: []Container{
					Container{
						Name: "foo",
					},
				},
			},
		},
	})
	expectNoError(t, err)
	container := manifest.Containers[0]
	if len(container.Env) != 2 ||
		container.Env[0].Name != "TEST_SERVICE_PORT" ||
		container.Env[0].Value != "8080" ||
		container.Env[1].Name != "SERVICE_HOST" ||
		container.Env[1].Value != "machine" {
		t.Errorf("Expected 2 env vars, got: %#v", manifest)
	}
}

func TestMakeManifestServicesExistingEnvVar(t *testing.T) {
	registry := MockServiceRegistry{
		list: ServiceList{
			Items: []Service{
				Service{
					JSONBase: JSONBase{ID: "test"},
					Port:     8080,
				},
			},
		},
	}
	factory := &BasicManifestFactory{
		serviceRegistry: &registry,
	}

	manifest, err := factory.MakeManifest("machine", Task{
		DesiredState: TaskState{
			Manifest: ContainerManifest{
				Containers: []Container{
					Container{
						Env: []EnvVar{
							EnvVar{
								Name:  "foo",
								Value: "bar",
							},
						},
					},
				},
			},
		},
	})
	expectNoError(t, err)
	container := manifest.Containers[0]
	if len(container.Env) != 3 ||
		container.Env[0].Name != "foo" ||
		container.Env[0].Value != "bar" ||
		container.Env[1].Name != "TEST_SERVICE_PORT" ||
		container.Env[1].Value != "8080" ||
		container.Env[2].Name != "SERVICE_HOST" ||
		container.Env[2].Value != "machine" {
		t.Errorf("Expected no env vars, got: %#v", manifest)
	}
}
