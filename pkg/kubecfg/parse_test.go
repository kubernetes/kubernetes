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

package kubecfg

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"gopkg.in/v1/yaml"
)

func TestParseBadStorage(t *testing.T) {
	_, err := ToWireFormat([]byte("{}"), "badstorage")
	if err == nil {
		t.Errorf("Expected error, received none")
	}
}

func DoParseTest(t *testing.T, storage string, obj interface{}) {
	jsonData, _ := api.Encode(obj)
	yamlData, _ := yaml.Marshal(obj)
	t.Logf("Intermediate yaml:\n%v\n", string(yamlData))

	jsonGot, jsonErr := ToWireFormat(jsonData, storage)
	yamlGot, yamlErr := ToWireFormat(yamlData, storage)

	if jsonErr != nil {
		t.Errorf("json err: %#v", jsonErr)
	}
	if yamlErr != nil {
		t.Errorf("yaml err: %#v", yamlErr)
	}
	if string(jsonGot) != string(jsonData) {
		t.Errorf("json output didn't match:\nGot:\n%v\n\nWanted:\n%v\n",
			string(jsonGot), string(jsonData))
	}
	if string(yamlGot) != string(jsonData) {
		t.Errorf("yaml parsed output didn't match:\nGot:\n%v\n\nWanted:\n%v\n",
			string(yamlGot), string(jsonData))
	}
}

func TestParsePod(t *testing.T) {
	DoParseTest(t, "pods", api.Pod{
		JSONBase: api.JSONBase{ID: "test pod"},
		DesiredState: api.PodState{
			Manifest: api.ContainerManifest{
				ID: "My manifest",
				Containers: []api.Container{
					{Name: "my container"},
				},
				Volumes: []api.Volume{
					{Name: "volume"},
				},
			},
		},
	})
}

func TestParseService(t *testing.T) {
	DoParseTest(t, "services", api.Service{
		JSONBase: api.JSONBase{ID: "my service"},
		Port:     8080,
		Labels: map[string]string{
			"area": "staging",
		},
		Selector: map[string]string{
			"area": "staging",
		},
	})
}

func TestParseController(t *testing.T) {
	DoParseTest(t, "replicationControllers", api.ReplicationController{
		DesiredState: api.ReplicationControllerState{
			Replicas: 9001,
			PodTemplate: api.PodTemplate{
				DesiredState: api.PodState{
					Manifest: api.ContainerManifest{
						ID: "My manifest",
						Containers: []api.Container{
							{Name: "my container"},
						},
						Volumes: []api.Volume{
							{Name: "volume"},
						},
					},
				},
			},
		},
	})
}
