package cloudcfg

import (
	"encoding/json"
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
	json_data, _ := json.Marshal(obj)
	yaml_data, _ := yaml.Marshal(obj)
	t.Logf("Intermediate yaml:\n%v\n", string(yaml_data))

	json_got, json_err := ToWireFormat(json_data, storage)
	yaml_got, yaml_err := ToWireFormat(yaml_data, storage)

	if json_err != nil {
		t.Errorf("json err: %#v", json_err)
	}
	if yaml_err != nil {
		t.Errorf("yaml err: %#v", yaml_err)
	}
	if string(json_got) != string(json_data) {
		t.Errorf("json output didn't match:\nGot:\n%v\n\nWanted:\n%v\n",
			string(json_got), string(json_data))
	}
	if string(yaml_got) != string(json_data) {
		t.Errorf("yaml parsed output didn't match:\nGot:\n%v\n\nWanted:\n%v\n",
			string(yaml_got), string(json_data))
	}
}

func TestParsePod(t *testing.T) {
	DoParseTest(t, "pods", api.Pod{
		JSONBase: api.JSONBase{ID: "test pod"},
		DesiredState: api.PodState{
			Manifest: api.ContainerManifest{
				Id: "My manifest",
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
						Id: "My manifest",
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
