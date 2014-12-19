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

package validation

import (
	"io/ioutil"
	"strconv"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	docker "github.com/fsouza/go-dockerclient"
	fuzz "github.com/google/gofuzz"
)

func LoadSchemaForTest(file string) (Schema, error) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, err
	}
	return NewSwaggerSchemaFromBytes(data)
}

// TODO: this is cloned from serialization_test.go, refactor to somewhere common like util
// apiObjectFuzzer can randomly populate api objects.
var apiObjectFuzzer = fuzz.New().NilChance(.5).NumElements(1, 1).Funcs(
	func(j *runtime.PluginBase, c fuzz.Continue) {
		// Do nothing; this struct has only a Kind field and it must stay blank in memory.
	},
	func(j *runtime.TypeMeta, c fuzz.Continue) {
		// We have to customize the randomization of TypeMetas because their
		// APIVersion and Kind must remain blank in memory.
		j.APIVersion = ""
		j.Kind = ""
	},
	func(j *api.TypeMeta, c fuzz.Continue) {
		// We have to customize the randomization of TypeMetas because their
		// APIVersion and Kind must remain blank in memory.
		j.APIVersion = ""
		j.Kind = ""
	},
	func(j *api.ObjectMeta, c fuzz.Continue) {
		j.Name = c.RandString()
		j.ResourceVersion = strconv.FormatUint(c.RandUint64(), 10)
		j.SelfLink = c.RandString()

		var sec, nsec int64
		c.Fuzz(&sec)
		c.Fuzz(&nsec)
		j.CreationTimestamp = util.Unix(sec, nsec).Rfc3339Copy()
	},
	func(j *api.ListMeta, c fuzz.Continue) {
		j.ResourceVersion = strconv.FormatUint(c.RandUint64(), 10)
		j.SelfLink = c.RandString()
	},
	func(j *api.PodPhase, c fuzz.Continue) {
		statuses := []api.PodPhase{api.PodPending, api.PodRunning, api.PodFailed, api.PodUnknown}
		*j = statuses[c.Rand.Intn(len(statuses))]
	},
	func(j *api.ReplicationControllerSpec, c fuzz.Continue) {
		// TemplateRef must be nil for round trip
		c.Fuzz(&j.Template)
		if j.Template == nil {
			// TODO: v1beta1/2 can't round trip a nil template correctly, fix by having v1beta1/2
			// conversion compare converted object to nil via DeepEqual
			j.Template = &api.PodTemplateSpec{}
		}
		j.Template.ObjectMeta = api.ObjectMeta{Labels: j.Template.ObjectMeta.Labels}
		j.Template.Spec.NodeSelector = nil
		c.Fuzz(&j.Selector)
		j.Replicas = int(c.RandUint64())
	},
	func(j *api.ReplicationControllerStatus, c fuzz.Continue) {
		// only replicas round trips
		j.Replicas = int(c.RandUint64())
	},
	func(intstr *util.IntOrString, c fuzz.Continue) {
		// util.IntOrString will panic if its kind is set wrong.
		if c.RandBool() {
			intstr.Kind = util.IntstrInt
			intstr.IntVal = int(c.RandUint64())
			intstr.StrVal = ""
		} else {
			intstr.Kind = util.IntstrString
			intstr.IntVal = 0
			intstr.StrVal = c.RandString()
		}
	},
	func(pb map[docker.Port][]docker.PortBinding, c fuzz.Continue) {
		// This is necessary because keys with nil values get omitted.
		// TODO: Is this a bug?
		pb[docker.Port(c.RandString())] = []docker.PortBinding{
			{c.RandString(), c.RandString()},
			{c.RandString(), c.RandString()},
		}
	},
	func(pm map[string]docker.PortMapping, c fuzz.Continue) {
		// This is necessary because keys with nil values get omitted.
		// TODO: Is this a bug?
		pm[c.RandString()] = docker.PortMapping{
			c.RandString(): c.RandString(),
		}
	},
)

func TestLoad(t *testing.T) {
	_, err := LoadSchemaForTest("v1beta1-swagger.json")
	if err != nil {
		t.Errorf("Failed to load: %v", err)
	}
}

func TestValidateOk(t *testing.T) {
	schema, err := LoadSchemaForTest("v1beta1-swagger.json")
	if err != nil {
		t.Errorf("Failed to load: %v", err)
	}
	tests := []struct {
		obj      runtime.Object
		typeName string
	}{
		{obj: &api.Pod{}},
		{obj: &api.Service{}},
		{obj: &api.ReplicationController{}},
	}

	for i := 0; i < 5; i++ {
		for _, test := range tests {
			testObj := test.obj
			apiObjectFuzzer.Fuzz(testObj)
			data, err := v1beta1.Codec.Encode(testObj)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			err = schema.ValidateBytes(data)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		}
	}
}

var invalidPod = `{
  "id": "name",
  "kind": "Pod",
  "apiVersion": "v1beta1",
  "desiredState": {
    "manifest": {
      "version": "v1beta1",
      "id": "redis-master",
      "containers": [{
        "name": "master",
        "image": "dockerfile/redis",
        "command": "this is a bad command",
      }]
    }
  },
  "labels": {
    "name": "redis-master"
  }
}
`

var invalidPod2 = `{
  "apiVersion": "v1beta1",
  "kind": "Pod",
  "id": "apache-php",
  "desiredState": {
    "manifest": {
      "version": "v1beta1",
      "id": "apache-php",
      "containers": [
             {
           "name": "apache-php",
           "image": "php:5.6.2-apache",
           "ports": [{ "name": "apache", "containerPort": 80, "hostPort":"13380", "protocol":"TCP" }],
           "volumeMounts": [{"name": "shared-disk","mountPath": "/var/www/html", "readOnly": false}]
        }
           ]
    }
  },
  "labels": { "name": "apache-php" },
  "restartPolicy": {"always": {}},
  "volumes": [
    "name": "shared-disk",
    "source": {
      "GCEPersistentDisk": {
        "path": "shared-disk"
      }
    }
  ]
}
`

var invalidPod3 = `{
  "apiVersion": "v1beta1",
  "kind": "Pod",
  "id": "apache-php",
  "desiredState": {
    "manifest": {
      "version": "v1beta1",
      "id": "apache-php",
      "containers": [
             {
           "name": "apache-php",
           "image": "php:5.6.2-apache",
           "ports": [{ "name": "apache", "containerPort": 80, "hostPort":"13380", "protocol":"TCP" }],
           "volumeMounts": [{"name": "shared-disk","mountPath": "/var/www/html", "readOnly": false}]
        }
           ]
    }
  },
  "labels": { "name": "apache-php" },
  "restartPolicy": {"always": {}},
  "volumes": [
    {
      "name": "shared-disk",
      "source": {
        "GCEPersistentDisk": {
          "path": "shared-disk"
        }
      }
    }
  ]
}
`

var invalidYaml = `
id: name
kind: Pod
apiVersion: v1beta1
desiredState:
  manifest:
    version: v1beta1
    id: redis-master
    containers:
      - name: "master"
        image: "dockerfile/redis"
        command: "this is a bad command"
labels:
  name: "redis-master"
`

func TestInvalid(t *testing.T) {
	schema, err := LoadSchemaForTest("v1beta1-swagger.json")
	if err != nil {
		t.Errorf("Failed to load: %v", err)
	}
	tests := []string{invalidPod, invalidPod2, invalidPod3, invalidYaml}
	for _, test := range tests {
		err = schema.ValidateBytes([]byte(test))
		if err == nil {
			t.Errorf("unexpected non-error\n%s", test)
		}
	}
}

var validYaml = `
id: name
kind: Pod
apiVersion: v1beta1
desiredState:
  manifest:
    version: v1beta1
    id: redis-master
    containers:
      - name: "master"
        image: "dockerfile/redis"
        command:
        	- this
        	- is
        	- an
        	- ok
        	- command
labels:
  name: "redis-master"
`

func TestValid(t *testing.T) {
	schema, err := LoadSchemaForTest("v1beta1-swagger.json")
	if err != nil {
		t.Errorf("Failed to load: %v", err)
	}
	tests := []string{validYaml}
	for _, test := range tests {
		err = schema.ValidateBytes([]byte(test))
		if err == nil {
			t.Errorf("unexpected non-error\n%s", test)
		}
	}
}
