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
	"math/rand"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	apitesting "github.com/GoogleCloudPlatform/kubernetes/pkg/api/testing"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

func LoadSchemaForTest(file string) (Schema, error) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, err
	}
	return NewSwaggerSchemaFromBytes(data)
}

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

	seed := rand.Int63()
	apiObjectFuzzer := apitesting.FuzzerFor(nil, "", rand.NewSource(seed))
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
