/*
Copyright 2019 The Kubernetes Authors.

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

package container

import (
	"encoding/json"
	"testing"

	"k8s.io/api/core/v1"
)

var (
	sampleContainer = `
{
  "name": "test_container",
  "image": "foo/image:v1",
  "command": [
    "/bin/testcmd"
  ],
  "args": [
    "/bin/sh",
    "-c",
    "echo abc"
  ],
  "ports": [
    {
      "containerPort": 8001
    }
  ],
  "env": [
    {
      "name": "ENV_FOO",
      "value": "bar"
    },
    {
      "name": "ENV_BAR",
      "valueFrom": {
        "secretKeyRef": {
          "name": "foo",
          "key": "bar",
          "optional": true
        }
      }
    }
  ],
  "resources": {
    "limits": {
      "foo": "1G"
    },
    "requests": {
      "foo": "500M"
    }
  }
}
`

	sampleV131HashValue = uint64(0x8e45cbd0)
)

func TestConsistentHashContainer(t *testing.T) {
	container := &v1.Container{}
	if err := json.Unmarshal([]byte(sampleContainer), container); err != nil {
		t.Error(err)
	}

	currentHash := HashContainer(container)
	if currentHash != sampleV131HashValue {
		t.Errorf("mismatched hash value with v1.31")
	}
}
