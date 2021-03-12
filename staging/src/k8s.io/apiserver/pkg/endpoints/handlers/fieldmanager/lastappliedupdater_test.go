/*
Copyright 2020 The Kubernetes Authors.

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

package fieldmanager

import (
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/yaml"
)

func TestLastAppliedUpdater(t *testing.T) {
	f := NewTestFieldManager(schema.FromAPIVersionAndKind("apps/v1", "Deployment"),
		false,
		func(m Manager) Manager {
			return NewLastAppliedUpdater(m)
		})

	originalLastApplied := `nonempty`
	appliedObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	appliedDeployment := []byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  annotations:
    "kubectl.kubernetes.io/last-applied-configuration": "` + originalLastApplied + `"
  labels:
    app: my-app
spec:
  replicas: 20
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-c
        image: my-image
`)
	if err := yaml.Unmarshal(appliedDeployment, &appliedObj.Object); err != nil {
		t.Errorf("error decoding YAML: %v", err)
	}

	if err := f.Apply(appliedObj, "NOT-KUBECTL", false); err != nil {
		t.Errorf("error applying object: %v", err)
	}

	lastApplied, err := getLastApplied(f.liveObj)
	if err != nil {
		t.Errorf("failed to get last applied: %v", err)
	}

	if lastApplied != originalLastApplied {
		t.Errorf("expected last applied annotation to be %q and NOT be updated, but got: %q", originalLastApplied, lastApplied)
	}

	if err := f.Apply(appliedObj, "kubectl", false); err != nil {
		t.Errorf("error applying object: %v", err)
	}

	lastApplied, err = getLastApplied(f.liveObj)
	if err != nil {
		t.Errorf("failed to get last applied: %v", err)
	}

	if lastApplied == originalLastApplied ||
		!strings.Contains(lastApplied, "my-app") ||
		!strings.Contains(lastApplied, "my-image") {
		t.Errorf("expected last applied annotation to be updated, but got: %q", lastApplied)
	}
}
