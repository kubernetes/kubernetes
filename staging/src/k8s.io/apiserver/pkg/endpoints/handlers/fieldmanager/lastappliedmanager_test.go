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

package fieldmanager_test

import (
	"fmt"
	"reflect"
	"testing"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager/internal"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v4/merge"
	"sigs.k8s.io/yaml"
)

// TestApplyUsingLastAppliedAnnotation tests that applying to an object
// created with the client-side apply last-applied annotation
// will not give conflicts
func TestApplyUsingLastAppliedAnnotation(t *testing.T) {
	f := NewDefaultTestFieldManager(schema.FromAPIVersionAndKind("apps/v1", "Deployment"))

	tests := []struct {
		lastApplied       []byte
		original          []byte
		applied           []byte
		fieldManager      string
		expectConflictSet *fieldpath.Set
	}{
		{
			fieldManager: "kubectl",
			lastApplied: []byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
spec:
  replicas: 3
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
        image: my-image-v1
      - name: my-c2
        image: my-image2
`),
			original: []byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  labels:
    app: my-app # missing from last-applied
spec:
  replicas: 100 # does not match last-applied
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
        image: my-image-v2 # does no match last-applied
      # note that second container in last-applied is missing
`),
			applied: []byte(`
# test conflicts due to fields not allowed by last-applied

apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  labels:
    app: my-new-label # NOT allowed: update label
spec:
  replicas: 333 # NOT allowed: update replicas
  selector:
    matchLabels:
      app: my-new-label # allowed: update label
  template:
    metadata:
      labels:
        app: my-new-label # allowed: update-label
    spec:
      containers:
      - name: my-c
        image: my-image-new # NOT allowed: update image
`),
			expectConflictSet: fieldpath.NewSet(
				fieldpath.MakePathOrDie("metadata", "labels", "app"),
				fieldpath.MakePathOrDie("spec", "replicas"),
				fieldpath.MakePathOrDie("spec", "template", "spec", "containers", fieldpath.KeyByFields("name", "my-c"), "image"),
			),
		},
		{
			fieldManager: "kubectl",
			lastApplied: []byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  labels:
    app: my-app
spec:
  replicas: 3
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
`),
			original: []byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  labels:
    app: my-app
spec:
  replicas: 100 # does not match last applied
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
`),
			applied: []byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  labels:
    app: my-new-label
spec:
  replicas: 3 # expect conflict
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-c
        image: my-image
`),
			expectConflictSet: fieldpath.NewSet(
				fieldpath.MakePathOrDie("spec", "replicas"),
			),
		},
		{
			fieldManager: "kubectl",
			original: []byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  labels:
    app: my-app
spec:
  replicas: 100
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
`),
			applied: []byte(`
# applied object matches original

apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  labels:
    app: my-app
spec:
  replicas: 100
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
`),
		},
		{
			fieldManager: "kubectl",
			original: []byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  labels:
    app: my-app
spec:
  replicas: 3
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
`),
			applied: []byte(`
# test allowed update with no conflicts

apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  labels:
    app: my-new-label # update label
spec:
  replicas: 333 # update replicas
  selector:
    matchLabels:
      app: my-new-label # update label
  template:
    metadata:
      labels:
        app: my-new-label # update-label
    spec:
      containers:
      - name: my-c
        image: my-image
`),
		},
		{
			fieldManager: "not_kubectl",
			lastApplied: []byte(`
# expect conflicts because field manager is NOT kubectl

apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  labels:
    app: my-app
spec:
  replicas: 3
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
        image: my-image-v1
`),
			original: []byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  labels:
    app: my-app
spec:
  replicas: 100 # does not match last-applied
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
        image: my-image-v2 # does no match last-applied
`),
			applied: []byte(`
# test conflicts due to fields not allowed by last-applied

apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  labels:
    app: my-new-label # update label
spec:
  replicas: 333 # update replicas
  selector:
    matchLabels:
      app: my-new-label # update label
  template:
    metadata:
      labels:
        app: my-new-label # update-label
    spec:
      containers:
      - name: my-c
        image: my-image-new # update image
`),
			expectConflictSet: fieldpath.NewSet(
				fieldpath.MakePathOrDie("metadata", "labels", "app"),
				fieldpath.MakePathOrDie("spec", "replicas"),
				fieldpath.MakePathOrDie("spec", "selector", "matchLabels", "app"),
				fieldpath.MakePathOrDie("spec", "template", "metadata", "labels", "app"),
				fieldpath.MakePathOrDie("spec", "template", "spec", "containers", fieldpath.KeyByFields("name", "my-c"), "image"),
			),
		},
		{
			fieldManager: "kubectl",
			original: []byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  labels:
    app: my-app
spec:
  replicas: 3
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
`),
			applied: []byte(`
# test allowed update with no conflicts

apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  labels:
    app: my-new-label
spec:
  replicas: 3
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
        image: my-new-image # update image
`),
		},
		{
			fieldManager: "not_kubectl",
			original: []byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  labels:
    app: my-app
spec:
  replicas: 100
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
`),
			applied: []byte(`

# expect changes to fail because field manager is not kubectl

apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  labels:
    app: my-new-label # update label
spec:
  replicas: 3 # update replicas
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
        image: my-new-image # update image
`),
			expectConflictSet: fieldpath.NewSet(
				fieldpath.MakePathOrDie("metadata", "labels", "app"),
				fieldpath.MakePathOrDie("spec", "replicas"),
				fieldpath.MakePathOrDie("spec", "template", "spec", "containers", fieldpath.KeyByFields("name", "my-c"), "image"),
			),
		},
		{
			fieldManager: "kubectl",
			original: []byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
spec:
  replicas: 3
`),
			applied: []byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
spec:
  replicas: 100 # update replicas
`),
		},
		{
			fieldManager: "kubectl",
			lastApplied: []byte(`
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  name: my-deployment
spec:
  replicas: 3
`),
			original: []byte(`
apiVersion: apps/v1 # expect conflict due to apiVersion mismatch with last-applied
kind: Deployment
metadata:
  name: my-deployment
spec:
  replicas: 3
`),
			applied: []byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
spec:
  replicas: 100 # update replicas
`),
			expectConflictSet: fieldpath.NewSet(
				fieldpath.MakePathOrDie("spec", "replicas"),
			),
		},
		{
			fieldManager: "kubectl",
			lastApplied: []byte(`
apiVerison: foo
kind: bar
spec: expect conflict due to invalid object
`),
			original: []byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
spec:
  replicas: 3
`),
			applied: []byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
spec:
  replicas: 100 # update replicas
`),
			expectConflictSet: fieldpath.NewSet(
				fieldpath.MakePathOrDie("spec", "replicas"),
			),
		},
		{
			fieldManager: "kubectl",
			// last-applied is empty
			lastApplied: []byte{},
			original: []byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
spec:
  replicas: 3
`),
			applied: []byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
spec:
  replicas: 100 # update replicas
`),
			expectConflictSet: fieldpath.NewSet(
				fieldpath.MakePathOrDie("spec", "replicas"),
			),
		},
	}

	for i, test := range tests {
		t.Run(fmt.Sprintf("test %d", i), func(t *testing.T) {
			f.Reset()

			originalObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
			if err := yaml.Unmarshal(test.original, &originalObj.Object); err != nil {
				t.Errorf("error decoding YAML: %v", err)
			}

			if test.lastApplied == nil {
				test.lastApplied = test.original
			}
			if err := setLastAppliedFromEncoded(originalObj, test.lastApplied); err != nil {
				t.Errorf("failed to set last applied: %v", err)
			}

			if err := f.Update(originalObj, "test_client_side_apply"); err != nil {
				t.Errorf("failed to apply object: %v", err)
			}

			appliedObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
			if err := yaml.Unmarshal(test.applied, &appliedObj.Object); err != nil {
				t.Errorf("error decoding YAML: %v", err)
			}

			err := f.Apply(appliedObj, test.fieldManager, false)

			if test.expectConflictSet == nil {
				if err != nil {
					t.Errorf("expected no error but got %v", err)
				}
				return
			}

			if err == nil || !apierrors.IsConflict(err) {
				t.Errorf("expected to get conflicts but got %v", err)
			}

			expectedConflicts := merge.Conflicts{}
			test.expectConflictSet.Iterate(func(p fieldpath.Path) {
				expectedConflicts = append(expectedConflicts, merge.Conflict{
					Manager: `{"manager":"test_client_side_apply","operation":"Update","apiVersion":"apps/v1"}`,
					Path:    p,
				})
			})
			expectedConflictErr := internal.NewConflictError(expectedConflicts)
			if !reflect.DeepEqual(expectedConflictErr, err) {
				t.Errorf("expected to get\n%+v\nbut got\n%+v", expectedConflictErr, err)
			}
		})
	}
}
