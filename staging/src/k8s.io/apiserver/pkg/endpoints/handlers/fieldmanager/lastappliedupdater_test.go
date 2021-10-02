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
	"fmt"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/yaml"
)

func TestLastAppliedUpdater(t *testing.T) {
	f := NewTestFieldManager(schema.FromAPIVersionAndKind("apps/v1", "Deployment"),
		"",
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

func TestLargeLastApplied(t *testing.T) {
	tests := []struct {
		name      string
		oldObject *corev1.ConfigMap
		newObject *corev1.ConfigMap
	}{
		{
			name: "old object + new object last-applied annotation is too big",
			oldObject: &corev1.ConfigMap{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "v1",
					Kind:       "ConfigMap",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "large-update-test-cm",
					Namespace: "default",
					Annotations: map[string]string{
						corev1.LastAppliedConfigAnnotation: "nonempty",
					},
				},
				Data: map[string]string{"k": "v"},
			},
			newObject: func() *corev1.ConfigMap {
				cfg := &corev1.ConfigMap{
					TypeMeta: metav1.TypeMeta{
						APIVersion: "v1",
						Kind:       "ConfigMap",
					},
					ObjectMeta: metav1.ObjectMeta{
						Name:      "large-update-test-cm",
						Namespace: "default",
						Annotations: map[string]string{
							corev1.LastAppliedConfigAnnotation: "nonempty",
						},
					},
					Data: map[string]string{"k": "v"},
				}
				for i := 0; i < 9999; i++ {
					unique := fmt.Sprintf("this-key-is-very-long-so-as-to-create-a-very-large-serialized-fieldset-%v", i)
					cfg.Data[unique] = "A"
				}
				return cfg
			}(),
		},
		{
			name: "old object + new object annotations + new object last-applied annotation is too big",
			oldObject: func() *corev1.ConfigMap {
				cfg := &corev1.ConfigMap{
					TypeMeta: metav1.TypeMeta{
						APIVersion: "v1",
						Kind:       "ConfigMap",
					},
					ObjectMeta: metav1.ObjectMeta{
						Name:      "large-update-test-cm",
						Namespace: "default",
						Annotations: map[string]string{
							corev1.LastAppliedConfigAnnotation: "nonempty",
						},
					},
					Data: map[string]string{"k": "v"},
				}
				for i := 0; i < 2000; i++ {
					unique := fmt.Sprintf("this-key-is-very-long-so-as-to-create-a-very-large-serialized-fieldset-%v", i)
					cfg.Data[unique] = "A"
				}
				return cfg
			}(),
			newObject: func() *corev1.ConfigMap {
				cfg := &corev1.ConfigMap{
					TypeMeta: metav1.TypeMeta{
						APIVersion: "v1",
						Kind:       "ConfigMap",
					},
					ObjectMeta: metav1.ObjectMeta{
						Name:      "large-update-test-cm",
						Namespace: "default",
						Annotations: map[string]string{
							corev1.LastAppliedConfigAnnotation: "nonempty",
						},
					},
					Data: map[string]string{"k": "v"},
				}
				for i := 0; i < 2000; i++ {
					unique := fmt.Sprintf("this-key-is-very-long-so-as-to-create-a-very-large-serialized-fieldset-%v", i)
					cfg.Data[unique] = "A"
					cfg.ObjectMeta.Annotations[unique] = "A"
				}
				return cfg
			}(),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			f := NewTestFieldManager(schema.FromAPIVersionAndKind("v1", "ConfigMap"),
				"",
				func(m Manager) Manager {
					return NewLastAppliedUpdater(m)
				})

			if err := f.Apply(test.oldObject, "kubectl", false); err != nil {
				t.Errorf("Error applying object: %v", err)
			}

			lastApplied, err := getLastApplied(f.liveObj)
			if err != nil {
				t.Errorf("Failed to access last applied annotation: %v", err)
			}
			if len(lastApplied) == 0 || lastApplied == "nonempty" {
				t.Errorf("Expected an updated last-applied annotation, but got: %q", lastApplied)
			}

			if err := f.Apply(test.newObject, "kubectl", false); err != nil {
				t.Errorf("Error applying object: %v", err)
			}

			accessor := meta.NewAccessor()
			annotations, err := accessor.Annotations(f.liveObj)
			if err != nil {
				t.Errorf("Failed to access annotations: %v", err)
			}
			if annotations == nil {
				t.Errorf("No annotations on obj: %v", f.liveObj)
			}
			lastApplied, ok := annotations[corev1.LastAppliedConfigAnnotation]
			if ok || len(lastApplied) > 0 {
				t.Errorf("Expected no last applied annotation, but got last applied with length: %d", len(lastApplied))
			}
		})
	}
}
