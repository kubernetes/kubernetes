/*
Copyright 2026 The Kubernetes Authors.

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

package fuzzer

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
)

func TestExemplaryPodFuzzer(t *testing.T) {
	fuzzer := NewExemplaryPodFuzzer(42)

	sharedSpec := &v1.PodSpec{
		Containers: []v1.Container{
			{
				Name:  "web",
				Image: "nginx",
			},
		},
	}

	template := &ExemplaryPodTemplate{
		Name:     "representative",
		BaseSpec: sharedSpec,
		ManagedFields: []ExemplaryManagedFieldTemplate{
			{
				Manager:      "kube-scheduler",
				Operation:    "Update",
				FieldsSchema: `{"f:spec":{"f:nodeName":{}}}`,
			},
		},
		Annotations: []ExemplaryAnnotationTemplate{
			{
				Key:    "fuzz.metadata/blob",
				Length: 100,
			},
		},
	}

	pod1 := fuzzer.FuzzPod(template, 1)
	pod2 := fuzzer.FuzzPod(template, 2)

	// Verify basic fields
	assert.Equal(t, "representative-1", pod1.Name)
	assert.Equal(t, "representative-2", pod2.Name)

	// Verify shared spec (interning test)
	assert.Equal(t, sharedSpec.Containers[0].Image, pod1.Spec.Containers[0].Image)
	assert.Equal(t, pod1.Spec.Containers[0].Image, pod2.Spec.Containers[0].Image)

	// Verify ManagedFields
	assert.Len(t, pod1.ManagedFields, 1)
	assert.Equal(t, "kube-scheduler", pod1.ManagedFields[0].Manager)
	assert.Equal(t, `{"f:spec":{"f:nodeName":{}}}`, string(pod1.ManagedFields[0].FieldsV1.Raw))

	// Verify Annotations
	assert.Len(t, pod1.Annotations, 1)
	assert.Contains(t, pod1.Annotations, "fuzz.metadata/blob")
	assert.Len(t, pod1.Annotations["fuzz.metadata/blob"], 100)

	// Verify stability (interning test: identical strings across pods)
	assert.Equal(t, pod1.Annotations["fuzz.metadata/blob"], pod2.Annotations["fuzz.metadata/blob"])
	assert.Equal(t, string(pod1.ManagedFields[0].FieldsV1.Raw), string(pod2.ManagedFields[0].FieldsV1.Raw))
}

func TestLoadTemplateFromFile(t *testing.T) {
	template, err := LoadTemplateFromFile("templates/representative-pod.yaml")
	assert.NoError(t, err)
	assert.Equal(t, "representative-pod", template.Name)
	assert.NotNil(t, template.BaseSpec)
	assert.Len(t, template.ManagedFields, 2)
	assert.Len(t, template.Annotations, 1)
	assert.Equal(t, 24000, template.Annotations[0].Length)
}

func TestWriteExemplaryPodsToDir(t *testing.T) {
	creator := NewExemplaryPodCreator(nil, 42)
	template := &ExemplaryPodTemplate{
		Name: "test-pod",
	}

	dir, err := creator.WriteExemplaryPodsToDir(context.TODO(), template, 5, 2, "")
	assert.NoError(t, err)
	defer os.RemoveAll(dir)

	files, err := os.ReadDir(dir)
	assert.NoError(t, err)
	assert.Len(t, files, 5)

	for i := 0; i < 5; i++ {
		expectedFile := filepath.Join(dir, fmt.Sprintf("test-pod-%d.yaml", i))
		assert.FileExists(t, expectedFile)
	}
}
