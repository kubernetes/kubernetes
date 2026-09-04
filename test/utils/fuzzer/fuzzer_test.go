/*
Copyright The Kubernetes Authors.

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
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestExemplaryPodFuzzer(t *testing.T) {
	fuzzer := NewExemplaryPodFuzzer(42, "fuzzed-pod", "fuzzed-ns")

	basePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "original-pod",
			Namespace: "original-ns",
			Annotations: map[string]string{
				"sensitive-info": "secret-data",
			},
			OwnerReferences: []metav1.OwnerReference{
				{
					Name: "original-owner",
					UID:  "original-uid",
				},
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "web",
					Image: "nginx",
					Env: []v1.EnvVar{
						{Name: "DB_PASSWORD", Value: "password123"},
					},
				},
			},
		},
	}

	pod1 := fuzzer.FuzzPod(basePod, 1)
	pod2 := fuzzer.FuzzPod(basePod, 2)

	// Verify basic fields
	assert.Equal(t, "fuzzed-pod-1", pod1.Name)
	assert.Equal(t, "fuzzed-pod-2", pod2.Name)
	assert.Equal(t, "fuzzed-ns", pod1.Namespace)

	// Verify sanitization (no original names/data)
	assert.NotEqual(t, "original-pod", pod1.Name)
	assert.NotEqual(t, "secret-data", pod1.Annotations["sensitive-info"])
	assert.Contains(t, pod1.Annotations["sensitive-info"], "fuzzed-val-")

	// Verify Env Vars (Keys and Values fuzzed)
	assert.Len(t, pod1.Spec.Containers[0].Env, 1)
	assert.NotEqual(t, "DB_PASSWORD", pod1.Spec.Containers[0].Env[0].Name)
	assert.NotEqual(t, "password123", pod1.Spec.Containers[0].Env[0].Value)
	assert.Contains(t, pod1.Spec.Containers[0].Env[0].Name, "FUZZED_ENV_")

	// Verify OwnerRefs
	assert.Len(t, pod1.OwnerReferences, 1)
	assert.Contains(t, pod1.OwnerReferences[0].Name, "fuzzed-owner-")
	assert.Contains(t, string(pod1.OwnerReferences[0].UID), "fuzzed-uid-")

	// Verify stability/interning (pod1 and pod2 should share identical strings for everything except Name/UID)
	assert.Equal(t, pod1.Spec.Containers[0].Env[0].Name, pod2.Spec.Containers[0].Env[0].Name)
	assert.Equal(t, pod1.Annotations["sensitive-info"], pod2.Annotations["sensitive-info"])
}

func TestDeeplyNestedManagedFields(t *testing.T) {
	fuzzer := NewExemplaryPodFuzzer(42, "nested-test", "default")
	basePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "base",
			ManagedFields: []metav1.ManagedFieldsEntry{
				{
					Manager:    "kubelet",
					APIVersion: "v1",
					FieldsV1: &metav1.FieldsV1{
						Raw: []byte(`{"f:spec":{"f:containers":{"k:{\"name\":\"web\"}":{".":{},"f:image":{}}}}}`),
					},
				},
			},
		},
	}

	pod := fuzzer.FuzzPod(basePod, 0)
	assert.Len(t, pod.ManagedFields, 1)
	assert.Equal(t, "kubelet", pod.ManagedFields[0].Manager)

	raw := string(pod.ManagedFields[0].FieldsV1.GetRawBytes())
	// Verify it contains fuzzed field paths
	assert.Contains(t, raw, "f:fuzzed_field_")
	assert.Contains(t, raw, "k:{\\\"id\\\":")
	// Verify it preserves nesting (count {)
	assert.Equal(t, strings.Count(string(basePod.ManagedFields[0].FieldsV1.Raw), "{"), strings.Count(raw, "{"))
}

func TestWriteExemplaryPodsToDir(t *testing.T) {
	creator := NewExemplaryPodCreator(nil, 42, "test-pod", "default")
	basePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "base"},
	}

	dir, err := creator.WriteExemplaryPodsToDir(context.TODO(), basePod, 5, 0, 2, "", nil)
	require.NoError(t, err)
	defer func() {
		_ = os.RemoveAll(dir)
	}()

	files, err := os.ReadDir(dir)
	require.NoError(t, err)
	assert.Len(t, files, 5)
}
