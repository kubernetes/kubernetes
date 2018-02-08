/*
Copyright 2017 The Kubernetes Authors.

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

package unstructured

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestUnstructuredList(t *testing.T) {
	list := &UnstructuredList{
		Object: map[string]interface{}{"kind": "List", "apiVersion": "v1"},
		Items: []Unstructured{
			{Object: map[string]interface{}{"kind": "Pod", "apiVersion": "v1", "metadata": map[string]interface{}{"name": "test"}}},
		},
	}
	content := list.UnstructuredContent()
	items := content["items"].([]interface{})
	require.Len(t, items, 1)
	val, found, err := NestedFieldCopy(items[0].(map[string]interface{}), "metadata", "name")
	require.True(t, found)
	require.NoError(t, err)
	assert.Equal(t, "test", val)
}

func TestNilDeletionTimestamp(t *testing.T) {
	var u Unstructured
	del := u.GetDeletionTimestamp()
	if del != nil {
		t.Errorf("unexpected non-nil deletion timestamp: %v", del)
	}
	u.SetDeletionTimestamp(u.GetDeletionTimestamp())
	del = u.GetDeletionTimestamp()
	if del != nil {
		t.Errorf("unexpected non-nil deletion timestamp: %v", del)
	}
	_, ok := u.Object["metadata"]
	assert.False(t, ok)

	now := metav1.Now()
	u.SetDeletionTimestamp(&now)
	assert.Equal(t, now.Unix(), u.GetDeletionTimestamp().Unix())
	u.SetDeletionTimestamp(nil)
	metadata := u.Object["metadata"].(map[string]interface{})
	_, ok = metadata["deletionTimestamp"]
	assert.False(t, ok)
}

func TestEmptyCreationTimestampIsOmitted(t *testing.T) {
	var u Unstructured
	now := metav1.Now()

	// set an initial creationTimestamp and ensure the field exists
	u.SetCreationTimestamp(now)
	metadata := u.Object["metadata"].(map[string]interface{})
	creationTimestamp, exists := metadata["creationTimestamp"]
	if !exists {
		t.Fatalf("unexpected missing creationTimestamp")
	}

	// set an empty timestamp and ensure the field no longer exists
	u.SetCreationTimestamp(metav1.Time{})
	metadata = u.Object["metadata"].(map[string]interface{})
	creationTimestamp, exists = metadata["creationTimestamp"]
	if exists {
		t.Errorf("unexpected creation timestamp field: %q", creationTimestamp)
	}
}
