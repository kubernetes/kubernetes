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
	"io/ioutil"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestCodecOfUnstructuredList tests that there are no data races in Encode().
// i.e. that it does not mutate the object being encoded.
func TestCodecOfUnstructuredList(t *testing.T) {
	var wg sync.WaitGroup
	concurrency := 10
	list := UnstructuredList{
		Object: map[string]interface{}{},
	}
	wg.Add(concurrency)
	for i := 0; i < concurrency; i++ {
		go func() {
			defer wg.Done()
			assert.NoError(t, UnstructuredJSONScheme.Encode(&list, ioutil.Discard))
		}()
	}
	wg.Wait()
}

func TestUnstructuredList(t *testing.T) {
	list := &UnstructuredList{
		Object: map[string]interface{}{"kind": "List", "apiVersion": "v1"},
		Items: []Unstructured{
			{Object: map[string]interface{}{"kind": "Pod", "apiVersion": "v1", "metadata": map[string]interface{}{"name": "test"}}},
		},
	}
	content := list.UnstructuredContent()
	items := content["items"].([]interface{})
	if len(items) != 1 {
		t.Fatalf("unexpected items: %#v", items)
	}
	if getNestedField(items[0].(map[string]interface{}), "metadata", "name") != "test" {
		t.Fatalf("unexpected fields: %#v", items[0])
	}
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
	metadata := u.Object["metadata"].(map[string]interface{})
	deletionTimestamp := metadata["deletionTimestamp"]
	if deletionTimestamp != nil {
		t.Errorf("unexpected deletion timestamp field: %q", deletionTimestamp)
	}
}
