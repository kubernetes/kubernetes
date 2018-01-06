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

func TestRemoveNestedField(t *testing.T) {
	obj := map[string]interface{}{
		"x": map[string]interface{}{
			"y": 1,
			"a": "foo",
		},
	}
	RemoveNestedField(obj, "x", "a")
	assert.Len(t, obj["x"], 1)
	RemoveNestedField(obj, "x", "y")
	assert.Empty(t, obj["x"])
	RemoveNestedField(obj, "x")
	assert.Empty(t, obj)
	RemoveNestedField(obj, "x") // Remove of a non-existent field
	assert.Empty(t, obj)
}
