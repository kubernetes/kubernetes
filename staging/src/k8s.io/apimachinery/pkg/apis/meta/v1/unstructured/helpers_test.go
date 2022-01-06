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

	runtimetesting "k8s.io/apimachinery/pkg/runtime/testing"

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

func TestNestedFieldNoCopy(t *testing.T) {
	target := map[string]interface{}{"foo": "bar"}

	obj := map[string]interface{}{
		"a": map[string]interface{}{
			"b": target,
			"c": nil,
			"d": []interface{}{"foo"},
			"e": []interface{}{
				map[string]interface{}{
					"f": "bar",
				},
			},
		},
	}

	// case 1: field exists and is non-nil
	res, exists, err := NestedFieldNoCopy(obj, "a", "b")
	assert.True(t, exists)
	assert.NoError(t, err)
	assert.Equal(t, target, res)
	target["foo"] = "baz"
	assert.Equal(t, target["foo"], res.(map[string]interface{})["foo"], "result should be a reference to the expected item")

	// case 2: field exists and is nil
	res, exists, err = NestedFieldNoCopy(obj, "a", "c")
	assert.True(t, exists)
	assert.NoError(t, err)
	assert.Nil(t, res)

	// case 3: error traversing obj
	res, exists, err = NestedFieldNoCopy(obj, "a", "d", "foo")
	assert.False(t, exists)
	assert.Error(t, err)
	assert.Nil(t, res)

	// case 4: field does not exist
	res, exists, err = NestedFieldNoCopy(obj, "a", "g")
	assert.False(t, exists)
	assert.NoError(t, err)
	assert.Nil(t, res)

	// case 5: intermediate field does not exist
	res, exists, err = NestedFieldNoCopy(obj, "a", "g", "f")
	assert.False(t, exists)
	assert.NoError(t, err)
	assert.Nil(t, res)

	// case 6: intermediate field is null
	//         (background: happens easily in YAML)
	res, exists, err = NestedFieldNoCopy(obj, "a", "c", "f")
	assert.False(t, exists)
	assert.NoError(t, err)
	assert.Nil(t, res)

	// case 7: array/slice syntax is not supported
	//         (background: users may expect this to be supported)
	res, exists, err = NestedFieldNoCopy(obj, "a", "e[0]")
	assert.False(t, exists)
	assert.NoError(t, err)
	assert.Nil(t, res)
}

func TestNestedFieldCopy(t *testing.T) {
	target := map[string]interface{}{"foo": "bar"}

	obj := map[string]interface{}{
		"a": map[string]interface{}{
			"b": target,
			"c": nil,
			"d": []interface{}{"foo"},
		},
	}

	// case 1: field exists and is non-nil
	res, exists, err := NestedFieldCopy(obj, "a", "b")
	assert.True(t, exists)
	assert.NoError(t, err)
	assert.Equal(t, target, res)
	target["foo"] = "baz"
	assert.NotEqual(t, target["foo"], res.(map[string]interface{})["foo"], "result should be a copy of the expected item")

	// case 2: field exists and is nil
	res, exists, err = NestedFieldCopy(obj, "a", "c")
	assert.True(t, exists)
	assert.NoError(t, err)
	assert.Nil(t, res)

	// case 3: error traversing obj
	res, exists, err = NestedFieldCopy(obj, "a", "d", "foo")
	assert.False(t, exists)
	assert.Error(t, err)
	assert.Nil(t, res)

	// case 4: field does not exist
	res, exists, err = NestedFieldCopy(obj, "a", "e")
	assert.False(t, exists)
	assert.NoError(t, err)
	assert.Nil(t, res)
}

func TestCacheableObject(t *testing.T) {
	runtimetesting.CacheableObjectTest(t, UnstructuredJSONScheme)
}

func TestSetNestedStringSlice(t *testing.T) {
	obj := map[string]interface{}{
		"x": map[string]interface{}{
			"y": 1,
			"a": "foo",
		},
	}

	err := SetNestedStringSlice(obj, []string{"bar"}, "x", "z")
	assert.NoError(t, err)
	assert.Len(t, obj["x"], 3)
	assert.Len(t, obj["x"].(map[string]interface{})["z"], 1)
	assert.Equal(t, obj["x"].(map[string]interface{})["z"].([]interface{})[0], "bar")
}

func TestSetNestedSlice(t *testing.T) {
	obj := map[string]interface{}{
		"x": map[string]interface{}{
			"y": 1,
			"a": "foo",
		},
	}

	err := SetNestedSlice(obj, []interface{}{"bar"}, "x", "z")
	assert.NoError(t, err)
	assert.Len(t, obj["x"], 3)
	assert.Len(t, obj["x"].(map[string]interface{})["z"], 1)
	assert.Equal(t, obj["x"].(map[string]interface{})["z"].([]interface{})[0], "bar")
}

func TestSetNestedStringMap(t *testing.T) {
	obj := map[string]interface{}{
		"x": map[string]interface{}{
			"y": 1,
			"a": "foo",
		},
	}

	err := SetNestedStringMap(obj, map[string]string{"b": "bar"}, "x", "z")
	assert.NoError(t, err)
	assert.Len(t, obj["x"], 3)
	assert.Len(t, obj["x"].(map[string]interface{})["z"], 1)
	assert.Equal(t, obj["x"].(map[string]interface{})["z"].(map[string]interface{})["b"], "bar")
}

func TestSetNestedMap(t *testing.T) {
	obj := map[string]interface{}{
		"x": map[string]interface{}{
			"y": 1,
			"a": "foo",
		},
	}

	err := SetNestedMap(obj, map[string]interface{}{"b": "bar"}, "x", "z")
	assert.NoError(t, err)
	assert.Len(t, obj["x"], 3)
	assert.Len(t, obj["x"].(map[string]interface{})["z"], 1)
	assert.Equal(t, obj["x"].(map[string]interface{})["z"].(map[string]interface{})["b"], "bar")
}
