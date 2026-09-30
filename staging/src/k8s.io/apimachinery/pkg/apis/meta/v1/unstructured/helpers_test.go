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
	"fmt"
	"io/ioutil"
	"math"
	"reflect"
	"strings"
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
	assert.Equal(t, "bar", obj["x"].(map[string]interface{})["z"].([]interface{})[0])
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
	assert.Equal(t, "bar", obj["x"].(map[string]interface{})["z"].([]interface{})[0])
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
	assert.Equal(t, "bar", obj["x"].(map[string]interface{})["z"].(map[string]interface{})["b"])
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
	assert.Equal(t, "bar", obj["x"].(map[string]interface{})["z"].(map[string]interface{})["b"])
}

func TestNestedNumberAsFloat64(t *testing.T) {
	for _, tc := range []struct {
		name           string
		obj            map[string]interface{}
		path           []string
		wantFloat64    float64
		wantBool       bool
		wantErrMessage string
	}{
		{
			name:           "not found",
			obj:            nil,
			path:           []string{"missing"},
			wantFloat64:    0,
			wantBool:       false,
			wantErrMessage: "",
		},
		{
			name:           "found float64",
			obj:            map[string]interface{}{"value": float64(42)},
			path:           []string{"value"},
			wantFloat64:    42,
			wantBool:       true,
			wantErrMessage: "",
		},
		{
			name:           "found unexpected type bool",
			obj:            map[string]interface{}{"value": true},
			path:           []string{"value"},
			wantFloat64:    0,
			wantBool:       false,
			wantErrMessage: ".value accessor error: true is of the type bool, expected float64 or int64",
		},
		{
			name:           "found int64",
			obj:            map[string]interface{}{"value": int64(42)},
			path:           []string{"value"},
			wantFloat64:    42,
			wantBool:       true,
			wantErrMessage: "",
		},
		{
			name:           "found int64 not representable as float64",
			obj:            map[string]interface{}{"value": int64(math.MaxInt64)},
			path:           []string{"value"},
			wantFloat64:    0,
			wantBool:       false,
			wantErrMessage: ".value accessor error: int64 value 9223372036854775807 cannot be losslessly converted to float64",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gotFloat64, gotBool, gotErr := NestedNumberAsFloat64(tc.obj, tc.path...)
			if gotFloat64 != tc.wantFloat64 {
				t.Errorf("got %v, wanted %v", gotFloat64, tc.wantFloat64)
			}
			if gotBool != tc.wantBool {
				t.Errorf("got %t, wanted %t", gotBool, tc.wantBool)
			}
			if tc.wantErrMessage != "" {
				if gotErr == nil {
					t.Errorf("got nil error, wanted %s", tc.wantErrMessage)
				} else if gotErrMessage := gotErr.Error(); gotErrMessage != tc.wantErrMessage {
					t.Errorf("wanted error %q, got: %v", gotErrMessage, tc.wantErrMessage)
				}
			} else if gotErr != nil {
				t.Errorf("wanted nil error, got %v", gotErr)
			}
		})
	}
}

func TestNestedNullCoercingStringMap(t *testing.T) {
	for _, tc := range []struct {
		name           string
		obj            map[string]interface{}
		path           []string
		wantObj        map[string]string
		wantFound      bool
		wantErrMessage string
	}{
		{
			name:           "missing map",
			obj:            nil,
			path:           []string{"path"},
			wantObj:        nil,
			wantFound:      false,
			wantErrMessage: "",
		},
		{
			name:           "null map",
			obj:            map[string]interface{}{"path": nil},
			path:           []string{"path"},
			wantObj:        nil,
			wantFound:      true,
			wantErrMessage: "",
		},
		{
			name:           "non map",
			obj:            map[string]interface{}{"path": 0},
			path:           []string{"path"},
			wantObj:        nil,
			wantFound:      false,
			wantErrMessage: "type int",
		},
		{
			name:           "empty map",
			obj:            map[string]interface{}{"path": map[string]interface{}{}},
			path:           []string{"path"},
			wantObj:        map[string]string{},
			wantFound:      true,
			wantErrMessage: "",
		},
		{
			name:           "string value",
			obj:            map[string]interface{}{"path": map[string]interface{}{"a": "1", "b": "2"}},
			path:           []string{"path"},
			wantObj:        map[string]string{"a": "1", "b": "2"},
			wantFound:      true,
			wantErrMessage: "",
		},
		{
			name:           "null value",
			obj:            map[string]interface{}{"path": map[string]interface{}{"a": "1", "b": nil}},
			path:           []string{"path"},
			wantObj:        map[string]string{"a": "1", "b": ""},
			wantFound:      true,
			wantErrMessage: "",
		},
		{
			name:           "invalid value",
			obj:            map[string]interface{}{"path": map[string]interface{}{"a": "1", "b": nil, "c": 0}},
			path:           []string{"path"},
			wantObj:        nil,
			wantFound:      false,
			wantErrMessage: `key "c": 0`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gotObj, gotFound, gotErr := NestedNullCoercingStringMap(tc.obj, tc.path...)
			if !reflect.DeepEqual(gotObj, tc.wantObj) {
				t.Errorf("got %#v, wanted %#v", gotObj, tc.wantObj)
			}
			if gotFound != tc.wantFound {
				t.Errorf("got %v, wanted %v", gotFound, tc.wantFound)
			}
			if tc.wantErrMessage != "" {
				if gotErr == nil {
					t.Errorf("got nil error, wanted %s", tc.wantErrMessage)
				} else if gotErrMessage := gotErr.Error(); !strings.Contains(gotErrMessage, tc.wantErrMessage) {
					t.Errorf("wanted error %q, got: %v", gotErrMessage, tc.wantErrMessage)
				}
			} else if gotErr != nil {
				t.Errorf("wanted nil error, got %v", gotErr)
			}
		})
	}
}

func TestDecodeList(t *testing.T) {
	for _, tc := range []struct {
		name    string
		data    string
		list    *UnstructuredList
		want    *UnstructuredList
		wantErr string
	}{
		{
			name: "typed list",
			data: `{"kind":"PodList","apiVersion":"v1","metadata":{"resourceVersion":"5"},"items":[{"metadata":{"name":"a","generation":2}},{"kind":"Pod","metadata":{"name":"b"}},null]}`,
			want: &UnstructuredList{
				Object: map[string]interface{}{"kind": "PodList", "apiVersion": "v1", "metadata": map[string]interface{}{"resourceVersion": "5"}},
				Items: []Unstructured{
					{Object: map[string]interface{}{"kind": "Pod", "apiVersion": "v1", "metadata": map[string]interface{}{"name": "a", "generation": int64(2)}}},
					// Kind and apiVersion are only set on items that have neither.
					{Object: map[string]interface{}{"kind": "Pod", "metadata": map[string]interface{}{"name": "b"}}},
					{Object: map[string]interface{}{"kind": "Pod", "apiVersion": "v1"}},
				},
			},
		},
		{
			name: "list of mixed kinds",
			data: `{"kind":"List","apiVersion":"v1","items":[{"kind":"Pod","apiVersion":"v1","spec":{"volumes":[{"configMap":{"items":[{"key":"k"}]}}]}},{"kind":"Service","apiVersion":"v1","spec":{"ports":[{"port":1.5}]}}]}`,
			want: &UnstructuredList{
				Object: map[string]interface{}{"kind": "List", "apiVersion": "v1"},
				Items: []Unstructured{
					{Object: map[string]interface{}{"kind": "Pod", "apiVersion": "v1", "spec": map[string]interface{}{"volumes": []interface{}{map[string]interface{}{"configMap": map[string]interface{}{"items": []interface{}{map[string]interface{}{"key": "k"}}}}}}}},
					{Object: map[string]interface{}{"kind": "Service", "apiVersion": "v1", "spec": map[string]interface{}{"ports": []interface{}{map[string]interface{}{"port": 1.5}}}}},
				},
			},
		},
		{
			name: "repeated items key, longer last",
			data: `{"kind":"PodList","apiVersion":"v1","items":[{"a":1}],"items":[{"b":1},{"c":1}]}`,
			want: &UnstructuredList{
				Object: map[string]interface{}{"kind": "PodList", "apiVersion": "v1"},
				Items: []Unstructured{
					{Object: map[string]interface{}{"kind": "Pod", "apiVersion": "v1", "b": int64(1)}},
					{Object: map[string]interface{}{"kind": "Pod", "apiVersion": "v1", "c": int64(1)}},
				},
			},
		},
		{
			name: "repeated items key, shorter last",
			data: `{"kind":"PodList","apiVersion":"v1","items":[{"b":1},{"c":1}],"items":[{"a":1}]}`,
			want: &UnstructuredList{
				Object: map[string]interface{}{"kind": "PodList", "apiVersion": "v1"},
				Items:  []Unstructured{{Object: map[string]interface{}{"kind": "Pod", "apiVersion": "v1", "a": int64(1)}}},
			},
		},
		{
			// Keys that data does not set are kept, as json.Unmarshal into a
			// non-nil map does, and items are replaced.
			name: "list with content",
			data: `{"kind":"PodList","apiVersion":"v1","items":[{"metadata":{"name":"a"}}]}`,
			list: &UnstructuredList{
				Object: map[string]interface{}{"kind": "OldList", "stale": "x", "items": []interface{}{map[string]interface{}{"kind": "Stale"}}},
				Items:  []Unstructured{{Object: map[string]interface{}{"kind": "Stale"}}},
			},
			want: &UnstructuredList{
				Object: map[string]interface{}{"kind": "PodList", "apiVersion": "v1", "stale": "x"},
				Items:  []Unstructured{{Object: map[string]interface{}{"kind": "Pod", "apiVersion": "v1", "metadata": map[string]interface{}{"name": "a"}}}},
			},
		},
		{
			name:    "item that is not an object",
			data:    `{"kind":"PodList","items":[{"a":1},2]}`,
			wantErr: "cannot unmarshal number into Go value of type map[string]interface {}",
		},
		{
			// A map decode keeps only the last items value, so this error
			// comes from decoding dList.
			name:    "repeated items key with a non-array value",
			data:    `{"kind":"PodList","items":5,"items":[{"a":1}]}`,
			wantErr: "cannot unmarshal number",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			list := tc.list
			if list == nil {
				list = &UnstructuredList{}
			}
			err := list.UnmarshalJSON([]byte(tc.data))
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("got error %v, want one containing %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(list, tc.want) {
				t.Errorf("got  %#v\nwant %#v", list, tc.want)
			}
		})
	}
}

func benchmarkListJSON(n int) []byte {
	const item = `{"metadata":{"name":"web-%[1]d","namespace":"default","uid":"4c1f2a9e-0d3b-4c1a-9f2e-%012[1]d","resourceVersion":"%[1]d","labels":{"app":"web","tier":"frontend"},` +
		`"ownerReferences":[{"apiVersion":"apps/v1","kind":"ReplicaSet","name":"web","uid":"a1b2c3d4","controller":true}],` +
		`"managedFields":[{"manager":"kube-controller-manager","operation":"Update","apiVersion":"v1","fieldsType":"FieldsV1","fieldsV1":{"f:metadata":{"f:labels":{".":{},"f:app":{}}},"f:spec":{"f:containers":{"k:{\"name\":\"web\"}":{".":{},"f:image":{}}}}}}]},` +
		`"spec":{"containers":[{"name":"web","image":"registry.k8s.io/nginx-slim:0.27","ports":[{"containerPort":8080,"protocol":"TCP"}],"resources":{"requests":{"cpu":"100m","memory":"128Mi"}}}],` +
		`"volumes":[{"name":"kube-api-access","projected":{"defaultMode":420,"sources":[{"serviceAccountToken":{"expirationSeconds":3607,"path":"token"}},{"configMap":{"name":"kube-root-ca.crt","items":[{"key":"ca.crt","path":"ca.crt"}]}}]}}],` +
		`"nodeName":"node-%[1]d","restartPolicy":"Always","terminationGracePeriodSeconds":30},` +
		`"status":{"phase":"Running","podIP":"10.244.0.1","conditions":[{"type":"Ready","status":"True","lastTransitionTime":"2026-09-29T12:00:05Z"}]}}`
	items := make([]string, n)
	for i := range items {
		items[i] = fmt.Sprintf(item, i)
	}
	return []byte(`{"kind":"PodList","apiVersion":"v1","metadata":{"resourceVersion":"123"},"items":[` + strings.Join(items, ",") + `]}`)
}

func BenchmarkDecodeList(b *testing.B) {
	for _, n := range []int{10, 1000} {
		data := benchmarkListJSON(n)
		b.Run(fmt.Sprintf("items=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(data)))
			for b.Loop() {
				if err := (&UnstructuredList{}).UnmarshalJSON(data); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
