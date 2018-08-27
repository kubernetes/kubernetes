/*
Copyright 2018 The Kubernetes Authors.

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

package cache

import (
	"fmt"
	"reflect"
	"testing"
)

func TestThreadSafeStoreAdd(t *testing.T) {
	tests := []struct {
		obj             interface{}
		key             string
		store           ThreadSafeStore
		expectedItems   map[string]interface{}
		expectedIndices Indices
	}{
		{
			newFakeObject("111", "aaa"),
			"aaa",
			&threadSafeMap{
				items: map[string]interface{}{},
				indexers: Indexers{
					"test": fakeIndexFunc,
				},
				indices: Indices{},
			},
			map[string]interface{}{"aaa": fakeObject{key: "111", val: "aaa"}},
			Indices{"test": Index{"111": {"aaa": {}}}},
		},
		{
			newFakeObject("222", "bbb"),
			"bbb",
			&threadSafeMap{
				items: map[string]interface{}{"bbb": fakeObject{key: "222", val: "aaa"}},
				indexers: Indexers{
					"test": fakeIndexFunc,
				},
				indices: Indices{"test": Index{"222": {"aaa": {}}}},
			},
			map[string]interface{}{"bbb": fakeObject{key: "222", val: "bbb"}},
			Indices{"test": Index{"222": {"aaa": {}, "bbb": {}}}},
		},
	}

	for _, test := range tests {
		test.store.Add(test.key, test.obj)

		if !reflect.DeepEqual(test.expectedItems, test.store.(*threadSafeMap).items) {
			t.Errorf("expected %#v, got %#v", test.expectedItems, test.store.(*threadSafeMap).items)
		}
		if !reflect.DeepEqual(test.expectedIndices, test.store.(*threadSafeMap).indices) {
			t.Errorf("expected %#v, got %#v", test.expectedIndices, test.store.(*threadSafeMap).indices)
		}
	}
}

func TestThreadSafeStoreUpdate(t *testing.T) {
	tests := []struct {
		obj             interface{}
		key             string
		store           ThreadSafeStore
		expectedItems   map[string]interface{}
		expectedIndices Indices
	}{
		{
			newFakeObject("111", "aaa"),
			"aaa",
			&threadSafeMap{
				items: map[string]interface{}{},
				indexers: Indexers{
					"test": fakeIndexFunc,
				},
				indices: Indices{},
			},
			map[string]interface{}{"aaa": fakeObject{key: "111", val: "aaa"}},
			Indices{"test": Index{"111": {"aaa": {}}}},
		},
		{
			newFakeObject("222", "bbb"),
			"bbb",
			&threadSafeMap{
				items: map[string]interface{}{"bbb": fakeObject{key: "222", val: "aaa"}},
				indexers: Indexers{
					"test": fakeIndexFunc,
				},
				indices: Indices{"test": Index{"222": {"aaa": {}}}},
			},
			map[string]interface{}{"bbb": fakeObject{key: "222", val: "bbb"}},
			Indices{"test": Index{"222": {"aaa": {}, "bbb": {}}}},
		},
	}

	for _, test := range tests {
		test.store.Update(test.key, test.obj)

		if !reflect.DeepEqual(test.expectedItems, test.store.(*threadSafeMap).items) {
			t.Errorf("expected %#v, got %#v", test.expectedItems, test.store.(*threadSafeMap).items)
		}
		if !reflect.DeepEqual(test.expectedIndices, test.store.(*threadSafeMap).indices) {
			t.Errorf("expected %#v, got %#v", test.expectedIndices, test.store.(*threadSafeMap).indices)
		}
	}
}

func TestThreadSafeStoreDelete(t *testing.T) {
	tests := []struct {
		key             string
		store           ThreadSafeStore
		expectedItems   map[string]interface{}
		expectedIndices Indices
	}{
		{
			"aaa",
			&threadSafeMap{
				items: map[string]interface{}{},
				indexers: Indexers{
					"test": fakeIndexFunc,
				},
				indices: Indices{},
			},
			map[string]interface{}{},
			Indices{},
		},
		{
			"bbb",
			&threadSafeMap{
				items: map[string]interface{}{"bbb": fakeObject{key: "222", val: "bbb"}},
				indexers: Indexers{
					"test": fakeIndexFunc,
				},
				indices: Indices{"test": Index{"222": {"bbb": {}}}},
			},
			map[string]interface{}{},
			Indices{"test": Index{"222": {}}},
		},
		{
			"ccc",
			&threadSafeMap{
				items: map[string]interface{}{"ddd": fakeObject{key: "333", val: "ddd"}},
				indexers: Indexers{
					"test": fakeIndexFunc,
				},
				indices: Indices{"test": Index{"333": {"ddd": {}}}},
			},
			map[string]interface{}{"ddd": fakeObject{key: "333", val: "ddd"}},
			Indices{"test": Index{"333": {"ddd": {}}}},
		},
	}

	for _, test := range tests {
		test.store.Delete(test.key)

		if !reflect.DeepEqual(test.expectedItems, test.store.(*threadSafeMap).items) {
			t.Errorf("expected %#v, got %#v", test.expectedItems, test.store.(*threadSafeMap).items)
		}
		if !reflect.DeepEqual(test.expectedIndices, test.store.(*threadSafeMap).indices) {
			t.Errorf("expected %#v, got %#v", test.expectedIndices, test.store.(*threadSafeMap).indices)
		}
	}
}

func TestThreadSafeStoreGet(t *testing.T) {
	tests := []struct {
		key            string
		store          ThreadSafeStore
		expectedItem   interface{}
		expectedExists bool
	}{
		{
			"aaa",
			&threadSafeMap{
				items: map[string]interface{}{},
				indexers: Indexers{
					"test": fakeIndexFunc,
				},
				indices: Indices{},
			},
			nil,
			false,
		},
		{
			"bbb",
			&threadSafeMap{
				items: map[string]interface{}{"bbb": fakeObject{key: "222", val: "bbb"}},
				indexers: Indexers{
					"test": fakeIndexFunc,
				},
				indices: Indices{"test": Index{"222": {"bbb": {}}}},
			},
			fakeObject{key: "222", val: "bbb"},
			true,
		},
		{
			"ccc",
			&threadSafeMap{
				items: map[string]interface{}{"ddd": fakeObject{key: "333", val: "ddd"}},
				indexers: Indexers{
					"test": fakeIndexFunc,
				},
				indices: Indices{"test": Index{"333": {"ddd": {}}}},
			},
			nil,
			false,
		},
	}

	for _, test := range tests {
		item, exists := test.store.Get(test.key)

		if !reflect.DeepEqual(test.expectedExists, exists) {
			t.Errorf("expected %#v, got %#v", test.expectedExists, exists)
		}
		if !reflect.DeepEqual(test.expectedItem, item) {
			t.Errorf("expected %#v, got %#v", test.expectedItem, item)
		}
	}
}

func TestThreadSafeStoreList(t *testing.T) {
	tests := []struct {
		store         ThreadSafeStore
		expectedItems []interface{}
	}{
		{
			&threadSafeMap{
				items: map[string]interface{}{},
				indexers: Indexers{
					"test": fakeIndexFunc,
				},
				indices: Indices{},
			},
			[]interface{}{},
		},
		{
			&threadSafeMap{
				items: map[string]interface{}{"bbb": fakeObject{key: "222", val: "bbb"}},
				indexers: Indexers{
					"test": fakeIndexFunc,
				},
				indices: Indices{"test": Index{"222": {"bbb": {}}}},
			},
			[]interface{}{fakeObject{key: "222", val: "bbb"}},
		},
	}

	for _, test := range tests {
		items := test.store.List()

		if !reflect.DeepEqual(test.expectedItems, items) {
			t.Errorf("expected %#v, got %#v", test.expectedItems, items)
		}
	}
}

func TestThreadSafeStoreListKeys(t *testing.T) {
	tests := []struct {
		store        ThreadSafeStore
		expectedKeys []string
	}{
		{
			&threadSafeMap{
				items: map[string]interface{}{},
				indexers: Indexers{
					"test": fakeIndexFunc,
				},
				indices: Indices{},
			},
			[]string{},
		},
		{
			&threadSafeMap{
				items: map[string]interface{}{"bbb": fakeObject{key: "222", val: "bbb"}},
				indexers: Indexers{
					"test": fakeIndexFunc,
				},
				indices: Indices{"test": Index{"222": {"bbb": {}}}},
			},
			[]string{"bbb"},
		},
	}

	for _, test := range tests {
		keys := test.store.ListKeys()

		if !reflect.DeepEqual(test.expectedKeys, keys) {
			t.Errorf("expected %#v, got %#v", test.expectedKeys, keys)
		}
	}
}

func TestUpdateIndices(t *testing.T) {
	tests := []struct {
		oldObj          interface{}
		newObj          interface{}
		key             string
		testMap         *threadSafeMap
		expectedErr     bool
		expectedIndices Indices
	}{
		{
			nil,
			nil,
			"aaa",
			&threadSafeMap{
				items: map[string]interface{}{},
				indexers: Indexers{
					"test": fakeIndexFunc,
				},
				indices: Indices{},
			},
			true,
			Indices{},
		},
		{
			newFakeObject("111", "bbb"),
			nil,
			"bbb",
			&threadSafeMap{
				items: map[string]interface{}{},
				indexers: Indexers{
					"test": fakeIndexFunc,
				},
				indices: Indices{},
			},
			true,
			Indices{},
		},
		{
			nil,
			newFakeObject("333", "ccc"),
			"ccc",
			&threadSafeMap{
				items: map[string]interface{}{},
				indexers: Indexers{
					"test": fakeIndexFunc,
				},
				indices: Indices{},
			},
			false,
			Indices{"test": Index{"333": {"ccc": {}}}},
		},
		{
			newFakeObject("444", "ddd"),
			newFakeObject("555", "ddd"),
			"ddd",
			&threadSafeMap{
				items: map[string]interface{}{},
				indexers: Indexers{
					"test": fakeIndexFunc,
				},
				indices: Indices{"test": Index{"444": {"ddd": {}}}},
			},
			false,
			Indices{"test": Index{"444": {}, "555": {"ddd": {}}}},
		},
	}

	for _, test := range tests {
		err := testPanicHandler(test.testMap, test.key, test.oldObj, test.newObj)
		if err != nil {
			if !test.expectedErr {
				t.Errorf("expected no error, got %s", err)
			}
		}

		if !reflect.DeepEqual(test.expectedIndices, test.testMap.indices) {
			t.Errorf("expected %#v, got %#v", test.expectedIndices, test.testMap.indices)
		}
	}
}

func TestDeleteFromIndices(t *testing.T) {
	tests := []struct {
		obj             interface{}
		key             string
		testMap         *threadSafeMap
		expectedErr     bool
		expectedIndices Indices
	}{
		{
			nil,
			"aaa",
			&threadSafeMap{
				items: map[string]interface{}{},
				indexers: Indexers{
					"test": fakeIndexFunc,
				},
				indices: Indices{},
			},
			true,
			Indices{},
		},
		{
			newFakeObject("111", "bbb"),
			"bbb",
			&threadSafeMap{
				items: map[string]interface{}{},
				indexers: Indexers{
					"test": fakeIndexFunc,
				},
				indices: Indices{"test": Index{"111": {"bbb": {}}}},
			},
			false,
			Indices{"test": Index{"111": {}}},
		},
	}

	for _, test := range tests {
		err := testPanicHandler(test.testMap, test.key, test.obj)
		if err != nil {
			if !test.expectedErr {
				t.Errorf("expected no error, got %s", err)
			}
		}

		if !reflect.DeepEqual(test.expectedIndices, test.testMap.indices) {
			t.Errorf("expected %#v, got %#v", test.expectedIndices, test.testMap.indices)
		}
	}
}

type fakeObject struct {
	key string
	val string
}

func newFakeObject(key string, val string) fakeObject {
	return fakeObject{
		key: key,
		val: val,
	}
}

func fakeIndexFunc(obj interface{}) ([]string, error) {
	o, ok := obj.(fakeObject)
	if !ok {
		return []string{""}, fmt.Errorf("invalid object: %v", obj)
	}

	return []string{o.key}, nil
}

func testPanicHandler(m *threadSafeMap, key string, obj ...interface{}) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("%s", r)
		}
	}()

	if len(obj) == 1 {
		m.deleteFromIndices(obj[0], key)
	} else if len(obj) == 2 {
		m.updateIndices(obj[0], obj[1], key)
	} else {
		return fmt.Errorf("panic handler error")
	}

	return err
}
