/*
Copyright 2014 The Kubernetes Authors.

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
	"k8s.io/kubernetes/staging/src/k8s.io/apimachinery/pkg/util/sets"
	"testing"
)

func TestAddIndexerAfterAdd(t *testing.T) {
	store := NewThreadSafeStore(Indexers{}, Indices{})

	// Add first indexer
	err := store.AddIndexers(Indexers{
		"first": func(obj interface{}) ([]string, error) {
			value := obj.(string)
			return []string{
				value,
			}, nil
		},
	})
	if err != nil {
		t.Errorf("failed to add first indexer")
	}

	// Add some data to index
	store.Add("keya", "value")
	store.Add("keyb", "value")

	// Assert
	indexKeys, _ := store.IndexKeys("first", "value")
	expected := sets.NewString("keya", "keyb")
	actual := sets.NewString(indexKeys...)
	if !actual.Equal(expected) {
		t.Errorf("expected %v does not match actual %v", expected, actual)
	}

	// Add same indexer, which should fail
	err = store.AddIndexers(Indexers{
		"first": func(interface{}) ([]string, error) {
			return nil, nil
		},
	})
	if err == nil {
		t.Errorf("Add same index should have failed")
	}

	// Add new indexer
	err = store.AddIndexers(Indexers{
		"second": func(obj interface{}) ([]string, error) {
			v := obj.(string)
			return []string{
				v +"2",
			}, nil
		},
	})
	if err != nil {
		t.Errorf("failed to add second indexer")
	}

	// Assert indexers was added
	if _, ok := store.GetIndexers()["first"]; !ok {
		t.Errorf("missing indexer first")
	}
	if _, ok := store.GetIndexers()["second"]; !ok {
		t.Errorf("missing indexer second")
	}

	// Assert existing data is re-indexed
	indexKeys, _ = store.IndexKeys("first", "value")
	expected = sets.NewString("keya", "keyb")
	actual = sets.NewString(indexKeys...)
	if !actual.Equal(expected) {
		t.Errorf("expected %v does not match actual %v", expected, actual)
	}
	indexKeys, _ = store.IndexKeys("second", "value2")
	expected = sets.NewString("keya", "keyb")
	actual = sets.NewString(indexKeys...)
	if !actual.Equal(expected) {
		t.Errorf("expected %v does not match actual %v", expected, actual)
	}

	// Add more data
	store.Add("keyc", "value")
	store.Add("keyd", "value")

	// Assert new data is indexed
	indexKeys, _ = store.IndexKeys("first", "value")
	expected = sets.NewString("keya", "keyb", "keyc", "keyd")
	actual = sets.NewString(indexKeys...)
	if !actual.Equal(expected) {
		t.Errorf("expected %v does not match actual %v", expected, actual)
	}
	indexKeys, _ = store.IndexKeys("second", "value2")
	expected = sets.NewString("keya", "keyb", "keyc", "keyd")
	actual = sets.NewString(indexKeys...)
	if !actual.Equal(expected) {
		t.Errorf("expected %v does not match actual %v", expected, actual)
	}
}

