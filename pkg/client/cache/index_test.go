/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"testing"
)

func testIndexFunc(obj interface{}) (string, error) {
	pod := obj.(*api.Pod)
	return pod.Labels["foo"], nil
}

func TestGetIndexFuncValues(t *testing.T) {
	index := NewIndexer(MetaNamespaceKeyFunc, Indexers{"testmodes": testIndexFunc})

	pod1 := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "one", Labels: map[string]string{"foo": "bar"}}}
	pod2 := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "two", Labels: map[string]string{"foo": "bar"}}}
	pod3 := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "tre", Labels: map[string]string{"foo": "biz"}}}

	index.Add(pod1)
	index.Add(pod2)
	index.Add(pod3)

	keys := index.ListIndexFuncValues("testmodes")
	if len(keys) != 2 {
		t.Errorf("Expected 2 keys but got %v", len(keys))
	}

	for _, key := range keys {
		if key != "bar" && key != "biz" {
			t.Errorf("Expected only 'bar' or 'biz' but got %s", key)
		}
	}
}
