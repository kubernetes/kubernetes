/*
Copyright 2015 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/util/sets"
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func testIndexFunc(obj interface{}) ([]string, error) {
	pod := obj.(*v1.Pod)
	return []string{pod.Labels["foo"]}, nil
}

func TestGetIndexFuncValues(t *testing.T) {
	index := NewIndexer(MetaNamespaceKeyFunc, Indexers{"testmodes": testIndexFunc})

	pod1 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "one", Labels: map[string]string{"foo": "bar"}}}
	pod2 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "two", Labels: map[string]string{"foo": "bar"}}}
	pod3 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "tre", Labels: map[string]string{"foo": "biz"}}}

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

func testUsersIndexFunc(obj interface{}) ([]string, error) {
	pod := obj.(*v1.Pod)
	usersString := pod.Annotations["users"]

	return strings.Split(usersString, ","), nil
}

func TestMultiIndexKeys(t *testing.T) {
	index := NewIndexer(MetaNamespaceKeyFunc, Indexers{"byUser": testUsersIndexFunc})

	pod1 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "one", Annotations: map[string]string{"users": "ernie,bert"}}}
	pod2 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "two", Annotations: map[string]string{"users": "bert,oscar"}}}
	pod3 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "tre", Annotations: map[string]string{"users": "ernie,elmo"}}}

	index.Add(pod1)
	index.Add(pod2)
	index.Add(pod3)

	expected := map[string]sets.String{}
	expected["ernie"] = sets.NewString("one", "tre")
	expected["bert"] = sets.NewString("one", "two")
	expected["elmo"] = sets.NewString("tre")
	expected["oscar"] = sets.NewString("two")
	expected["elmo"] = sets.NewString() // let's just make sure we don't get anything back in this case
	{
		for k, v := range expected {
			found := sets.String{}
			indexResults, err := index.ByIndex("byUser", k)
			if err != nil {
				t.Errorf("Unexpected error %v", err)
			}
			for _, item := range indexResults {
				found.Insert(item.(*v1.Pod).Name)
			}
			items := v.List()
			if !found.HasAll(items...) {
				t.Errorf("missing items, index %s, expected %v but found %v", k, items, found.List())
			}
		}
	}

	index.Delete(pod3)
	erniePods, err := index.ByIndex("byUser", "ernie")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(erniePods) != 1 {
		t.Errorf("Expected 1 pods but got %v", len(erniePods))
	}
	for _, erniePod := range erniePods {
		if erniePod.(*v1.Pod).Name != "one" {
			t.Errorf("Expected only 'one' but got %s", erniePod.(*v1.Pod).Name)
		}
	}

	elmoPods, err := index.ByIndex("byUser", "elmo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(elmoPods) != 0 {
		t.Errorf("Expected 0 pods but got %v", len(elmoPods))
	}

	copyOfPod2 := pod2.DeepCopy()
	copyOfPod2.Annotations["users"] = "oscar"
	index.Update(copyOfPod2)
	bertPods, err := index.ByIndex("byUser", "bert")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(bertPods) != 1 {
		t.Errorf("Expected 1 pods but got %v", len(bertPods))
	}
	for _, bertPod := range bertPods {
		if bertPod.(*v1.Pod).Name != "one" {
			t.Errorf("Expected only 'one' but got %s", bertPod.(*v1.Pod).Name)
		}
	}
}
