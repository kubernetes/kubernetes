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

package framework

import (
	"sync"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/watch"
)

// ensure the watch delivers the requested and only the requested items.
func consume(t *testing.T, w watch.Interface, rvs []string, done *sync.WaitGroup) {
	defer done.Done()
	for _, rv := range rvs {
		got, ok := <-w.ResultChan()
		if !ok {
			t.Errorf("%#v: unexpected channel close, wanted %v", rvs, rv)
			return
		}
		gotRV := got.Object.(*v1.Pod).ObjectMeta.ResourceVersion
		if e, a := rv, gotRV; e != a {
			t.Errorf("wanted %v, got %v", e, a)
		} else {
			t.Logf("Got %v as expected", gotRV)
		}
	}
	// We should not get anything else.
	got, open := <-w.ResultChan()
	if open {
		t.Errorf("%#v: unwanted object %#v", rvs, got)
	}
}

func TestRCNumber(t *testing.T) {
	pod := func(name string) *v1.Pod {
		return &v1.Pod{
			ObjectMeta: v1.ObjectMeta{
				Name: name,
			},
		}
	}

	wg := &sync.WaitGroup{}
	wg.Add(3)

	source := NewFakeControllerSource()
	source.Add(pod("foo"))
	source.Modify(pod("foo"))
	source.Modify(pod("foo"))

	w, err := source.Watch(v1.ListOptions{ResourceVersion: "1"})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	go consume(t, w, []string{"2", "3"}, wg)

	list, err := source.List(v1.ListOptions{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if e, a := "3", list.(*api.List).ResourceVersion; e != a {
		t.Errorf("wanted %v, got %v", e, a)
	}

	w2, err := source.Watch(v1.ListOptions{ResourceVersion: "2"})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	go consume(t, w2, []string{"3"}, wg)

	w3, err := source.Watch(v1.ListOptions{ResourceVersion: "3"})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	go consume(t, w3, []string{}, wg)
	source.Shutdown()
	wg.Wait()
}
