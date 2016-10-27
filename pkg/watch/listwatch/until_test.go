/*
Copyright 2016 The Kubernetes Authors.

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

package listwatch

import (
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

type lw struct {
	list  runtime.Object
	watch watch.Interface
}

func (w lw) List(options api.ListOptions) (runtime.Object, error) {
	return w.list, nil
}

func (w lw) Watch(options api.ListOptions) (watch.Interface, error) {
	return w.watch, nil
}

func TestListWatchUntil(t *testing.T) {
	fw := watch.NewFake()
	go func() {
		var obj *api.Pod
		fw.Modify(obj)
	}()
	listwatch := lw{
		list:  &api.PodList{Items: []api.Pod{{}}},
		watch: fw,
	}

	conditions := []watch.ConditionFunc{
		func(event watch.Event) (bool, error) {
			t.Logf("got %#v", event)
			return event.Type == watch.Added, nil
		},
		func(event watch.Event) (bool, error) {
			t.Logf("got %#v", event)
			return event.Type == watch.Modified, nil
		},
	}

	timeout := 10 * time.Second
	lastEvent, err := Until(timeout, listwatch, conditions...)
	if err != nil {
		t.Fatalf("expected nil error, got %#v", err)
	}
	if lastEvent == nil {
		t.Fatal("expected an event")
	}
	if lastEvent.Type != watch.Modified {
		t.Fatalf("expected MODIFIED event type, got %v", lastEvent.Type)
	}
	if got, isPod := lastEvent.Object.(*api.Pod); !isPod {
		t.Fatalf("expected a pod event, got %#v", got)
	}
}
