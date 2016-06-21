// +build !race

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
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

type testLW struct {
	ListFunc  func(options api.ListOptions) (runtime.Object, error)
	WatchFunc func(options api.ListOptions) (watch.Interface, error)
}

func (t *testLW) List(options api.ListOptions) (runtime.Object, error) {
	return t.ListFunc(options)
}
func (t *testLW) Watch(options api.ListOptions) (watch.Interface, error) {
	return t.WatchFunc(options)
}

func TestMutationDetector(t *testing.T) {
	source := NewFakeControllerSource()
	fakeWatch := watch.NewFake()
	lw := &testLW{
		WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
			return fakeWatch, nil
		},
		ListFunc: func(options api.ListOptions) (runtime.Object, error) {
			return source.List(options)
		},
	}
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:   "anything",
			Labels: map[string]string{"check": "foo"},
		},
	}
	stopCh := make(chan struct{})
	defer close(stopCh)
	addReceived := make(chan bool)
	mutationFound := make(chan bool)

	informer := NewSharedInformer(lw, &api.Pod{}, 1*time.Second).(*sharedIndexInformer)
	informer.cacheMutationDetector = &defaultCacheMutationDetector{
		name:   "name",
		period: 1 * time.Second,
		failureFunc: func(message string) {
			mutationFound <- true
		},
	}
	informer.AddEventHandler(
		ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				addReceived <- true
			},
		},
	)
	go informer.Run(stopCh)

	fakeWatch.Add(pod)

	select {
	case <-addReceived:
	}

	pod.Labels["change"] = "true"

	select {
	case <-mutationFound:
	}

}
