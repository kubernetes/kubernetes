/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

	"sync"

	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/watch"
)

type fakeQueue struct {
	cache.Queue
	deleteNotify chan bool
}

func (q *fakeQueue) Delete(obj interface{}) error {
	err := q.Queue.Delete(obj)
	// We should only notify after Delete() is called.
	select {
	case q.deleteNotify <- true:
	default:
	}
	return err
}

func TestInformerRace(t *testing.T) {
	// Sequence:
	// - List of two pod
	// - Block AddFunc
	// - For the pod it blocks, we try to delete the other pod
	// - Unblock
	// - Wait for DeleteFunc
	podFoo := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	podBar := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "bar"}}
	var wg sync.WaitGroup
	wg.Add(1)
	nameChan := make(chan string, 1)
	doneCh := make(chan struct{})
	w := watch.NewFake()
	lw := &cache.ListWatch{
		ListFunc: func(options api.ListOptions) (runtime.Object, error) {
			return &api.PodList{
				Items: []api.Pod{*podFoo, *podBar},
			}, nil
		},
		WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
			return w, nil
		}}

	_, controller := NewInformer(lw, &api.Pod{}, 0,
		ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				nameChan <- obj.(*api.Pod).Name
				wg.Wait()
			},
			DeleteFunc: func(obj interface{}) {
				close(doneCh)
			},
		})

	notify := make(chan bool, 1)
	q := &fakeQueue{
		Queue:        controller.config.Queue,
		deleteNotify: notify,
	}
	controller.config.Queue = q

	stopCh := make(chan struct{})
	defer close(stopCh)
	go controller.Run(stopCh)

	name := <-nameChan
	if name == "foo" {
		w.Delete(podBar)
	} else {
		w.Delete(podFoo)
	}
	<-notify
	wg.Done()

	select {
	case <-doneCh:
	case <-time.After(3 * time.Second):
		t.Errorf("timeout after %v", wait.ForeverTestTimeout)
	}
}
