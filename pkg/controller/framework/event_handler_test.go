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
	"time"

	"k8s.io/kubernetes/pkg/util/wait"
)

// TestQueuedEventHandler tests that queuedEventHandler should calls corresponding
// callback funcs, and shouldn't block on handlers.
func TestQueuedEventHandler(t *testing.T) {
	var res int
	resCh := make(chan int)
	h := newQueuedEventHandler(ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			resCh <- obj.(int)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			resCh <- newObj.(int)
		},
		DeleteFunc: func(obj interface{}) {
			resCh <- obj.(int)
		},
	})
	stopCh := make(chan struct{})
	defer close(stopCh)
	go h.run(stopCh)

	h.OnAdd(1)
	select {
	case res = <-resCh:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("test timeout after %v", wait.ForeverTestTimeout)
	}
	if res != 1 {
		t.Errorf("result want=1, get=%d", res)
	}

	h.OnUpdate(1, 2)
	select {
	case res = <-resCh:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("test timeout after %v", wait.ForeverTestTimeout)
	}
	if res != 2 {
		t.Errorf("result want=1, get=%d", res)
	}

	h.OnDelete(1)
	select {
	case res = <-resCh:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("test timeout after %v", wait.ForeverTestTimeout)
	}
	if res != 1 {
		t.Errorf("result want=1, get=%d", res)
	}
}
