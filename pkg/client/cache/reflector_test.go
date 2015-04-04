/*
Copyright 2014 Google Inc. All rights reserved.

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
	"strconv"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

type testLW struct {
	ListFunc  func() (runtime.Object, error)
	WatchFunc func(resourceVersion string) (watch.Interface, error)
}

func (t *testLW) List() (runtime.Object, error) { return t.ListFunc() }
func (t *testLW) Watch(resourceVersion string) (watch.Interface, error) {
	return t.WatchFunc(resourceVersion)
}

type stopFunc func(w *watch.FakeWatcher, r *Reflector)

func TestTimeoutReflector(t *testing.T) {
	s := NewFIFO(MetaNamespaceKeyFunc)
	r := NewReflector(&testLW{}, &api.Pod{}, s, 0)

	// Test at least 2 pods to confirm we aren't trying to do something like
	// send down a closed channel.
	pod1 := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "bar"}}
	pod2 := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	newPods := map[*api.Pod]stopFunc{
		pod1: func(w *watch.FakeWatcher, r *Reflector) { w.Error(pod1) },
		pod2: func(w *watch.FakeWatcher, r *Reflector) { w.Error(pod1) },
		pod1: func(w *watch.FakeWatcher, r *Reflector) { r.Cancel(nil) },
		pod2: func(w *watch.FakeWatcher, r *Reflector) { r.Cancel(nil) },
	}
	for p, stopper := range newPods {
		fw := watch.NewFake()
		r.listerWatcher = &testLW{
			WatchFunc: func(rv string) (watch.Interface, error) {
				return fw, nil
			},
			ListFunc: func() (runtime.Object, error) {
				return &api.PodList{ListMeta: api.ListMeta{ResourceVersion: "1"}}, nil
			},
		}
		go r.listAndWatch()
		fw.Add(p)
		if !r.open || fw.Stopped {
			t.Errorf("Channels should only be stopped on cancellation, r")
		}
		// This error should force a cancellation of the listWatch above,
		// wait for it, then check if the channel is closed.
		stopper(fw, r)
		cOpen, wOpen := true, true
		for cOpen && wOpen {
			select {
			case _, ok := <-r.cancellationCh:
				if ok {
					t.Errorf("Cancellation channel left open after cancellation")
				}
				cOpen = false
			case _, ok := <-fw.ResultChan():
				if ok {
					t.Errorf("Watch channel left open after cancellation")
				}
				wOpen = false
			case <-time.After(100 * time.Millisecond):
				t.Errorf("the cancellation is at least 99 milliseconds late")
				break
			}
		}

		// Get the inserted pod out of the fifo, but give up if it takes too long.
		podCh := make(chan interface{})
		go func() {
			podCh <- s.Pop()
			close(podCh)
		}()
		select {
		case pod, ok := <-podCh:
			if !reflect.DeepEqual(p, pod.(*api.Pod)) {
				t.Errorf("Unexpected pod retrieved from fifo")
			}
			fmt.Printf("Retrieved pod %+v with ok %+v", pod, ok)
		case <-time.After(100 * time.Millisecond):
			s.Replace([]interface{}{})
			if _, ok := <-podCh; ok {
				t.Errorf("Expected a closed pod channel on timeout")
			}
			t.Errorf("Retrieving inserted pod from fifo took too long")
		}
	}
	fmt.Printf("\nCancelled reflector, ending test %+v", r.store)
}

func TestReflector_resyncChan(t *testing.T) {
	s := NewStore(MetaNamespaceKeyFunc)
	g := NewReflector(&testLW{}, &api.Pod{}, s, time.Millisecond)
	a, _ := g.RenewCancellationChan()
	b := time.After(100 * time.Millisecond)
	select {
	case <-a:
		t.Logf("got timeout as expected")
	case <-b:
		t.Errorf("the cancellation channel is at least 99 milliseconds late??")
	}
}

func TestReflector_watchHandlerError(t *testing.T) {
	s := NewStore(MetaNamespaceKeyFunc)
	g := NewReflector(&testLW{}, &api.Pod{}, s, 0)
	fw := watch.NewFake()
	go func() {
		fw.Stop()
	}()
	var resumeRV string
	err := g.watchHandler(fw, &resumeRV, neverExitWatch)
	if err == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestReflector_watchHandler(t *testing.T) {
	s := NewStore(MetaNamespaceKeyFunc)
	g := NewReflector(&testLW{}, &api.Pod{}, s, 0)
	fw := watch.NewFake()
	s.Add(&api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}})
	s.Add(&api.Pod{ObjectMeta: api.ObjectMeta{Name: "bar"}})
	go func() {
		fw.Add(&api.Service{ObjectMeta: api.ObjectMeta{Name: "rejected"}})
		fw.Delete(&api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}})
		fw.Modify(&api.Pod{ObjectMeta: api.ObjectMeta{Name: "bar", ResourceVersion: "55"}})
		fw.Add(&api.Pod{ObjectMeta: api.ObjectMeta{Name: "baz", ResourceVersion: "32"}})
		fw.Stop()
	}()
	var resumeRV string
	err := g.watchHandler(fw, &resumeRV, neverExitWatch)
	if err != nil {
		t.Errorf("unexpected error %v", err)
	}

	mkPod := func(id string, rv string) *api.Pod {
		return &api.Pod{ObjectMeta: api.ObjectMeta{Name: id, ResourceVersion: rv}}
	}

	table := []struct {
		Pod    *api.Pod
		exists bool
	}{
		{mkPod("foo", ""), false},
		{mkPod("rejected", ""), false},
		{mkPod("bar", "55"), true},
		{mkPod("baz", "32"), true},
	}
	for _, item := range table {
		obj, exists, _ := s.Get(item.Pod)
		if e, a := item.exists, exists; e != a {
			t.Errorf("%v: expected %v, got %v", item.Pod, e, a)
		}
		if !exists {
			continue
		}
		if e, a := item.Pod.ResourceVersion, obj.(*api.Pod).ResourceVersion; e != a {
			t.Errorf("%v: expected %v, got %v", item.Pod, e, a)
		}
	}

	// RV should send the last version we see.
	if e, a := "32", resumeRV; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	// last sync resource version should be the last version synced with store
	if e, a := "32", g.LastSyncResourceVersion(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
}

func TestReflector_watchHandlerTimeout(t *testing.T) {
	s := NewStore(MetaNamespaceKeyFunc)
	g := NewReflector(&testLW{}, &api.Pod{}, s, 0)
	fw := watch.NewFake()
	var resumeRV string
	exit := make(chan interface{}, 1)
	exit <- time.Now()
	err := g.watchHandler(fw, &resumeRV, exit)
	if err != errorResyncRequested {
		t.Errorf("expected timeout error, but got %q", err)
	}
}

func TestReflector_listAndWatch(t *testing.T) {
	createdFakes := make(chan *watch.FakeWatcher)

	// The ListFunc says that it's at revision 1. Therefore, we expect our WatchFunc
	// to get called at the beginning of the watch with 1, and again with 3 when we
	// inject an error.
	expectedRVs := []string{"1", "3"}
	lw := &testLW{
		WatchFunc: func(rv string) (watch.Interface, error) {
			fw := watch.NewFake()
			if e, a := expectedRVs[0], rv; e != a {
				t.Errorf("Expected rv %v, but got %v", e, a)
			}
			expectedRVs = expectedRVs[1:]
			// channel is not buffered because the for loop below needs to block. But
			// we don't want to block here, so report the new fake via a go routine.
			go func() { createdFakes <- fw }()
			return fw, nil
		},
		ListFunc: func() (runtime.Object, error) {
			return &api.PodList{ListMeta: api.ListMeta{ResourceVersion: "1"}}, nil
		},
	}
	s := NewFIFO(MetaNamespaceKeyFunc)
	r := NewReflector(lw, &api.Pod{}, s, 0)
	go r.listAndWatch()

	ids := []string{"foo", "bar", "baz", "qux", "zoo"}
	var fw *watch.FakeWatcher
	for i, id := range ids {
		if fw == nil {
			fw = <-createdFakes
		}
		sendingRV := strconv.FormatUint(uint64(i+2), 10)
		fw.Add(&api.Pod{ObjectMeta: api.ObjectMeta{Name: id, ResourceVersion: sendingRV}})
		if sendingRV == "3" {
			// Inject a failure.
			fw.Stop()
			fw = nil
		}
	}

	// Verify we received the right ids with the right resource versions.
	for i, id := range ids {
		pod := s.Pop().(*api.Pod)
		if e, a := id, pod.Name; e != a {
			t.Errorf("%v: Expected %v, got %v", i, e, a)
		}
		if e, a := strconv.FormatUint(uint64(i+2), 10), pod.ResourceVersion; e != a {
			t.Errorf("%v: Expected %v, got %v", i, e, a)
		}
	}

	if len(expectedRVs) != 0 {
		t.Error("called watchStarter an unexpected number of times")
	}
}

func TestReflector_listAndWatchWithErrors(t *testing.T) {
	mkPod := func(id string, rv string) *api.Pod {
		return &api.Pod{ObjectMeta: api.ObjectMeta{Name: id, ResourceVersion: rv}}
	}
	mkList := func(rv string, pods ...*api.Pod) *api.PodList {
		list := &api.PodList{ListMeta: api.ListMeta{ResourceVersion: rv}}
		for _, pod := range pods {
			list.Items = append(list.Items, *pod)
		}
		return list
	}
	table := []struct {
		list     *api.PodList
		listErr  error
		events   []watch.Event
		watchErr error
	}{
		{
			list: mkList("1"),
			events: []watch.Event{
				{watch.Added, mkPod("foo", "2")},
				{watch.Added, mkPod("bar", "3")},
			},
		}, {
			list: mkList("3", mkPod("foo", "2"), mkPod("bar", "3")),
			events: []watch.Event{
				{watch.Deleted, mkPod("foo", "4")},
				{watch.Added, mkPod("qux", "5")},
			},
		}, {
			listErr: fmt.Errorf("a list error"),
		}, {
			list:     mkList("5", mkPod("bar", "3"), mkPod("qux", "5")),
			watchErr: fmt.Errorf("a watch error"),
		}, {
			list: mkList("5", mkPod("bar", "3"), mkPod("qux", "5")),
			events: []watch.Event{
				{watch.Added, mkPod("baz", "6")},
			},
		}, {
			list: mkList("6", mkPod("bar", "3"), mkPod("qux", "5"), mkPod("baz", "6")),
		},
	}

	s := NewFIFO(MetaNamespaceKeyFunc)
	for line, item := range table {
		if item.list != nil {
			// Test that the list is what currently exists in the store.
			current := s.List()
			checkMap := map[string]string{}
			for _, item := range current {
				pod := item.(*api.Pod)
				checkMap[pod.Name] = pod.ResourceVersion
			}
			for _, pod := range item.list.Items {
				if e, a := pod.ResourceVersion, checkMap[pod.Name]; e != a {
					t.Errorf("%v: expected %v, got %v for pod %v", line, e, a, pod.Name)
				}
			}
			if e, a := len(item.list.Items), len(checkMap); e != a {
				t.Errorf("%v: expected %v, got %v", line, e, a)
			}
		}
		watchRet, watchErr := item.events, item.watchErr
		lw := &testLW{
			WatchFunc: func(rv string) (watch.Interface, error) {
				if watchErr != nil {
					return nil, watchErr
				}
				watchErr = fmt.Errorf("second watch")
				fw := watch.NewFake()
				go func() {
					for _, e := range watchRet {
						fw.Action(e.Type, e.Object)
					}
					fw.Stop()
				}()
				return fw, nil
			},
			ListFunc: func() (runtime.Object, error) {
				return item.list, item.listErr
			},
		}
		r := NewReflector(lw, &api.Pod{}, s, 0)
		r.listAndWatch()
	}
}
