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

package framework_test

import (
	"fmt"
	"math/rand"
	"sync"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"

	"github.com/google/gofuzz"
)

func Example() {
	// source simulates an apiserver object endpoint.
	source := framework.NewFakeControllerSource()

	// This will hold the downstream state, as we know it.
	downstream := cache.NewStore(framework.DeletionHandlingMetaNamespaceKeyFunc)

	// This will hold incoming changes. Note how we pass downstream in as a
	// KeyLister, that way resync operations will result in the correct set
	// of update/delete deltas.
	fifo := cache.NewDeltaFIFO(cache.MetaNamespaceKeyFunc, nil, downstream)

	// Let's do threadsafe output to get predictable test results.
	deletionCounter := make(chan string, 1000)

	cfg := &framework.Config{
		Queue:            fifo,
		ListerWatcher:    source,
		ObjectType:       &api.Pod{},
		FullResyncPeriod: time.Millisecond * 100,
		RetryOnError:     false,

		// Let's implement a simple controller that just deletes
		// everything that comes in.
		Process: func(obj interface{}) error {
			// Obj is from the Pop method of the Queue we make above.
			newest := obj.(cache.Deltas).Newest()

			if newest.Type != cache.Deleted {
				// Update our downstream store.
				err := downstream.Add(newest.Object)
				if err != nil {
					return err
				}

				// Delete this object.
				source.Delete(newest.Object.(runtime.Object))
			} else {
				// Update our downstream store.
				err := downstream.Delete(newest.Object)
				if err != nil {
					return err
				}

				// fifo's KeyOf is easiest, because it handles
				// DeletedFinalStateUnknown markers.
				key, err := fifo.KeyOf(newest.Object)
				if err != nil {
					return err
				}

				// Report this deletion.
				deletionCounter <- key
			}
			return nil
		},
	}

	// Create the controller and run it until we close stop.
	stop := make(chan struct{})
	defer close(stop)
	go framework.New(cfg).Run(stop)

	// Let's add a few objects to the source.
	testIDs := []string{"a-hello", "b-controller", "c-framework"}
	for _, name := range testIDs {
		// Note that these pods are not valid-- the fake source doesn't
		// call validation or anything.
		source.Add(&api.Pod{ObjectMeta: api.ObjectMeta{Name: name}})
	}

	// Let's wait for the controller to process the things we just added.
	outputSet := sets.String{}
	for i := 0; i < len(testIDs); i++ {
		outputSet.Insert(<-deletionCounter)
	}

	for _, key := range outputSet.List() {
		fmt.Println(key)
	}
	// Output:
	// a-hello
	// b-controller
	// c-framework
}

func ExampleInformer() {
	// source simulates an apiserver object endpoint.
	source := framework.NewFakeControllerSource()

	// Let's do threadsafe output to get predictable test results.
	deletionCounter := make(chan string, 1000)

	// Make a controller that immediately deletes anything added to it, and
	// logs anything deleted.
	_, controller := framework.NewInformer(
		source,
		&api.Pod{},
		time.Millisecond*100,
		framework.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				source.Delete(obj.(runtime.Object))
			},
			DeleteFunc: func(obj interface{}) {
				key, err := framework.DeletionHandlingMetaNamespaceKeyFunc(obj)
				if err != nil {
					key = "oops something went wrong with the key"
				}

				// Report this deletion.
				deletionCounter <- key
			},
		},
	)

	// Run the controller and run it until we close stop.
	stop := make(chan struct{})
	defer close(stop)
	go controller.Run(stop)

	// Let's add a few objects to the source.
	testIDs := []string{"a-hello", "b-controller", "c-framework"}
	for _, name := range testIDs {
		// Note that these pods are not valid-- the fake source doesn't
		// call validation or anything.
		source.Add(&api.Pod{ObjectMeta: api.ObjectMeta{Name: name}})
	}

	// Let's wait for the controller to process the things we just added.
	outputSet := sets.String{}
	for i := 0; i < len(testIDs); i++ {
		outputSet.Insert(<-deletionCounter)
	}

	for _, key := range outputSet.List() {
		fmt.Println(key)
	}
	// Output:
	// a-hello
	// b-controller
	// c-framework
}

func TestHammerController(t *testing.T) {
	// This test executes a bunch of requests through the fake source and
	// controller framework to make sure there's no locking/threading
	// errors. If an error happens, it should hang forever or trigger the
	// race detector.

	// source simulates an apiserver object endpoint.
	source := framework.NewFakeControllerSource()

	// Let's do threadsafe output to get predictable test results.
	outputSetLock := sync.Mutex{}
	// map of key to operations done on the key
	outputSet := map[string][]string{}

	recordFunc := func(eventType string, obj interface{}) {
		key, err := framework.DeletionHandlingMetaNamespaceKeyFunc(obj)
		if err != nil {
			t.Errorf("something wrong with key: %v", err)
			key = "oops something went wrong with the key"
		}

		// Record some output when items are deleted.
		outputSetLock.Lock()
		defer outputSetLock.Unlock()
		outputSet[key] = append(outputSet[key], eventType)
	}

	// Make a controller which just logs all the changes it gets.
	_, controller := framework.NewInformer(
		source,
		&api.Pod{},
		time.Millisecond*100,
		framework.ResourceEventHandlerFuncs{
			AddFunc:    func(obj interface{}) { recordFunc("add", obj) },
			UpdateFunc: func(oldObj, newObj interface{}) { recordFunc("update", newObj) },
			DeleteFunc: func(obj interface{}) { recordFunc("delete", obj) },
		},
	)

	if controller.HasSynced() {
		t.Errorf("Expected HasSynced() to return false before we started the controller")
	}

	// Run the controller and run it until we close stop.
	stop := make(chan struct{})
	go controller.Run(stop)

	// Let's wait for the controller to do its initial sync
	time.Sleep(100 * time.Millisecond)
	if !controller.HasSynced() {
		t.Errorf("Expected HasSynced() to return true after the initial sync")
	}

	wg := sync.WaitGroup{}
	const threads = 3
	wg.Add(threads)
	for i := 0; i < threads; i++ {
		go func() {
			defer wg.Done()
			// Let's add a few objects to the source.
			currentNames := sets.String{}
			rs := rand.NewSource(rand.Int63())
			f := fuzz.New().NilChance(.5).NumElements(0, 2).RandSource(rs)
			r := rand.New(rs) // Mustn't use r and f concurrently!
			for i := 0; i < 100; i++ {
				var name string
				var isNew bool
				if currentNames.Len() == 0 || r.Intn(3) == 1 {
					f.Fuzz(&name)
					isNew = true
				} else {
					l := currentNames.List()
					name = l[r.Intn(len(l))]
				}

				pod := &api.Pod{}
				f.Fuzz(pod)
				pod.ObjectMeta.Name = name
				pod.ObjectMeta.Namespace = "default"
				// Add, update, or delete randomly.
				// Note that these pods are not valid-- the fake source doesn't
				// call validation or perform any other checking.
				if isNew {
					currentNames.Insert(name)
					source.Add(pod)
					continue
				}
				switch r.Intn(2) {
				case 0:
					currentNames.Insert(name)
					source.Modify(pod)
				case 1:
					currentNames.Delete(name)
					source.Delete(pod)
				}
			}
		}()
	}
	wg.Wait()

	// Let's wait for the controller to finish processing the things we just added.
	time.Sleep(100 * time.Millisecond)
	close(stop)

	outputSetLock.Lock()
	t.Logf("got: %#v", outputSet)
}

func TestUpdate(t *testing.T) {
	// This test is going to exercise the various paths that result in a
	// call to update.

	// source simulates an apiserver object endpoint.
	source := framework.NewFakeControllerSource()

	const (
		FROM       = "from"
		ADD_MISSED = "missed the add event"
		TO         = "to"
	)

	// These are the transitions we expect to see; because this is
	// asynchronous, there are a lot of valid possibilities.
	type pair struct{ from, to string }
	allowedTransitions := map[pair]bool{
		pair{FROM, TO}:         true,
		pair{FROM, ADD_MISSED}: true,
		pair{ADD_MISSED, TO}:   true,

		// Because a resync can happen when we've already observed one
		// of the above but before the item is deleted.
		pair{TO, TO}: true,
		// Because a resync could happen before we observe an update.
		pair{FROM, FROM}: true,
	}

	var testDoneWG sync.WaitGroup

	// Make a controller that deletes things once it observes an update.
	// It calls Done() on the wait group on deletions so we can tell when
	// everything we've added has been deleted.
	_, controller := framework.NewInformer(
		source,
		&api.Pod{},
		time.Millisecond*1,
		framework.ResourceEventHandlerFuncs{
			UpdateFunc: func(oldObj, newObj interface{}) {
				o, n := oldObj.(*api.Pod), newObj.(*api.Pod)
				from, to := o.Labels["check"], n.Labels["check"]
				if !allowedTransitions[pair{from, to}] {
					t.Errorf("observed transition %q -> %q for %v", from, to, n.Name)
				}
				source.Delete(n)
			},
			DeleteFunc: func(obj interface{}) {
				testDoneWG.Done()
			},
		},
	)

	pod := func(name, check string) *api.Pod {
		return &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:   name,
				Labels: map[string]string{"check": check},
			},
		}
	}

	tests := []func(string){
		func(name string) {
			name = "a-" + name
			source.Add(pod(name, FROM))
			source.Modify(pod(name, TO))
		},
		func(name string) {
			name = "b-" + name
			source.Add(pod(name, FROM))
			source.ModifyDropWatch(pod(name, TO))
		},
		func(name string) {
			name = "c-" + name
			source.AddDropWatch(pod(name, FROM))
			source.Modify(pod(name, ADD_MISSED))
			source.Modify(pod(name, TO))
		},
		func(name string) {
			name = "d-" + name
			source.Add(pod(name, FROM))
		},
	}

	const threads = 3
	testDoneWG.Add(threads * len(tests))

	// Run the controller and run it until we close stop.
	// Once Run() is called, calls to testDoneWG.Done() might start, so
	// all testDoneWG.Add() calls must happen before this point
	stop := make(chan struct{})
	go controller.Run(stop)

	// run every test a few times, in parallel
	var wg sync.WaitGroup
	wg.Add(threads * len(tests))
	for i := 0; i < threads; i++ {
		for j, f := range tests {
			go func(name string, f func(string)) {
				defer wg.Done()
				f(name)
			}(fmt.Sprintf("%v-%v", i, j), f)
		}
	}
	wg.Wait()

	// Let's wait for the controller to process the things we just added.
	testDoneWG.Wait()
	close(stop)
}
