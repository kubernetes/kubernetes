/*
Copyright 2015 Google Inc. All rights reserved.

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
	"sync"
	"time"
	// "testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/controller/framework"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func Example() {
	// source simulates an apiserver object endpoint.
	source := framework.NewFakeControllerSource()

	// This will hold the downstream state, as we know it.
	downstream := cache.NewStore(cache.MetaNamespaceKeyFunc)

	// This will hold incoming changes. Note how we pass downstream in as a
	// KeyLister, that way resync operations will result in the correct set
	// of update/delete deltas.
	fifo := cache.NewDeltaFIFO(cache.MetaNamespaceKeyFunc, nil, downstream)

	// Let's do threadsafe output to get predictable test results.
	outputSetLock := sync.Mutex{}
	outputSet := util.StringSet{}

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

				// Record some output.
				outputSetLock.Lock()
				defer outputSetLock.Unlock()
				outputSet.Insert(key)
			}
			return nil
		},
	}

	// Create the controller and run it until we close stop.
	stop := make(chan struct{})
	framework.New(cfg).Run(stop)

	// Let's add a few objects to the source.
	for _, name := range []string{"a-hello", "b-controller", "c-framework"} {
		// Note that these pods are not valid-- the fake source doesn't
		// call validation or anything.
		source.Add(&api.Pod{ObjectMeta: api.ObjectMeta{Name: name}})
	}

	// Let's wait for the controller to process the things we just added.
	time.Sleep(500 * time.Millisecond)
	close(stop)

	outputSetLock.Lock()
	for _, key := range outputSet.List() {
		fmt.Println(key)
	}
	// Output:
	// a-hello
	// b-controller
	// c-framework
}
