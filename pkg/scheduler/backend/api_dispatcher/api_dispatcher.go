/*
Copyright 2025 The Kubernetes Authors.

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

package apidispatcher

import (
	"context"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
)

// APIDispatcher implements the fwk.APIDispatcher interface and allows for queueing and dispatching API calls asynchronously.
type APIDispatcher struct {
	stop chan struct{}

	client            clientset.Interface
	callController    *callController
	goroutinesLimiter *goroutinesLimiter
}

// New returns a new APIDispatcher object.
func New(client clientset.Interface, parallelization int, apiCallRelevances fwk.APICallRelevances) *APIDispatcher {
	d := APIDispatcher{
		stop:              make(chan struct{}),
		client:            client,
		callController:    newCallController(apiCallRelevances),
		goroutinesLimiter: newGoroutinesLimiter(parallelization),
	}

	return &d
}

// Add adds an API call to the dispatcher's queue. It returns an error if the call is not enqueued
// (e.g., if it's skipped). The caller should handle ErrCallSkipped if returned.
func (ad *APIDispatcher) Add(newAPICall fwk.APICall, opts fwk.APICallOptions) error {
	apiCall := &queuedAPICall{
		APICall:  newAPICall,
		onFinish: opts.OnFinish,
	}
	return ad.callController.add(apiCall)
}

// SyncObject performs a two-way synchronization between the given object
// and a pending API call held within the dispatcher and returns the modified object.
func (ad *APIDispatcher) SyncObject(obj metav1.Object) (metav1.Object, error) {
	return ad.callController.syncObject(obj)
}

// Run starts the main processing loop of the APIDispatcher, which pops calls
// from the queue and dispatches them to worker goroutines for execution.
func (ad *APIDispatcher) Run(logger klog.Logger) {
	go func() {
		// Create a new context to allow to cancel the APICalls' execution when the APIDispatcher is closed.
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		for {
			select {
			case <-ad.stop:
				// APIDispatcher is closed.
				return
			default:
			}

			// Acquire a goroutine before popping a call. This ordering prevents a popped
			// call from waiting (being in in-flight) for a long time.
			runner := ad.goroutinesLimiter.acquire()
			if runner == nil {
				// goroutinesLimiter is closed.
				return
			}
			apiCall, err := ad.callController.pop()
			if err != nil {
				utilruntime.HandleErrorWithLogger(logger, err, "popping API call from call controller failed")
				ad.goroutinesLimiter.release()
				continue
			}
			if apiCall == nil {
				// callController is closed.
				ad.goroutinesLimiter.release()
				return
			}

			runner.run(func() {
				err := apiCall.Execute(ctx, ad.client)
				ad.callController.finalize(apiCall)
				apiCall.sendOnFinish(err)
			})
		}
	}()
}

// Close shuts down the APIDispatcher.
func (ad *APIDispatcher) Close() {
	close(ad.stop)
	ad.callController.close()
	ad.goroutinesLimiter.close()
}
