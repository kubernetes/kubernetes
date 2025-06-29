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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
func (ad *APIDispatcher) Add(incomingAPICall fwk.APICall, opts fwk.APICallOptions) error {
	apiCall := &queuedAPICall{
		APICall:   incomingAPICall,
		onFinish:  opts.OnFinish,
		timestamp: time.Now(),
	}
	return ad.callController.add(apiCall)
}

// UpdateObject applies pending changes from a queued API call to the given object,
// if one exists in the dispatcher, and returns the potentially modified object.
func (ad *APIDispatcher) UpdateObject(obj metav1.Object) (metav1.Object, error) {
	return ad.callController.updateObject(obj)
}

// Run starts the main processing loop of the APIDispatcher, which pops calls
// from the queue and dispatches them to worker goroutines for execution.
func (ad *APIDispatcher) Run(ctx context.Context) {
	go func() {
		logger := klog.FromContext(ctx)
		for {
			select {
			case <-ad.stop:
				// APIDispatcher is closed.
				return
			default:
			}

			// Acquire a goroutine before popping a call. This ordering prevents a popped
			// call from waiting (being in in-flight) for a long time.
			ok := ad.goroutinesLimiter.acquire()
			if !ok {
				// goroutinesLimiter is closed.
				return
			}
			apiCall, err := ad.callController.pop()
			if err != nil {
				logger.Error(err, "Popping API call from call controller failed")
				continue
			}
			if apiCall == nil {
				// callController is closed.
				return
			}

			ad.goroutinesLimiter.run(func() {
				err := apiCall.Execute(ctx, ad.client)
				apiCall.sendOnFinish(err)
				ad.callController.finalize(apiCall)
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
