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
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
)

// APIDispatcher implements the fwk.APIDispatcher interface and allows for queueing and dispatching API calls asynchronously.
type APIDispatcher struct {
	cancel func()

	client    clientset.Interface
	callQueue *callQueue
}

// New returns a new APIDispatcher object.
func New(client clientset.Interface, parallelization int, apiCallRelevances fwk.APICallRelevances) *APIDispatcher {
	d := APIDispatcher{
		client:    client,
		callQueue: newCallQueue(apiCallRelevances),
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
	return ad.callQueue.add(apiCall)
}

// SyncObject performs a two-way synchronization between the given object
// and a pending API call held within the dispatcher and returns the modified object.
func (ad *APIDispatcher) SyncObject(obj metav1.Object) (metav1.Object, error) {
	return ad.callQueue.syncObject(obj)
}

// Run starts the main processing loop of the APIDispatcher, which pops calls
// from the queue and dispatches them to worker goroutines for execution.
func (ad *APIDispatcher) Run(logger klog.Logger) {
	// Create a new context to allow to cancel the APICalls' execution when the APIDispatcher is closed.
	ctx, cancel := context.WithCancel(context.Background())
	ad.cancel = cancel

	go func() {
		for {
			select {
			case <-ctx.Done():
				// APIDispatcher is closed.
				return
			default:
			}

			apiCall, err := ad.callQueue.pop()
			if err != nil {
				utilruntime.HandleErrorWithLogger(logger, err, "popping API call from call controller failed")
				continue
			}
			if apiCall == nil {
				// callController is closed.
				return
			}

			go func() {
				startTime := time.Now()

				err := apiCall.Execute(ctx, ad.client)

				result := metrics.GoroutineResultSuccess
				if err != nil {
					result = metrics.GoroutineResultError
				}
				callType := string(apiCall.CallType())
				metrics.AsyncAPICallsTotal.WithLabelValues(callType, result).Inc()
				metrics.AsyncAPICallDuration.WithLabelValues(callType, result).Observe(time.Since(startTime).Seconds())

				ad.callQueue.finalize(apiCall)
				apiCall.sendOnFinish(err)
			}()
		}
	}()
}

// Close shuts down the APIDispatcher.
func (ad *APIDispatcher) Close() {
	ad.callQueue.close()
	if ad.cancel != nil {
		ad.cancel()
	}
}
