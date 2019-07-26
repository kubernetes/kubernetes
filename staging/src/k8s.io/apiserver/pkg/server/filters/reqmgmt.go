/*
Copyright 2019 The Kubernetes Authors.

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

package filters

import (
	"fmt"
	"net/http"

	// TODO: decide whether to use the existing metrics, which
	// categorize according to mutating vs readonly, or make new
	// metrics because this filter does not pay attention to that
	// distinction

	// "k8s.io/apiserver/pkg/endpoints/metrics"

	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilflowcontrol "k8s.io/apiserver/pkg/util/flowcontrol"
	"k8s.io/klog"
)

// WithRequestManagement limits the number of in-flight
// requests in a fine-grained way and is more appropriate than
// WithRequestManagement for testing
func WithRequestManagement(
	handler http.Handler,
	longRunningRequestCheck apirequest.LongRunningRequestCheck,
	reqMgmt utilflowcontrol.Interface,
) http.Handler {
	if reqMgmt == nil {
		klog.Warningf("request management system not found, skipping setup flow-control system")
		return handler
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()
		requestInfo, ok := apirequest.RequestInfoFrom(ctx)
		if !ok {
			handleError(w, r, fmt.Errorf("no RequestInfo found in context"))
			return
		}
		user, ok := apirequest.UserFrom(ctx)
		if !ok {
			handleError(w, r, fmt.Errorf("no User found in context"))
			return
		}
		requestDigest := utilflowcontrol.RequestDigest{requestInfo, user}

		// Skip tracking long running events.
		if longRunningRequestCheck != nil && longRunningRequestCheck(r, requestInfo) {
			handler.ServeHTTP(w, r)
			return
		}

		for {
			rmState := reqMgmt.GetCurrentState()
			fs := utilflowcontrol.PickFlowSchema(requestDigest, rmState.GetFlowSchemas(), rmState.GetPriorityLevelStates())
			ps := utilflowcontrol.RequestPriorityState(requestDigest, fs, rmState.GetPriorityLevelStates())
			if ps.IsExempt() {
				klog.V(5).Infof("Serving %v without delay\n", r)
				handler.ServeHTTP(w, r)
				return
			}
			hashValue := utilflowcontrol.ComputeFlowDistinguisher(requestDigest, fs)
			quiescent, execute, afterExecute := ps.GetFairQueuingSystem().Wait(hashValue, ps.GetHandSize())
			if quiescent {
				klog.V(3).Infof("Request %v landed in timing splinter, re-classifying", r)
				continue
			}
			if execute {
				klog.V(5).Infof("Serving %v after queuing\n", r)
				timedOut := ctx.Done()
				finished := make(chan struct{})
				go func() {
					handler.ServeHTTP(w, r)
					close(finished)
				}()
				select {
				case <-timedOut:
					klog.V(5).Infof("Timed out waiting for %v to finish\n", r)
				case <-finished:
				}
				afterExecute()
			} else {
				klog.V(5).Infof("Rejecting %v\n", r)

				tooManyRequests(r, w)
			}
		}
		return
	})
}
