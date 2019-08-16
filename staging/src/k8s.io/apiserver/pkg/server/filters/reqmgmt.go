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
// requests in a fine-grained way.
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
			klog.V(6).Infof("Serving RequestInfo=%#+v, user.Info=%#+v as longrunning\n", requestInfo, user)
			handler.ServeHTTP(w, r)
			return
		}

		execute, afterExecute := reqMgmt.Wait(requestDigest)
		if execute {
			klog.V(6).Infof("Serving RequestInfo=%#+v, user.Info=%#+v after queuing\n", requestInfo, user)
			timedOut := ctx.Done()
			finished := make(chan struct{})
			go func() {
				handler.ServeHTTP(w, r)
				close(finished)
			}()
			select {
			case <-timedOut:
				klog.V(6).Infof("Timed out waiting for RequestInfo=%#+v, user.Info=%#+v to finish\n", requestInfo, user)
			case <-finished:
			}
			afterExecute()
		} else {
			klog.V(6).Infof("Rejecting RequestInfo=%#+v, user.Info=%#+v\n", requestInfo, user)

			tooManyRequests(r, w)
		}
		return
	})
}
