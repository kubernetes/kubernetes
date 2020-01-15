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
	"runtime"

	// TODO: decide whether to use the existing metrics, which
	// categorize according to mutating vs readonly, or make new
	// metrics because this filter does not pay attention to that
	// distinction

	// "k8s.io/apiserver/pkg/endpoints/metrics"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilflowcontrol "k8s.io/apiserver/pkg/util/flowcontrol"
	utilfilterconfig "k8s.io/apiserver/pkg/util/flowcontrol/filterconfig"
	"k8s.io/klog"
)

// WithPriorityAndFairness limits the number of in-flight
// requests in a fine-grained way.
func WithPriorityAndFairness(
	handler http.Handler,
	longRunningRequestCheck apirequest.LongRunningRequestCheck,
	fcIfc utilflowcontrol.Interface,
) http.Handler {
	if fcIfc == nil {
		klog.Warningf("priority and fairness support not found, skipping")
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
		requestDigest := utilfilterconfig.RequestDigest{requestInfo, user}

		// Skip tracking long running requests.
		if longRunningRequestCheck != nil && longRunningRequestCheck(r, requestInfo) {
			klog.V(6).Infof("Serving RequestInfo=%#+v, user.Info=%#+v as longrunning\n", requestInfo, user)
			handler.ServeHTTP(w, r)
			return
		}

		execute, afterExecute := fcIfc.Wait(ctx, requestDigest)
		if !execute {
			tooManyRequests(r, w)
			return
		}
		defer afterExecute()
		
		// Serve the request, but asynchronously, and return from here
		// as soon as either the request is finished or the context is
		// canceled.  The logic here is heavily cribbed from the
		// timeout filter.
		timedOut := ctx.Done()
		
		// gets sent to exactly once, with the result of recover() for serving the request
		resultCh := make(chan interface{})
		
		go func() { // serve the request, with recovery from panics
			defer func() {
				err := recover()
				if err != nil && err != http.ErrAbortHandler {
					// Same as stdlib http server code. Manually allocate stack
					// trace buffer size to prevent excessively large logs
					const size = 64 << 10
					buf := make([]byte, size)
					buf = buf[:runtime.Stack(buf, false)]
					err = fmt.Sprintf("%v\n%s", err, buf)
				}
				resultCh <- err
			}()
			handler.ServeHTTP(w, r)
		}()
		
		select {
		case err := <-resultCh:
			if err != nil {
				panic(err)
			}
		case <-timedOut:
			// Satisfy the need for a receive from resultCh
			go func() {
				err := <-resultCh
				if err != nil {
					switch t := err.(type) {
					case error:
						utilruntime.HandleError(t)
					default:
						utilruntime.HandleError(fmt.Errorf("%v", err))
					}
				}
			}()
			klog.V(6).Infof("Timed out waiting for RequestInfo=%#+v, user.Info=%#+v to finish\n", requestInfo, user)
		}
		return
	})
}
