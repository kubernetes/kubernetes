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
	"context"
	"fmt"
	"net/http"
	"runtime"
	"sync/atomic"
	"time"

	flowcontrol "k8s.io/api/flowcontrol/v1beta2"
	apitypes "k8s.io/apimachinery/pkg/types"
	epmetrics "k8s.io/apiserver/pkg/endpoints/metrics"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/server/httplog"
	utilflowcontrol "k8s.io/apiserver/pkg/util/flowcontrol"
	fcmetrics "k8s.io/apiserver/pkg/util/flowcontrol/metrics"
	flowcontrolrequest "k8s.io/apiserver/pkg/util/flowcontrol/request"
	"k8s.io/klog/v2"
)

// PriorityAndFairnessClassification identifies the results of
// classification for API Priority and Fairness
type PriorityAndFairnessClassification struct {
	FlowSchemaName    string
	FlowSchemaUID     apitypes.UID
	PriorityLevelName string
	PriorityLevelUID  apitypes.UID
}

// waitingMark tracks requests waiting rather than being executed
var waitingMark = &requestWatermark{
	phase:            epmetrics.WaitingPhase,
	readOnlyObserver: fcmetrics.ReadWriteConcurrencyObserverPairGenerator.Generate(1, 1, []string{epmetrics.ReadOnlyKind}).RequestsWaiting,
	mutatingObserver: fcmetrics.ReadWriteConcurrencyObserverPairGenerator.Generate(1, 1, []string{epmetrics.MutatingKind}).RequestsWaiting,
}

var atomicMutatingExecuting, atomicReadOnlyExecuting int32
var atomicMutatingWaiting, atomicReadOnlyWaiting int32

// newInitializationSignal is defined for testing purposes.
var newInitializationSignal = utilflowcontrol.NewInitializationSignal

func truncateLogField(s string) string {
	const maxFieldLogLength = 64

	if len(s) > maxFieldLogLength {
		s = s[0:maxFieldLogLength]
	}
	return s
}

// WithPriorityAndFairness limits the number of in-flight
// requests in a fine-grained way.
func WithPriorityAndFairness(
	handler http.Handler,
	longRunningRequestCheck apirequest.LongRunningRequestCheck,
	fcIfc utilflowcontrol.Interface,
	workEstimator flowcontrolrequest.WorkEstimatorFunc,
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

		isWatchRequest := watchVerbs.Has(requestInfo.Verb)

		// Skip tracking long running non-watch requests.
		if longRunningRequestCheck != nil && longRunningRequestCheck(r, requestInfo) && !isWatchRequest {
			klog.V(6).Infof("Serving RequestInfo=%#+v, user.Info=%#+v as longrunning\n", requestInfo, user)
			handler.ServeHTTP(w, r)
			return
		}

		var classification *PriorityAndFairnessClassification
		noteFn := func(fs *flowcontrol.FlowSchema, pl *flowcontrol.PriorityLevelConfiguration, flowDistinguisher string) {
			classification = &PriorityAndFairnessClassification{
				FlowSchemaName:    fs.Name,
				FlowSchemaUID:     fs.UID,
				PriorityLevelName: pl.Name,
				PriorityLevelUID:  pl.UID}

			httplog.AddKeyValue(ctx, "apf_pl", truncateLogField(pl.Name))
			httplog.AddKeyValue(ctx, "apf_fs", truncateLogField(fs.Name))
			httplog.AddKeyValue(ctx, "apf_fd", truncateLogField(flowDistinguisher))
		}
		// estimateWork is called, if at all, after noteFn
		estimateWork := func() flowcontrolrequest.WorkEstimate {
			if classification == nil {
				// workEstimator is being invoked before classification of
				// the request has completed, we should never be here though.
				klog.ErrorS(fmt.Errorf("workEstimator is being invoked before classification of the request has completed"),
					"Using empty FlowSchema and PriorityLevelConfiguration name", "verb", r.Method, "URI", r.RequestURI)

				return workEstimator(r, "", "")
			}

			return workEstimator(r, classification.FlowSchemaName, classification.PriorityLevelName)
		}

		var served bool
		isMutatingRequest := !nonMutatingRequestVerbs.Has(requestInfo.Verb)
		noteExecutingDelta := func(delta int32) {
			if isMutatingRequest {
				watermark.recordMutating(int(atomic.AddInt32(&atomicMutatingExecuting, delta)))
			} else {
				watermark.recordReadOnly(int(atomic.AddInt32(&atomicReadOnlyExecuting, delta)))
			}
		}
		noteWaitingDelta := func(delta int32) {
			if isMutatingRequest {
				waitingMark.recordMutating(int(atomic.AddInt32(&atomicMutatingWaiting, delta)))
			} else {
				waitingMark.recordReadOnly(int(atomic.AddInt32(&atomicReadOnlyWaiting, delta)))
			}
		}
		queueNote := func(inQueue bool) {
			if inQueue {
				noteWaitingDelta(1)
			} else {
				noteWaitingDelta(-1)
			}
		}

		digest := utilflowcontrol.RequestDigest{
			RequestInfo: requestInfo,
			User:        user,
		}

		if isWatchRequest {
			// This channel blocks calling handler.ServeHTTP() until closed, and is closed inside execute().
			// If APF rejects the request, it is never closed.
			shouldStartWatchCh := make(chan struct{})

			watchInitializationSignal := newInitializationSignal()
			// This wraps the request passed to handler.ServeHTTP(),
			// setting a context that plumbs watchInitializationSignal to storage
			var watchReq *http.Request
			// This is set inside execute(), prior to closing shouldStartWatchCh.
			// If the request is rejected by APF it is left nil.
			var forgetWatch utilflowcontrol.ForgetWatchFunc

			defer func() {
				// Protect from the situation when request will not reach storage layer
				// and the initialization signal will not be send.
				if watchInitializationSignal != nil {
					watchInitializationSignal.Signal()
				}
				// Forget the watcher if it was registered.
				//
				// // This is race-free because by this point, one of the following occurred:
				// case <-shouldStartWatchCh: execute() completed the assignment to forgetWatch
				// case <-resultCh: Handle() completed, and Handle() does not return
				//   while execute() is running
				if forgetWatch != nil {
					forgetWatch()
				}
			}()

			execute := func() {
				startedAt := time.Now()
				defer func() {
					httplog.AddKeyValue(ctx, "apf_init_latency", time.Since(startedAt))
				}()
				noteExecutingDelta(1)
				defer noteExecutingDelta(-1)
				served = true
				setResponseHeaders(classification, w)

				forgetWatch = fcIfc.RegisterWatch(r)

				// Notify the main thread that we're ready to start the watch.
				close(shouldStartWatchCh)

				// Wait until the request is finished from the APF point of view
				// (which is when its initialization is done).
				watchInitializationSignal.Wait()
			}

			// Ensure that an item can be put to resultCh asynchronously.
			resultCh := make(chan interface{}, 1)

			// Call Handle in a separate goroutine.
			// The reason for it is that from APF point of view, the request processing
			// finishes as soon as watch is initialized (which is generally orders of
			// magnitude faster then the watch request itself). This means that Handle()
			// call finishes much faster and for performance reasons we want to reduce
			// the number of running goroutines - so we run the shorter thing in a
			// dedicated goroutine and the actual watch handler in the main one.
			go func() {
				defer func() {
					err := recover()
					// do not wrap the sentinel ErrAbortHandler panic value
					if err != nil && err != http.ErrAbortHandler {
						// Same as stdlib http server code. Manually allocate stack
						// trace buffer size to prevent excessively large logs
						const size = 64 << 10
						buf := make([]byte, size)
						buf = buf[:runtime.Stack(buf, false)]
						err = fmt.Sprintf("%v\n%s", err, buf)
					}

					// Ensure that the result is put into resultCh independently of the panic.
					resultCh <- err
				}()

				// We create handleCtx with explicit cancelation function.
				// The reason for it is that Handle() underneath may start additional goroutine
				// that is blocked on context cancellation. However, from APF point of view,
				// we don't want to wait until the whole watch request is processed (which is
				// when it context is actually cancelled) - we want to unblock the goroutine as
				// soon as the request is processed from the APF point of view.
				//
				// Note that we explicitly do NOT call the actuall handler using that context
				// to avoid cancelling request too early.
				handleCtx, handleCtxCancel := context.WithCancel(ctx)
				defer handleCtxCancel()

				// Note that Handle will return irrespective of whether the request
				// executes or is rejected. In the latter case, the function will return
				// without calling the passed `execute` function.
				fcIfc.Handle(handleCtx, digest, noteFn, estimateWork, queueNote, execute)
			}()

			select {
			case <-shouldStartWatchCh:
				watchCtx := utilflowcontrol.WithInitializationSignal(ctx, watchInitializationSignal)
				watchReq = r.WithContext(watchCtx)
				handler.ServeHTTP(w, watchReq)
				// Protect from the situation when request will not reach storage layer
				// and the initialization signal will not be send.
				// It has to happen before waiting on the resultCh below.
				watchInitializationSignal.Signal()
				// TODO: Consider finishing the request as soon as Handle call panics.
				if err := <-resultCh; err != nil {
					panic(err)
				}
			case err := <-resultCh:
				if err != nil {
					panic(err)
				}
			}
		} else {
			execute := func() {
				noteExecutingDelta(1)
				defer noteExecutingDelta(-1)
				served = true
				setResponseHeaders(classification, w)

				handler.ServeHTTP(w, r)
			}

			fcIfc.Handle(ctx, digest, noteFn, estimateWork, queueNote, execute)
		}

		if !served {
			setResponseHeaders(classification, w)

			if isMutatingRequest {
				epmetrics.DroppedRequests.WithContext(ctx).WithLabelValues(epmetrics.MutatingKind).Inc()
			} else {
				epmetrics.DroppedRequests.WithContext(ctx).WithLabelValues(epmetrics.ReadOnlyKind).Inc()
			}
			epmetrics.RecordRequestTermination(r, requestInfo, epmetrics.APIServerComponent, http.StatusTooManyRequests)
			tooManyRequests(r, w)
		}
	})
}

// StartPriorityAndFairnessWatermarkMaintenance starts the goroutines to observe and maintain watermarks for
// priority-and-fairness requests.
func StartPriorityAndFairnessWatermarkMaintenance(stopCh <-chan struct{}) {
	startWatermarkMaintenance(watermark, stopCh)
	startWatermarkMaintenance(waitingMark, stopCh)
}

func setResponseHeaders(classification *PriorityAndFairnessClassification, w http.ResponseWriter) {
	if classification == nil {
		return
	}

	// We intentionally set the UID of the flow-schema and priority-level instead of name. This is so that
	// the names that cluster-admins choose for categorization and priority levels are not exposed, also
	// the names might make it obvious to the users that they are rejected due to classification with low priority.
	w.Header().Set(flowcontrol.ResponseHeaderMatchedPriorityLevelConfigurationUID, string(classification.PriorityLevelUID))
	w.Header().Set(flowcontrol.ResponseHeaderMatchedFlowSchemaUID, string(classification.FlowSchemaUID))
}
