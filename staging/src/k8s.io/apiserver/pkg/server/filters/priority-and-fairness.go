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
	"sync/atomic"

	flowcontrol "k8s.io/api/flowcontrol/v1beta1"
	apitypes "k8s.io/apimachinery/pkg/types"
	epmetrics "k8s.io/apiserver/pkg/endpoints/metrics"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
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

// WithPriorityAndFairness limits the number of in-flight
// requests in a fine-grained way.
func WithPriorityAndFairness(
	handler http.Handler,
	longRunningRequestCheck apirequest.LongRunningRequestCheck,
	fcIfc utilflowcontrol.Interface,
	widthEstimator flowcontrolrequest.WidthEstimatorFunc,
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
		note := func(fs *flowcontrol.FlowSchema, pl *flowcontrol.PriorityLevelConfiguration) {
			classification = &PriorityAndFairnessClassification{
				FlowSchemaName:    fs.Name,
				FlowSchemaUID:     fs.UID,
				PriorityLevelName: pl.Name,
				PriorityLevelUID:  pl.UID}
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
		var resultCh chan interface{}
		if isWatchRequest {
			resultCh = make(chan interface{})
		}
		execute := func() {
			noteExecutingDelta(1)
			defer noteExecutingDelta(-1)
			served = true

			innerCtx := ctx
			innerReq := r

			var watchInitializationSignal utilflowcontrol.InitializationSignal
			if isWatchRequest {
				watchInitializationSignal = newInitializationSignal()
				innerCtx = utilflowcontrol.WithInitializationSignal(ctx, watchInitializationSignal)
				innerReq = r.Clone(innerCtx)
			}
			setResponseHeaders(classification, w)

			if isWatchRequest {
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
						resultCh <- err
					}()

					// Protect from the situations when request will not reach storage layer
					// and the initialization signal will not be send.
					defer watchInitializationSignal.Signal()

					handler.ServeHTTP(w, innerReq)
				}()

				watchInitializationSignal.Wait()
			} else {
				handler.ServeHTTP(w, innerReq)
			}
		}

		// find the estimated "width" of the request
		width := widthEstimator.EstimateWidth(r)
		digest := utilflowcontrol.RequestDigest{RequestInfo: requestInfo, User: user, Width: width}

		fcIfc.Handle(ctx, digest, note, func(inQueue bool) {
			if inQueue {
				noteWaitingDelta(1)
			} else {
				noteWaitingDelta(-1)
			}
		}, execute)
		if !served {
			setResponseHeaders(classification, w)

			if isMutatingRequest {
				epmetrics.DroppedRequests.WithContext(ctx).WithLabelValues(epmetrics.MutatingKind).Inc()
			} else {
				epmetrics.DroppedRequests.WithContext(ctx).WithLabelValues(epmetrics.ReadOnlyKind).Inc()
			}
			epmetrics.RecordRequestTermination(r, requestInfo, epmetrics.APIServerComponent, http.StatusTooManyRequests)
			if isWatchRequest {
				close(resultCh)
			}
			tooManyRequests(r, w)
		}

		// For watch requests, from the APF point of view the request is already
		// finished at this point. However, that doesn't mean it is already finished
		// from the non-APF point of view. So we need to wait here until the request is:
		// 1) finished being processed or
		// 2) rejected
		if isWatchRequest {
			err := <-resultCh
			if err != nil {
				panic(err)
			}
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
