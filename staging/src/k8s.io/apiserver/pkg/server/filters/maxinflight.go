/*
Copyright 2016 The Kubernetes Authors.

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
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/metrics"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	fcmetrics "k8s.io/apiserver/pkg/util/flowcontrol/metrics"

	"k8s.io/klog/v2"
)

const (
	// Constant for the retry-after interval on rate limiting.
	// TODO: maybe make this dynamic? or user-adjustable?
	retryAfter = "1"

	// How often inflight usage metric should be updated. Because
	// the metrics tracks maximal value over period making this
	// longer will increase the metric value.
	inflightUsageMetricUpdatePeriod = time.Second
)

var (
	nonMutatingRequestVerbs = sets.NewString("get", "list", "watch")
	watchVerbs              = sets.NewString("watch")
)

func handleError(w http.ResponseWriter, r *http.Request, err error) {
	errorMsg := fmt.Sprintf("Internal Server Error: %#v", r.RequestURI)
	http.Error(w, errorMsg, http.StatusInternalServerError)
	klog.Errorf(err.Error())
}

// requestWatermark is used to track current and maximal numbers of
// requests in a particular phase of handling, broken down by readonly
// vs mutating.
type requestWatermark struct {
	phase         string
	observersFunc func(readonlyLimit, mutatingLimit int) (readOnlyObserver, mutatingObserver fcmetrics.RatioedObserver)

	// Filling in the observers is not done at package
	// initialization time because that happens before the
	// relevant metric vectors are registered, which means
	// extracting a vector's member during initialization of this
	// package would capture a permanent noop.  We expect actual
	// request handling follows metric vector registration, so it
	// is safe to extract an efficient vector member when handling
	// the first request.
	initializeOnce sync.Once

	readOnlyObserver, mutatingObserver fcmetrics.RatioedObserver // observes current values

	watermarksLock                       sync.Mutex // for the following
	readOnlyWatermark, mutatingWatermark int
}

func (w *requestWatermark) ensureInitialized(readonlyLimit, mutatingLimit int) {
	w.initializeOnce.Do(func() {
		if readonlyLimit == 0 {
			readonlyLimit = 1
		}
		if mutatingLimit == 0 {
			mutatingLimit = 1
		}
		w.readOnlyObserver, w.mutatingObserver = w.observersFunc(readonlyLimit, mutatingLimit)
	})
}

func (w *requestWatermark) recordMutating(mutatingVal int) {
	w.mutatingObserver.Observe(float64(mutatingVal))

	w.watermarksLock.Lock()
	defer w.watermarksLock.Unlock()

	if w.mutatingWatermark < mutatingVal {
		w.mutatingWatermark = mutatingVal
	}
}

func (w *requestWatermark) recordReadOnly(readOnlyVal int) {
	w.readOnlyObserver.Observe(float64(readOnlyVal))

	w.watermarksLock.Lock()
	defer w.watermarksLock.Unlock()

	if w.readOnlyWatermark < readOnlyVal {
		w.readOnlyWatermark = readOnlyVal
	}
}

// watermark tracks requests being executed (not waiting in a queue).
// Initialized in first call to a handler from here.
var watermark = requestWatermark{
	phase: metrics.ExecutingPhase,
	observersFunc: func(readonlyLimit, mutatingLimit int) (readOnlyObserver, mutatingObserver fcmetrics.RatioedObserver) {
		var err error
		readOnlyObserver, err = fcmetrics.ReadWriteConcurrencyObserverVec.WithLabelValuesChecked(float64(readonlyLimit), fcmetrics.LabelValueExecuting, metrics.ReadOnlyKind)
		if err != nil {
			klog.Errorf("Failed to get readonly executing member of %v: %s", fcmetrics.ReadWriteConcurrencyObserverVec.FQName(), err)
		}
		mutatingObserver, err = fcmetrics.ReadWriteConcurrencyObserverVec.WithLabelValuesChecked(float64(mutatingLimit), fcmetrics.LabelValueExecuting, metrics.MutatingKind)
		if err != nil {
			klog.Errorf("Failed to get mutating executing member of %v: %s", fcmetrics.ReadWriteConcurrencyObserverVec.FQName(), err)
		}
		return
	}}

// startWatermarkMaintenance starts the goroutines to observe and maintain the specified watermark.
func startWatermarkMaintenance(watermark *requestWatermark, stopCh <-chan struct{}) {
	// Periodically update the inflight usage metric.
	go wait.Until(func() {
		watermark.watermarksLock.Lock()
		readOnlyWatermark := watermark.readOnlyWatermark
		mutatingWatermark := watermark.mutatingWatermark
		watermark.readOnlyWatermark = 0
		watermark.mutatingWatermark = 0
		watermark.watermarksLock.Unlock()

		metrics.UpdateInflightRequestMetrics(watermark.phase, readOnlyWatermark, mutatingWatermark)
	}, inflightUsageMetricUpdatePeriod, stopCh)
}

// WithMaxInFlightLimit limits the number of in-flight requests to buffer size of the passed in channel.
func WithMaxInFlightLimit(
	handler http.Handler,
	nonMutatingLimit int,
	mutatingLimit int,
	longRunningRequestCheck apirequest.LongRunningRequestCheck,
) http.Handler {
	if nonMutatingLimit == 0 && mutatingLimit == 0 {
		return handler
	}
	var nonMutatingChan chan bool
	var mutatingChan chan bool
	if nonMutatingLimit != 0 {
		nonMutatingChan = make(chan bool, nonMutatingLimit)
	}
	if mutatingLimit != 0 {
		mutatingChan = make(chan bool, mutatingLimit)
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()
		requestInfo, ok := apirequest.RequestInfoFrom(ctx)
		if !ok {
			handleError(w, r, fmt.Errorf("no RequestInfo found in context, handler chain must be wrong"))
			return
		}

		// Skip tracking long running events.
		if longRunningRequestCheck != nil && longRunningRequestCheck(r, requestInfo) {
			handler.ServeHTTP(w, r)
			return
		}
		watermark.ensureInitialized(nonMutatingLimit, mutatingLimit)

		var c chan bool
		isMutatingRequest := !nonMutatingRequestVerbs.Has(requestInfo.Verb)
		if isMutatingRequest {
			c = mutatingChan
		} else {
			c = nonMutatingChan
		}

		if c == nil {
			handler.ServeHTTP(w, r)
		} else {

			select {
			case c <- true:
				// We note the concurrency level both while the
				// request is being served and after it is done being
				// served, because both states contribute to the
				// sampled stats on concurrency.
				if isMutatingRequest {
					watermark.recordMutating(len(c))
				} else {
					watermark.recordReadOnly(len(c))
				}
				defer func() {
					<-c
					if isMutatingRequest {
						watermark.recordMutating(len(c))
					} else {
						watermark.recordReadOnly(len(c))
					}
				}()
				handler.ServeHTTP(w, r)

			default:
				// at this point we're about to return a 429, BUT not all actors should be rate limited.  A system:master is so powerful
				// that they should always get an answer.  It's a super-admin or a loopback connection.
				if currUser, ok := apirequest.UserFrom(ctx); ok {
					for _, group := range currUser.GetGroups() {
						if group == user.SystemPrivilegedGroup {
							handler.ServeHTTP(w, r)
							return
						}
					}
				}
				// We need to split this data between buckets used for throttling.
				metrics.RecordDroppedRequest(r, requestInfo, metrics.APIServerComponent, isMutatingRequest)
				metrics.RecordRequestTermination(r, requestInfo, metrics.APIServerComponent, http.StatusTooManyRequests)
				tooManyRequests(r, w)
			}
		}
	})
}

// StartMaxInFlightWatermarkMaintenance starts the goroutines to observe and maintain watermarks for max-in-flight
// requests.
func StartMaxInFlightWatermarkMaintenance(stopCh <-chan struct{}) {
	startWatermarkMaintenance(&watermark, stopCh)
}

func tooManyRequests(req *http.Request, w http.ResponseWriter) {
	// Return a 429 status indicating "Too Many Requests"
	w.Header().Set("Retry-After", retryAfter)
	http.Error(w, "Too many requests, please try again later.", http.StatusTooManyRequests)
}
