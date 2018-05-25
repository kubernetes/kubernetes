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

	"github.com/golang/glog"
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

var nonMutatingRequestVerbs = sets.NewString("get", "list", "watch")

func handleError(w http.ResponseWriter, r *http.Request, err error) {
	w.WriteHeader(http.StatusInternalServerError)
	fmt.Fprintf(w, "Internal Server Error: %#v", r.RequestURI)
	glog.Errorf(err.Error())
}

// requestWatermark is used to trak maximal usage of inflight requests.
type requestWatermark struct {
	lock                                 sync.Mutex
	readOnlyWatermark, mutatingWatermark int
}

func (w *requestWatermark) recordMutating(mutatingVal int) {
	w.lock.Lock()
	defer w.lock.Unlock()

	if w.mutatingWatermark < mutatingVal {
		w.mutatingWatermark = mutatingVal
	}
}

func (w *requestWatermark) recordReadOnly(readOnlyVal int) {
	w.lock.Lock()
	defer w.lock.Unlock()

	if w.readOnlyWatermark < readOnlyVal {
		w.readOnlyWatermark = readOnlyVal
	}
}

var watermark = &requestWatermark{}

func startRecordingUsage() {
	go func() {
		wait.Forever(func() {
			watermark.lock.Lock()
			readOnlyWatermark := watermark.readOnlyWatermark
			mutatingWatermark := watermark.mutatingWatermark
			watermark.readOnlyWatermark = 0
			watermark.mutatingWatermark = 0
			watermark.lock.Unlock()

			metrics.UpdateInflightRequestMetrics(readOnlyWatermark, mutatingWatermark)
		}, inflightUsageMetricUpdatePeriod)
	}()
}

var startOnce sync.Once

// WithMaxInFlightLimit limits the number of in-flight requests to buffer size of the passed in channel.
func WithMaxInFlightLimit(
	handler http.Handler,
	nonMutatingLimit int,
	mutatingLimit int,
	longRunningRequestCheck apirequest.LongRunningRequestCheck,
) http.Handler {
	startOnce.Do(startRecordingUsage)
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
				var mutatingLen, readOnlyLen int
				if isMutatingRequest {
					mutatingLen = len(mutatingChan)
				} else {
					readOnlyLen = len(nonMutatingChan)
				}

				defer func() {
					<-c
					if isMutatingRequest {
						watermark.recordMutating(mutatingLen)
					} else {
						watermark.recordReadOnly(readOnlyLen)
					}

				}()
				handler.ServeHTTP(w, r)

			default:
				// We need to split this data between buckets used for throttling.
				if isMutatingRequest {
					metrics.DroppedRequests.WithLabelValues(metrics.MutatingKind).Inc()
				} else {
					metrics.DroppedRequests.WithLabelValues(metrics.ReadOnlyKind).Inc()
				}
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
				metrics.Record(r, requestInfo, "", http.StatusTooManyRequests, 0, 0)
				tooManyRequests(r, w)
			}
		}
	})
}

func tooManyRequests(req *http.Request, w http.ResponseWriter) {
	// Return a 429 status indicating "Too Many Requests"
	w.Header().Set("Retry-After", retryAfter)
	http.Error(w, "Too many requests, please try again later.", http.StatusTooManyRequests)
}
