/*
Copyright The Kubernetes Authors.

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

package rest

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"sync"
	"testing"

	"k8s.io/client-go/tools/metrics"
)

type rejectedTestKeyType struct{}

var rejectedTestKey = rejectedTestKeyType{}

// rejectedRecorder records RequestRejected increments made on behalf of one test.
// Calls from other tests running in parallel are ignored.
type rejectedRecorder struct {
	lock  sync.Mutex
	calls []string
}

func (r *rejectedRecorder) IncrementRejected(ctx context.Context, method, host string) {
	if ctx.Value(rejectedTestKey) == nil {
		return
	}
	r.lock.Lock()
	defer r.lock.Unlock()
	r.calls = append(r.calls, method+" "+host)
}

func TestRequestRejectedMetric(t *testing.T) {
	type response struct {
		status  int
		headers map[string]string
	}
	overloaded := map[string]string{"X-Kubernetes-Overload-Rejected": "true"}
	overloadedRetry := map[string]string{"X-Kubernetes-Overload-Rejected": "true", "Retry-After": "0"}

	tests := []struct {
		name string
		// responses are returned in order, one per attempt
		responses []response
		// wantRejected is the number of attempts counted as overload rejections
		wantRejected int
	}{
		{
			name:         "429 from API Priority and Fairness or max-in-flight",
			responses:    []response{{http.StatusTooManyRequests, overloaded}},
			wantRejected: 1,
		},
		{
			name:         "429 without the header, such as an Eviction blocked by a PodDisruptionBudget",
			responses:    []response{{http.StatusTooManyRequests, nil}},
			wantRejected: 0,
		},
		{
			name:         "429 with Retry-After but without the header",
			responses:    []response{{http.StatusTooManyRequests, map[string]string{"Retry-After": "0"}}, {http.StatusOK, nil}},
			wantRejected: 0,
		},
		{
			name:         "header on a status other than 429 is ignored",
			responses:    []response{{http.StatusServiceUnavailable, overloaded}},
			wantRejected: 0,
		},
		{
			name:         "success",
			responses:    []response{{http.StatusOK, nil}},
			wantRejected: 0,
		},
		{
			name:         "every rejected attempt is counted, including ones the client retries",
			responses:    []response{{http.StatusTooManyRequests, overloadedRetry}, {http.StatusTooManyRequests, overloadedRetry}, {http.StatusOK, nil}},
			wantRejected: 2,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			recorder := &rejectedRecorder{}
			old := metrics.RequestRejected
			metrics.RequestRejected = recorder
			defer func() { metrics.RequestRejected = old }()

			var lock sync.Mutex
			attempt := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				lock.Lock()
				resp := test.responses[attempt]
				attempt++
				lock.Unlock()
				for k, v := range resp.headers {
					w.Header().Set(k, v)
				}
				w.WriteHeader(resp.status)
			}))
			defer server.Close()

			base, err := url.Parse(server.URL)
			if err != nil {
				t.Fatal(err)
			}
			ctx := context.WithValue(context.Background(), rejectedTestKey, true)
			NewRequestWithClient(base, "", defaultContentConfig(), server.Client()).
				Verb("GET").
				AbsPath("/api/v1/pods").
				Do(ctx)

			if attempt != len(test.responses) {
				t.Fatalf("expected %d attempts, got %d", len(test.responses), attempt)
			}
			var want []string
			for i := 0; i < test.wantRejected; i++ {
				want = append(want, "GET "+base.Host)
			}
			recorder.lock.Lock()
			defer recorder.lock.Unlock()
			if !reflect.DeepEqual(recorder.calls, want) {
				t.Errorf("expected rejected metric calls %v, got %v", want, recorder.calls)
			}
		})
	}
}
