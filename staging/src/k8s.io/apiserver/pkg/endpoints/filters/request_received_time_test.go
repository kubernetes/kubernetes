/*
Copyright 2020 The Kubernetes Authors.

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
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	utilclock "k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apiserver/pkg/endpoints/request"
)

func TestWithRequestReceivedTimestamp(t *testing.T) {
	receivedTimestampExpected := time.Now()

	var (
		callCount            int
		receivedTimestampGot time.Time
		ok                   bool
	)
	handler := http.HandlerFunc(func(_ http.ResponseWriter, req *http.Request) {
		callCount++

		// does the handler chain that follows see the request received timestamp?
		receivedTimestampGot, ok = request.ReceivedTimestampFrom(req.Context())
	})

	wrapped := withRequestReceivedTimestampWithClock(handler, utilclock.NewFakeClock(receivedTimestampExpected))

	testRequest, err := http.NewRequest(http.MethodGet, "/api/v1/namespaces", nil)
	if err != nil {
		t.Fatalf("failed to create new http request - %v", err)
	}

	w := httptest.NewRecorder()
	wrapped.ServeHTTP(w, testRequest)

	if callCount != 1 {
		t.Errorf("WithRequestReceivedTimestamp: expected the original handler to be invoked once, but was actually invoked %d times", callCount)
	}
	if !ok {
		t.Error("WithRequestReceivedTimestamp: expected request.ReceivedTimestampFrom to return true, but got false")
	}
	if receivedTimestampExpected != receivedTimestampGot {
		t.Errorf("WithRequestReceivedTimestamp: request received timestamp expected=%s but got=%s", receivedTimestampExpected, receivedTimestampGot)
	}
}
