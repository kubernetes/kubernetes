/*
Copyright 2022 The Kubernetes Authors.

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

package restclient

import (
	"context"
	"strings"
	"testing"

	"k8s.io/client-go/tools/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestClientGOMetrics(t *testing.T) {
	tests := []struct {
		description string
		name        string
		metric      interface{}
		update      func()
		want        string
	}{
		{
			description: "Number of HTTP requests, partitioned by status code, verb, and host.",
			name:        "rest_client_requests_total",
			metric:      requestResult,
			update: func() {
				metrics.RequestResult.Increment(context.TODO(), "200", "POST", "www.foo.com")
			},
			want: `
			            # HELP rest_client_requests_total [ALPHA] Number of HTTP requests, partitioned by status code, method, and host.
			            # TYPE rest_client_requests_total counter
			            rest_client_requests_total{code="200",host="www.foo.com",method="POST"} 1
				`,
		},
		{
			description: "Number of request retries, partitioned by status code, verb, and host.",
			name:        "rest_client_request_retries_total",
			metric:      requestRetry,
			update: func() {
				metrics.RequestRetry.IncrementRetry(context.TODO(), "500", "GET", "www.bar.com")
			},
			want: `
			            # HELP rest_client_request_retries_total [ALPHA] Number of request retries, partitioned by status code, verb, and host.
			            # TYPE rest_client_request_retries_total counter
			            rest_client_request_retries_total{code="500",host="www.bar.com",verb="GET"} 1
				`,
		},
	}

	// no need to register the metrics here, since the init function of
	// the package registers all the client-go metrics.
	for _, test := range tests {
		t.Run(test.description, func(t *testing.T) {
			resetter, resettable := test.metric.(interface {
				Reset()
			})
			if !resettable {
				t.Fatalf("the metric must be resettaable: %s", test.name)
			}

			// Since prometheus' gatherer is global, other tests may have updated
			// metrics already, so we need to reset them prior to running this test.
			// This also implies that we can't run this test in parallel with other tests.
			resetter.Reset()
			test.update()

			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(test.want), test.name); err != nil {
				t.Fatal(err)
			}
		})
	}
}
