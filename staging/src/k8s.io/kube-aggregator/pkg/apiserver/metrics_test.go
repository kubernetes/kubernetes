/*
Copyright 2026 The Kubernetes Authors.

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

package apiserver

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestRecordAggregatorRequest(t *testing.T) {
	aggregatorRequestCounter.Reset()
	aggregatorRequestDuration.Reset()
	t.Cleanup(func() {
		aggregatorRequestCounter.Reset()
		aggregatorRequestDuration.Reset()
	})

	recordAggregatorRequest("get", "metrics.k8s.io", "v1beta1", http.StatusOK, 50*time.Millisecond)
	recordAggregatorRequest("get", "metrics.k8s.io", "v1beta1", http.StatusOK, 30*time.Millisecond)
	recordAggregatorRequest("list", "metrics.k8s.io", "v1beta1", http.StatusServiceUnavailable, 5*time.Millisecond)
	recordAggregatorRequest("create", "", "v1", http.StatusCreated, 7*time.Millisecond)

	wantCounter := `
# HELP apiserver_kube_aggregator_request_total [ALPHA] Counter of requests proxied by the kube-aggregator to extension API servers, broken down by verb, group, version, and HTTP response code.
# TYPE apiserver_kube_aggregator_request_total counter
apiserver_kube_aggregator_request_total{code="200",group="metrics.k8s.io",verb="get",version="v1beta1"} 2
apiserver_kube_aggregator_request_total{code="201",group="",verb="create",version="v1"} 1
apiserver_kube_aggregator_request_total{code="503",group="metrics.k8s.io",verb="list",version="v1beta1"} 1
`
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(wantCounter), "apiserver_kube_aggregator_request_total"); err != nil {
		t.Errorf("unexpected counter state:\n%v", err)
	}

	// We don't pin every histogram bucket, but the total sample count
	// must equal the number of recordAggregatorRequest calls above.
	histVec, err := testutil.GetHistogramVecFromGatherer(legacyregistry.DefaultGatherer, "apiserver_kube_aggregator_request_duration_seconds", nil)
	if err != nil {
		t.Fatalf("get histogram vec: %v", err)
	}
	if got, want := histVec.GetAggregatedSampleCount(), uint64(4); got != want {
		t.Errorf("apiserver_kube_aggregator_request_duration_seconds total samples: got %d, want %d", got, want)
	}
}

func TestAPIServiceGroupVersion(t *testing.T) {
	tests := []struct {
		name           string
		apiServiceName string
		requestInfo    *request.RequestInfo
		wantGroup      string
		wantVersion    string
	}{
		{
			name:           "request info present, resource request, named group",
			apiServiceName: "v1beta1.metrics.k8s.io",
			requestInfo:    &request.RequestInfo{IsResourceRequest: true, APIGroup: "metrics.k8s.io", APIVersion: "v1beta1"},
			wantGroup:      "metrics.k8s.io",
			wantVersion:    "v1beta1",
		},
		{
			name:           "request info present, core group",
			apiServiceName: "v1.",
			requestInfo:    &request.RequestInfo{IsResourceRequest: true, APIGroup: "", APIVersion: "v1"},
			wantGroup:      "",
			wantVersion:    "v1",
		},
		{
			name:           "request info absent, fall back to apiservice name (named group)",
			apiServiceName: "v1beta1.metrics.k8s.io",
			wantGroup:      "metrics.k8s.io",
			wantVersion:    "v1beta1",
		},
		{
			name:           "request info absent, fall back to apiservice name (core group)",
			apiServiceName: "v1.",
			wantGroup:      "",
			wantVersion:    "v1",
		},
		{
			name:           "non-resource request falls back to apiservice name",
			apiServiceName: "v1beta1.metrics.k8s.io",
			requestInfo:    &request.RequestInfo{IsResourceRequest: false},
			wantGroup:      "metrics.k8s.io",
			wantVersion:    "v1beta1",
		},
		{
			name:           "malformed apiservice name with no separator",
			apiServiceName: "garbage",
			wantGroup:      "",
			wantVersion:    "garbage",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/", nil)
			if tc.requestInfo != nil {
				req = req.WithContext(request.WithRequestInfo(req.Context(), tc.requestInfo))
			}
			gotGroup, gotVersion := apiServiceGroupVersion(req, tc.apiServiceName)
			if gotGroup != tc.wantGroup || gotVersion != tc.wantVersion {
				t.Errorf("apiServiceGroupVersion(%q) = (%q, %q), want (%q, %q)",
					tc.apiServiceName, gotGroup, gotVersion, tc.wantGroup, tc.wantVersion)
			}
		})
	}
}
