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

package aggregated

import (
	"net/http"
	"net/http/httptest"

	"io"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apiserver/pkg/endpoints/request"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

const discoveryPath = "/apis"
const jsonAccept = "application/json"
const protobufAccept = "application/vnd.kubernetes.protobuf"
const aggregatedV2Beta1AcceptSuffix = ";g=apidiscovery.k8s.io;v=v2beta1;as=APIGroupDiscoveryList"
const aggregatedAcceptSuffix = ";g=apidiscovery.k8s.io;v=v2;as=APIGroupDiscoveryList"

const aggregatedV2Beta1JSONAccept = jsonAccept + aggregatedV2Beta1AcceptSuffix
const aggregatedV2Beta1ProtoAccept = protobufAccept + aggregatedV2Beta1AcceptSuffix
const aggregatedJSONAccept = jsonAccept + aggregatedAcceptSuffix
const aggregatedProtoAccept = protobufAccept + aggregatedAcceptSuffix
const aggregatedMergedJSONAccept = jsonAccept + aggregatedAcceptSuffix + ";profile=merged"
const aggregatedMergedProtoAccept = protobufAccept + aggregatedAcceptSuffix + ";profile=merged"

func fetchPath(handler http.Handler, path, accept string) string {
	w := httptest.NewRecorder()
	req := httptest.NewRequest(request.MethodGet, discoveryPath, nil)

	// Ask for JSON response
	req.Header.Set("Accept", accept)

	handler.ServeHTTP(w, req)
	return w.Body.String()
}

type fakeHTTPHandler struct {
	data string
}

func (f fakeHTTPHandler) ServeHTTP(resp http.ResponseWriter, req *http.Request) {
	io.WriteString(resp, f.data)
}

func TestAggregationEnabled(t *testing.T) {
	unaggregated := fakeHTTPHandler{data: "unaggregated"}
	aggregated := fakeHTTPHandler{data: "aggregated"}
	merged := fakeHTTPHandler{data: "merged"}
	wrapped := WrapAggregatedDiscoveryToHandler(unaggregated, aggregated, merged)

	testCases := []struct {
		accept       string
		expected     string
		enableMerged bool
	}{
		{
			// Misconstructed/incorrect accept headers should be passed to the unaggregated handler to return an error
			accept:   "application/json;foo=bar",
			expected: "unaggregated",
		}, {
			// Empty accept headers are valid and should be handled by the unaggregated handler
			accept:   "",
			expected: "unaggregated",
		}, {
			accept:   aggregatedV2Beta1JSONAccept,
			expected: "aggregated",
		}, {
			accept:   aggregatedV2Beta1ProtoAccept,
			expected: "aggregated",
		}, {
			accept:   aggregatedJSONAccept,
			expected: "aggregated",
		}, {
			accept:   aggregatedProtoAccept,
			expected: "aggregated",
		}, {
			accept:   jsonAccept,
			expected: "unaggregated",
		}, {
			accept:   protobufAccept,
			expected: "unaggregated",
		}, {
			// Server should return the first accepted type
			accept:   aggregatedJSONAccept + "," + jsonAccept,
			expected: "aggregated",
		}, {
			// Server should return the first accepted type
			accept:   aggregatedProtoAccept + "," + protobufAccept,
			expected: "aggregated",
		},
		// Peer Agg discovery cases.
		{
			accept:       aggregatedMergedJSONAccept,
			expected:     "merged",
			enableMerged: true,
		}, {
			accept:       aggregatedMergedProtoAccept,
			expected:     "merged",
			enableMerged: true,
		},
		// profile is not set (should default to merged)
		{
			accept:       aggregatedJSONAccept,
			expected:     "merged",
			enableMerged: true,
		},
		// profile is set to something other than unmerged (should default to merged)
		{
			accept:       aggregatedJSONAccept + ";profile=foo",
			expected:     "merged",
			enableMerged: true,
		},
	}

	for _, tc := range testCases {
		if tc.accept == aggregatedV2Beta1JSONAccept || tc.accept == aggregatedV2Beta1ProtoAccept {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.AggregatedDiscoveryRemoveBetaType, false)
		}
		if tc.enableMerged {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.UnknownVersionInteroperabilityProxy, true)
		}
		body := fetchPath(wrapped, discoveryPath, tc.accept)
		assert.Equal(t, tc.expected, body)
	}
}
