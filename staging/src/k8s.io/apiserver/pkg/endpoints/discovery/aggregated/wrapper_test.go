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
	"k8s.io/kubernetes/pkg/features"
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

const unmergedAccept = ";profile=unmerged"

func fetchPath(handler http.Handler, path, accept string) string {
	w := httptest.NewRecorder()
	req := httptest.NewRequest(request.MethodGet, discoveryPath, nil)

	// Ask for JSON response
	req.Header.Set("Accept", accept)

	handler.ServeHTTP(w, req)
	return string(w.Body.Bytes())
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
	wrapped := WrapAggregatedDiscoveryToHandler(unaggregated, aggregated)

	testCases := []struct {
		accept   string
		expected string
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
	}

	for _, tc := range testCases {
		if tc.accept == aggregatedV2Beta1JSONAccept || tc.accept == aggregatedV2Beta1ProtoAccept {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.AggregatedDiscoveryRemoveBetaType, false)
		}
		body := fetchPath(wrapped, discoveryPath, tc.accept)
		assert.Equal(t, tc.expected, body)
	}
}

func TestMergedDiscoveryEnabled(t *testing.T) {
	unaggregated := fakeHTTPHandler{data: "unaggregated"}
	aggregated := fakeHTTPHandler{data: "aggregated"}
	merged := fakeHTTPHandler{data: "merged"}
	wrapped := WrapMergedDiscoveryToHandler(unaggregated, aggregated, merged)

	testCases := []struct {
		name                 string
		accept               string
		mergedFeatureEnabled bool
		expected             string
	}{
		{
			name:                 "merged feature disabled - default JSON",
			accept:               jsonAccept,
			mergedFeatureEnabled: false,
			expected:             "unaggregated",
		}, {
			name:                 "merged feature disabled - aggregated JSON",
			accept:               aggregatedJSONAccept,
			mergedFeatureEnabled: false,
			expected:             "aggregated",
		}, {
			name:                 "merged feature enabled - default JSON without profile",
			accept:               jsonAccept,
			mergedFeatureEnabled: true,
			expected:             "unaggregated",
		}, {
			name:                 "merged feature enabled - default JSON profile=merged",
			accept:               jsonAccept + ";profile=merged",
			mergedFeatureEnabled: true,
			expected:             "unaggregated",
		}, {
			name:                 "merged feature enabled - default JSON profile=unmerged",
			accept:               jsonAccept + unmergedAccept,
			mergedFeatureEnabled: true,
			expected:             "unaggregated",
		}, {
			name:                 "merged feature enabled - aggregated JSON without profile",
			accept:               aggregatedJSONAccept,
			mergedFeatureEnabled: true,
			expected:             "merged",
		}, {
			name:                 "merged feature enabled - aggregated JSON profile=unmerged",
			accept:               aggregatedJSONAccept + unmergedAccept,
			mergedFeatureEnabled: true,
			expected:             "aggregated",
		}, {
			name:                 "merged feature enabled - aggregated JSON profile=merged",
			accept:               aggregatedJSONAccept + ";profile=merged",
			mergedFeatureEnabled: true,
			expected:             "merged",
		}, {
			name:                 "merged feature enabled - default protobuf",
			accept:               protobufAccept,
			mergedFeatureEnabled: true,
			expected:             "unaggregated",
		}, {
			name:                 "merged feature enabled - default protobuf profile=merged",
			accept:               protobufAccept + ";profile=merged",
			mergedFeatureEnabled: true,
			expected:             "unaggregated",
		}, {
			name:                 "merged feature enabled - default protobuf profile=unmerged",
			accept:               protobufAccept + ";profile=unmerged",
			mergedFeatureEnabled: true,
			expected:             "unaggregated",
		},
		{
			name:                 "merged feature enabled - aggregated protobuf without profile",
			accept:               aggregatedProtoAccept,
			mergedFeatureEnabled: true,
			expected:             "merged",
		}, {
			name:                 "merged feature enabled - aggregated protobuf profile=unmerged",
			accept:               aggregatedProtoAccept + ";profile=unmerged",
			mergedFeatureEnabled: true,
			expected:             "aggregated",
		}, {
			name:                 "merged feature enabled - aggregated protobuf profile=merged",
			accept:               aggregatedProtoAccept + ";profile=merged",
			mergedFeatureEnabled: true,
			expected:             "merged",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Set the merged discovery feature gate
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.UnknownVersionInteroperabilityProxy, tc.mergedFeatureEnabled)

			// Handle beta type feature gate if needed
			if tc.accept == aggregatedV2Beta1JSONAccept || tc.accept == aggregatedV2Beta1ProtoAccept {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.AggregatedDiscoveryRemoveBetaType, false)
			}

			body := fetchPath(wrapped, discoveryPath, tc.accept)
			assert.Equal(t, tc.expected, body)
		})
	}
}
