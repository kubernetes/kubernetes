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
const aggregatedLocalJSONAccept = jsonAccept + aggregatedAcceptSuffix + ";profile=local"
const aggregatedLocalProtoAccept = protobufAccept + aggregatedAcceptSuffix + ";profile=local"

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
	aggregated := fakeHTTPHandler{data: "aggregated-local"}
	peerAggregated := fakeHTTPHandler{data: "peer-aggregated"}
	wrapped := WrapAggregatedDiscoveryToHandler(unaggregated, aggregated, peerAggregated)

	testCases := []struct {
		accept               string
		expected             string
		enablePeerAggregated bool
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
			expected: "aggregated-local",
		}, {
			accept:   aggregatedV2Beta1ProtoAccept,
			expected: "aggregated-local",
		}, {
			accept:   aggregatedJSONAccept,
			expected: "aggregated-local",
		}, {
			accept:   aggregatedProtoAccept,
			expected: "aggregated-local",
		}, {
			accept:   jsonAccept,
			expected: "unaggregated",
		}, {
			accept:   protobufAccept,
			expected: "unaggregated",
		}, {
			// Server should return the first accepted type
			accept:   aggregatedJSONAccept + "," + jsonAccept,
			expected: "aggregated-local",
		}, {
			// Server should return the first accepted type
			accept:   aggregatedProtoAccept + "," + protobufAccept,
			expected: "aggregated-local",
		},
		// Peer Agg discovery cases.
		// profile is not set (should default to peer-aggregated)
		{
			accept:               aggregatedJSONAccept,
			expected:             "peer-aggregated",
			enablePeerAggregated: true,
		}, {
			accept:               aggregatedProtoAccept,
			expected:             "peer-aggregated",
			enablePeerAggregated: true,
		},
		// profile=local (should return local)
		{
			accept:               aggregatedLocalJSONAccept,
			expected:             "aggregated-local",
			enablePeerAggregated: true,
		}, {
			accept:               aggregatedLocalProtoAccept,
			expected:             "aggregated-local",
			enablePeerAggregated: true,
		},
		// profile is set to something other than local (should default to peer-aggregated)
		{
			accept:               aggregatedJSONAccept + ";profile=foo",
			expected:             "peer-aggregated",
			enablePeerAggregated: true,
		},
	}

	for _, tc := range testCases {
		if tc.accept == aggregatedV2Beta1JSONAccept || tc.accept == aggregatedV2Beta1ProtoAccept {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.AggregatedDiscoveryRemoveBetaType, false)
		}
		if tc.enablePeerAggregated {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.UnknownVersionInteroperabilityProxy, true)
		}
		body := fetchPath(wrapped, discoveryPath, tc.accept)
		assert.Equal(t, tc.expected, body)
	}
}
