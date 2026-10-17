/*
Copyright 2025 The Kubernetes Authors.

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

package client

import (
	"testing"

	resourceapi "k8s.io/api/resource/v1"
	resourcev1beta1 "k8s.io/api/resource/v1beta1"
	resourcev1beta2 "k8s.io/api/resource/v1beta2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/util/watchlist"
)

func TestDoesClientSupportWatchListSemantics(t *testing.T) {
	scenarios := []struct {
		name                                           string
		clientSet                                      kubernetes.Interface
		expectedDoesClientNotSupportWatchListSemantics bool
	}{
		{
			name:      "DR with real kube client supports WatchList semantics",
			clientSet: kubernetes.NewForConfigOrDie(&restclient.Config{}),
			expectedDoesClientNotSupportWatchListSemantics: false,
		},
		{
			name:      "DR with fake kube client does NOT support WatchList semantics",
			clientSet: fake.NewClientset(),
			expectedDoesClientNotSupportWatchListSemantics: true,
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			target := New(scenario.clientSet)
			actual := watchlist.DoesClientNotSupportWatchListSemantics(target)
			if actual != scenario.expectedDoesClientNotSupportWatchListSemantics {
				t.Fatalf("watchlist.DoesClientNotSupportWatchListSemantics, got: %v, want: %v", actual, scenario.expectedDoesClientNotSupportWatchListSemantics)
			}
		})
	}
}

// TestConvertingClientUpdateStatusUsesSelectedAPI verifies that UpdateStatus is
// served by the client for the API version the converting client has selected,
// like every other verb it forwards.
func TestConvertingClientUpdateStatusUsesSelectedAPI(t *testing.T) {
	objectMeta := metav1.ObjectMeta{Name: "test-claim", Namespace: "default"}

	scenarios := []struct {
		name     string
		useAPI   int32
		expected schema.GroupVersion
	}{
		{name: "latest", useAPI: useLatestAPI, expected: resourceapi.SchemeGroupVersion},
		{name: "v1beta2", useAPI: useV1beta2API, expected: resourcev1beta2.SchemeGroupVersion},
		{name: "v1beta1", useAPI: useV1beta1API, expected: resourcev1beta1.SchemeGroupVersion},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			// Every version needs its own claim. A version that does not have it
			// answers "not found", which sends the call on to the next version.
			fakeClientset := fake.NewClientset(
				&resourceapi.ResourceClaim{ObjectMeta: objectMeta},
				&resourcev1beta1.ResourceClaim{ObjectMeta: objectMeta},
				&resourcev1beta2.ResourceClaim{ObjectMeta: objectMeta},
			)
			target := New(fakeClientset)
			target.useAPI.Store(scenario.useAPI)

			claim := &resourceapi.ResourceClaim{ObjectMeta: objectMeta}
			if _, err := target.ResourceClaims(objectMeta.Namespace).UpdateStatus(t.Context(), claim, metav1.UpdateOptions{}); err != nil {
				t.Fatalf("UpdateStatus: %v", err)
			}

			var actual []schema.GroupVersion
			for _, action := range fakeClientset.Actions() {
				if action.GetVerb() == "update" && action.GetSubresource() == "status" {
					actual = append(actual, action.GetResource().GroupVersion())
				}
			}
			if len(actual) != 1 || actual[0] != scenario.expected {
				t.Errorf("status update issued through %v, want exactly one through %v", actual, scenario.expected)
			}
		})
	}
}
