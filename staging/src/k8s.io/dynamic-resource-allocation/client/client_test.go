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
	"context"
	"testing"

	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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

// TestConvertingClientUpdateStatusUsesV1beta2 forces the converting client onto
// the v1beta2 path and checks that UpdateStatus issues the request through the
// v1beta2 client. The buggy code asserted the v1beta1 client to an interface
// parameterized with v1beta2 object types, which panics before any request is
// sent. The claim does not need to exist: the fix is about which client is
// used, and the recorded action proves the v1beta2 client was selected.
func TestConvertingClientUpdateStatusUsesV1beta2(t *testing.T) {
	ctx := context.Background()
	const ns = "default"
	claim := &resourceapi.ResourceClaim{ObjectMeta: metav1.ObjectMeta{Name: "test-claim", Namespace: ns}}
	fakeClientset := fake.NewClientset()

	c := New(fakeClientset)
	c.useAPI.Store(useV1beta2API) // force the v1beta2 code path

	var panicValue any
	func() {
		defer func() { panicValue = recover() }()
		_, _ = c.ResourceClaims(ns).UpdateStatus(ctx, claim, metav1.UpdateOptions{})
	}()
	if panicValue != nil {
		t.Fatalf("UpdateStatus panicked on the v1beta2 path (v1beta1 client asserted to a v1beta2-typed interface): %v", panicValue)
	}

	var sawV1beta2StatusUpdate bool
	for _, a := range fakeClientset.Actions() {
		if a.GetVerb() == "update" && a.GetSubresource() == "status" && a.GetResource().Version == "v1beta2" {
			sawV1beta2StatusUpdate = true
		}
	}
	if !sawV1beta2StatusUpdate {
		t.Errorf("expected an UpdateStatus request through resource.k8s.io/v1beta2; got actions %v", fakeClientset.Actions())
	}
}
