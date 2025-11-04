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
