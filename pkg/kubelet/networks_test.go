/*
Copyright 2017 The Kubernetes Authors.

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

package kubelet

import (
	"testing"
)

func TestNetworkHostGetsPodNotFound(t *testing.T) {
	testKubelet := newTestKubelet(t, true)
	defer testKubelet.Cleanup()
	nh := networkHost{testKubelet.kubelet}

	actualPod, _ := nh.GetPodByName("", "")
	if actualPod != nil {
		t.Fatalf("Was expected nil, received %v instead", actualPod)
	}
}

func TestNetworkHostGetsKubeClient(t *testing.T) {
	testKubelet := newTestKubelet(t, true)
	defer testKubelet.Cleanup()
	nh := networkHost{testKubelet.kubelet}

	if nh.GetKubeClient() != testKubelet.fakeKubeClient {
		t.Fatalf("NetworkHost client does not match testKubelet's client")
	}
}

func TestNetworkHostGetsRuntime(t *testing.T) {
	testKubelet := newTestKubelet(t, true)
	defer testKubelet.Cleanup()
	nh := networkHost{testKubelet.kubelet}

	if nh.GetRuntime() != testKubelet.fakeRuntime {
		t.Fatalf("NetworkHost runtime does not match testKubelet's runtime")
	}
}

func TestNetworkHostSupportsLegacyFeatures(t *testing.T) {
	testKubelet := newTestKubelet(t, true)
	defer testKubelet.Cleanup()
	nh := networkHost{testKubelet.kubelet}

	if nh.SupportsLegacyFeatures() == false {
		t.Fatalf("SupportsLegacyFeatures should not be false")
	}
}

func TestNoOpHostGetsName(t *testing.T) {
	nh := NoOpLegacyHost{}
	pod, err := nh.GetPodByName("", "")
	if pod != nil && err != true {
		t.Fatalf("noOpLegacyHost getpodbyname expected to be nil and true")
	}
}

func TestNoOpHostGetsKubeClient(t *testing.T) {
	nh := NoOpLegacyHost{}
	if nh.GetKubeClient() != nil {
		t.Fatalf("noOpLegacyHost client expected to be nil")
	}
}

func TestNoOpHostGetsRuntime(t *testing.T) {
	nh := NoOpLegacyHost{}
	if nh.GetRuntime() != nil {
		t.Fatalf("noOpLegacyHost runtime expected to be nil")
	}
}

func TestNoOpHostSupportsLegacyFeatures(t *testing.T) {
	nh := NoOpLegacyHost{}
	if nh.SupportsLegacyFeatures() != false {
		t.Fatalf("noOpLegacyHost legacy features expected to be false")
	}
}
