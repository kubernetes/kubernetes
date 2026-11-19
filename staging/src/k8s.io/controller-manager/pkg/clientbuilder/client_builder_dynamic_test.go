/*
Copyright The Kubernetes Authors.

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

package clientbuilder

import (
	"net/http"
	"testing"

	"k8s.io/client-go/kubernetes/fake"
	restclient "k8s.io/client-go/rest"
)

// TestDynamicClientBuilderAppliesBaseWrapperOnce builds several configs for
// the same service account and checks that the base config's WrapTransport
// is applied exactly once per client, including on token wrapper cache hits.
func TestDynamicClientBuilderAppliesBaseWrapperOnce(t *testing.T) {
	baseWraps := 0
	base := &restclient.Config{Host: "https://localhost:6443"}
	base.Wrap(func(rt http.RoundTripper) http.RoundTripper {
		baseWraps++
		return rt
	})
	builder := NewTestDynamicClientBuilder(base, fake.NewSimpleClientset().CoreV1(), "kube-system", 3600, 50)

	for i := 1; i <= 3; i++ {
		config, err := builder.Config("controller")
		if err != nil {
			t.Fatalf("Config: %v", err)
		}
		if _, err := restclient.TransportFor(config); err != nil {
			t.Fatalf("TransportFor: %v", err)
		}
		if baseWraps != i {
			t.Fatalf("after %d clients the base wrapper was applied %d times, want %d", i, baseWraps, i)
		}
	}
}
