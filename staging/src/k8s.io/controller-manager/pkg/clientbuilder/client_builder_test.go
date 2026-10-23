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
	"strings"
	"sync"
	"testing"

	"k8s.io/client-go/kubernetes/fake"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/util/flowcontrol"
)

const (
	baseQPS   float32 = 3
	baseBurst int     = 7
)

// builders returns every TypedClientBuilder implementation in this package,
// each built on a fresh copy of the same base config. The base config carries
// a RateLimiter so that tests can check WithRateLimits drops it.
func builders(t *testing.T) map[string]func() TypedClientBuilder {
	t.Helper()
	base := func() *restclient.Config {
		return &restclient.Config{
			Host:        "https://localhost:6443",
			QPS:         baseQPS,
			Burst:       baseBurst,
			RateLimiter: flowcontrol.NewFakeAlwaysRateLimiter(),
		}
	}
	return map[string]func() TypedClientBuilder{
		"simple": func() TypedClientBuilder {
			return SimpleControllerClientBuilder{ClientConfig: base()}
		},
		"dynamic": func() TypedClientBuilder {
			return NewTestDynamicClientBuilder(base(), fake.NewSimpleClientset().CoreV1(), "kube-system", 3600, 50).(TypedClientBuilder)
		},
	}
}

func config(t *testing.T, b TypedClientBuilder, name string) *restclient.Config {
	t.Helper()
	cfg, err := b.Config(name)
	if err != nil {
		t.Fatalf("Config(%q): %v", name, err)
	}
	return cfg
}

func TestDerivedBuilders(t *testing.T) {
	type want struct {
		qps               float32
		burst             int
		rateLimiter       bool
		dropManagedFields bool
		// discoveryQPS is the QPS of the discovery client. Zero skips the
		// check: with the base RateLimiter in place QPS has no effect.
		discoveryQPS float32
	}
	tests := []struct {
		name   string
		derive func(TypedClientBuilder) TypedClientBuilder
		want   want
	}{
		{
			name:   "unchanged",
			derive: func(b TypedClientBuilder) TypedClientBuilder { return b },
			want:   want{qps: baseQPS, burst: baseBurst, rateLimiter: true},
		},
		{
			name:   "rate limits",
			derive: func(b TypedClientBuilder) TypedClientBuilder { return b.WithRateLimits(60, 300) },
			want:   want{qps: 60, burst: 300, discoveryQPS: 60},
		},
		{
			name:   "rate limits below the discovery floor",
			derive: func(b TypedClientBuilder) TypedClientBuilder { return b.WithRateLimits(5, 10) },
			want:   want{qps: 5, burst: 10, discoveryQPS: 20},
		},
		{
			name:   "drop managed fields",
			derive: func(b TypedClientBuilder) TypedClientBuilder { return b.WithDropManagedFields() },
			want:   want{qps: baseQPS, burst: baseBurst, rateLimiter: true, dropManagedFields: true},
		},
		{
			name: "rate limits then drop managed fields",
			derive: func(b TypedClientBuilder) TypedClientBuilder {
				return b.WithRateLimits(60, 300).WithDropManagedFields()
			},
			want: want{qps: 60, burst: 300, dropManagedFields: true, discoveryQPS: 60},
		},
		{
			name: "drop managed fields then rate limits",
			derive: func(b TypedClientBuilder) TypedClientBuilder {
				return b.WithDropManagedFields().WithRateLimits(60, 300)
			},
			want: want{qps: 60, burst: 300, dropManagedFields: true, discoveryQPS: 60},
		},
		{
			name: "rate limits twice",
			derive: func(b TypedClientBuilder) TypedClientBuilder {
				return b.WithRateLimits(1, 1).WithRateLimits(60, 300)
			},
			want: want{qps: 60, burst: 300, discoveryQPS: 60},
		},
	}
	for builderName, newBuilder := range builders(t) {
		for _, tt := range tests {
			t.Run(builderName+"/"+tt.name, func(t *testing.T) {
				parent := newBuilder()
				derived := tt.derive(parent)

				cfg := config(t, derived, "controller")
				if cfg.QPS != tt.want.qps || cfg.Burst != tt.want.burst {
					t.Errorf("derived config has QPS=%v Burst=%v, want QPS=%v Burst=%v", cfg.QPS, cfg.Burst, tt.want.qps, tt.want.burst)
				}
				if got := cfg.RateLimiter != nil; got != tt.want.rateLimiter {
					t.Errorf("derived config RateLimiter set = %v, want %v", got, tt.want.rateLimiter)
				}
				if cfg.DropManagedFields != tt.want.dropManagedFields {
					t.Errorf("derived config DropManagedFields = %v, want %v", cfg.DropManagedFields, tt.want.dropManagedFields)
				}
				if !strings.Contains(cfg.UserAgent, "controller") {
					t.Errorf("derived config UserAgent %q does not name the controller", cfg.UserAgent)
				}

				// Deriving must not change the parent.
				parentCfg := config(t, parent, "controller")
				if parentCfg.QPS != baseQPS || parentCfg.Burst != baseBurst || parentCfg.RateLimiter == nil || parentCfg.DropManagedFields {
					t.Errorf("parent config changed after deriving: QPS=%v Burst=%v RateLimiter=%v DropManagedFields=%v",
						parentCfg.QPS, parentCfg.Burst, parentCfg.RateLimiter != nil, parentCfg.DropManagedFields)
				}

				// The derived config must build every client type.
				if _, err := derived.Client("controller"); err != nil {
					t.Errorf("Client: %v", err)
				}
				discoveryClient, err := derived.DiscoveryClient("controller")
				if err != nil {
					t.Fatalf("DiscoveryClient: %v", err)
				}
				if tt.want.discoveryQPS != 0 {
					if got := discoveryClient.RESTClient().GetRateLimiter().QPS(); got != tt.want.discoveryQPS {
						t.Errorf("discovery client QPS = %v, want %v", got, tt.want.discoveryQPS)
					}
				}
				if _, err := derived.MetadataClient("controller"); err != nil {
					t.Errorf("MetadataClient: %v", err)
				}
				if _, err := derived.DynamicClient("controller"); err != nil {
					t.Errorf("DynamicClient: %v", err)
				}
			})
		}
	}
}

func TestDynamicDerivedBuildersShareTokenSources(t *testing.T) {
	coreClient := fake.NewSimpleClientset()
	parent := NewTestDynamicClientBuilder(&restclient.Config{Host: "https://localhost:6443"}, coreClient.CoreV1(), "kube-system", 3600, 50).(TypedClientBuilder)
	derived := parent.WithRateLimits(60, 300).WithDropManagedFields()

	for _, b := range []TypedClientBuilder{parent, derived, parent, derived} {
		config(t, b, "controller")
	}

	impl := parent.(*DynamicControllerClientBuilder)
	if got := len(impl.roundTripperFuncMap); got != 1 {
		t.Errorf("parent caches %d token wrappers, want 1", got)
	}
	if got := len(derived.(*DynamicControllerClientBuilder).roundTripperFuncMap); got != 1 {
		t.Errorf("derived builder caches %d token wrappers, want 1", got)
	}

	creates := 0
	for _, action := range coreClient.Actions() {
		if action.GetVerb() == "create" && action.GetResource().Resource == "serviceaccounts" {
			creates++
		}
	}
	if creates != 1 {
		t.Errorf("service account created %d times, want 1", creates)
	}
}

func TestDynamicDerivedBuildersConcurrentConfig(t *testing.T) {
	parent := NewTestDynamicClientBuilder(&restclient.Config{Host: "https://localhost:6443"}, fake.NewSimpleClientset().CoreV1(), "kube-system", 3600, 50).(TypedClientBuilder)
	derived := parent.WithRateLimits(60, 300)

	var wg sync.WaitGroup
	for range 10 {
		for _, b := range []TypedClientBuilder{parent, derived} {
			wg.Go(func() {
				if _, err := b.Config("controller"); err != nil {
					t.Error(err)
				}
			})
		}
	}
	wg.Wait()

	if got := len(parent.(*DynamicControllerClientBuilder).roundTripperFuncMap); got != 1 {
		t.Errorf("builders cache %d token wrappers, want 1", got)
	}
}
