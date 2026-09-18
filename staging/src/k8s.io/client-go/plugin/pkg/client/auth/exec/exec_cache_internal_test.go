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

package exec

import (
	"reflect"
	"testing"

	"k8s.io/client-go/pkg/apis/clientauthentication"
	"k8s.io/client-go/tools/clientcmd/api"
)

// TestCacheKeyFields verifies that every ExecConfig and Cluster field has been
// considered for cache identity. When either source type changes, update
// execConfigCacheKeyData or clusterCacheKeyData to include identity-relevant
// fields, or add intentionally excluded fields to the corresponding skipped
// list.
func TestCacheKeyFields(t *testing.T) {
	tests := []struct {
		name     string
		source   reflect.Type
		included reflect.Type
		skipped  []string
	}{
		{
			name:     "ExecConfig",
			source:   reflect.TypeFor[api.ExecConfig](),
			included: reflect.TypeFor[execConfigCacheKeyData](),
			skipped:  []string{
				// "ExampleExcludedField", // Example of a field intentionally excluded from cache identity.
			},
		},
		{
			name:     "Cluster",
			source:   reflect.TypeFor[clientauthentication.Cluster](),
			included: reflect.TypeFor[clusterCacheKeyData](),
			skipped:  []string{
				// "ExampleExcludedField", // Example of a field intentionally excluded from cache identity.
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			classified := make(map[string]struct{}, tc.included.NumField()+len(tc.skipped))

			for field := range tc.included.Fields() {
				classified[field.Name] = struct{}{}
			}
			for _, field := range tc.skipped {
				if _, ok := classified[field]; ok {
					t.Fatalf("field %q is both included and skipped", field)
				}
				classified[field] = struct{}{}
			}

			for field := range tc.source.Fields() {
				if _, ok := classified[field.Name]; !ok {
					t.Errorf("field %q has not been considered for cache identity", field.Name)
					continue
				}
				delete(classified, field.Name)
			}

			for field := range classified {
				t.Errorf("classified field %q does not exist", field)
			}
		})
	}
}

// TestCacheKeyPluginPolicy verifies that fields intentionally excluded from
// PluginPolicy's JSON representation still contribute to the cache key.
func TestCacheKeyPluginPolicy(t *testing.T) {
	conf1 := &api.ExecConfig{
		PluginPolicy: api.PluginPolicy{
			PolicyType: api.PluginPolicyAllowlist,
			Allowlist: []api.AllowlistEntry{
				{Command: "foo"},
			},
		},
	}
	conf2 := conf1.DeepCopy()
	conf2.PluginPolicy.Allowlist[0].Command = "bar"

	if makeCacheKey(t, conf1, nil) == makeCacheKey(t, conf2, nil) {
		t.Fatal("expected different cache keys for different plugin allowlists")
	}
}
