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

package compression

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/apis/apiserver"
)

func gr(group, resource string) schema.GroupResource {
	return schema.GroupResource{Group: group, Resource: resource}
}

type entryConfig = apiserver.CompressionResourceConfiguration

func entry(algorithm apiserver.CompressionAlgorithm, resources ...string) entryConfig {
	return entryConfig{Resources: resources, Algorithm: algorithm}
}

func withMinSize(e entryConfig, minSize int64) entryConfig {
	e.MinSize = resource.NewQuantity(minSize, resource.BinarySI)
	return e
}

func newPolicy(entries ...entryConfig) *Policy {
	return NewPolicy(&apiserver.StorageCompressionConfiguration{Resources: entries})
}

const (
	deflate = apiserver.CompressionAlgorithmDeflate
	none    = apiserver.CompressionAlgorithmNone
)

func TestPolicyEnabledFor(t *testing.T) {
	cases := []struct {
		name     string
		entries  []entryConfig
		enabled  []schema.GroupResource
		disabled []schema.GroupResource
	}{
		{
			name:     "no entries selects nothing",
			disabled: []schema.GroupResource{gr("", "configmaps"), gr("", "secrets"), gr("example.com", "widgets")},
		},
		{
			name:     "bare resource is core group",
			entries:  []entryConfig{entry(deflate, "configmaps")},
			enabled:  []schema.GroupResource{gr("", "configmaps")},
			disabled: []schema.GroupResource{gr("apps", "configmaps"), gr("", "pods")},
		},
		{
			name:     "resource dot group",
			entries:  []entryConfig{entry(deflate, "widgets.example.com")},
			enabled:  []schema.GroupResource{gr("example.com", "widgets")},
			disabled: []schema.GroupResource{gr("", "widgets"), gr("example.com", "gadgets")},
		},
		{
			name:     "group wildcard",
			entries:  []entryConfig{entry(deflate, "*.example.com")},
			enabled:  []schema.GroupResource{gr("example.com", "widgets"), gr("example.com", "gadgets")},
			disabled: []schema.GroupResource{gr("", "configmaps"), gr("other.com", "widgets")},
		},
		{
			name:     "core group wildcard",
			entries:  []entryConfig{entry(deflate, "*.")},
			enabled:  []schema.GroupResource{gr("", "configmaps"), gr("", "pods")},
			disabled: []schema.GroupResource{gr("apps", "deployments")},
		},
		{
			name:     "everything wildcard",
			entries:  []entryConfig{entry(deflate, "*.*")},
			enabled:  []schema.GroupResource{gr("", "configmaps"), gr("apps", "deployments"), gr("example.com", "widgets")},
			disabled: []schema.GroupResource{gr("", "secrets")},
		},
		{
			name:     "algorithm None selects nothing",
			entries:  []entryConfig{entry(none, "*.*")},
			disabled: []schema.GroupResource{gr("", "configmaps"), gr("apps", "deployments")},
		},
		{
			name:     "earlier None excludes a resource from a later wildcard",
			entries:  []entryConfig{entry(none, "events"), entry(deflate, "*.*")},
			enabled:  []schema.GroupResource{gr("", "configmaps"), gr("apps", "deployments")},
			disabled: []schema.GroupResource{gr("", "events")},
		},
		{
			name:     "earlier None excludes a whole group from a later wildcard",
			entries:  []entryConfig{entry(none, "*.coordination.k8s.io"), entry(deflate, "*.*")},
			enabled:  []schema.GroupResource{gr("", "configmaps")},
			disabled: []schema.GroupResource{gr("coordination.k8s.io", "leases")},
		},
		{
			name:     "an earlier Deflate wins over a later None",
			entries:  []entryConfig{entry(deflate, "configmaps"), entry(none, "*.")},
			enabled:  []schema.GroupResource{gr("", "configmaps")},
			disabled: []schema.GroupResource{gr("", "pods")},
		},
		{
			name:     "secrets named explicitly",
			entries:  []entryConfig{entry(deflate, "secrets")},
			enabled:  []schema.GroupResource{gr("", "secrets")},
			disabled: []schema.GroupResource{gr("", "configmaps")},
		},
		{
			name:     "the core group wildcard does not reach secrets",
			entries:  []entryConfig{entry(deflate, "*.")},
			disabled: []schema.GroupResource{gr("", "secrets")},
		},
		{
			name:    "secrets named after a wildcard still works, since no wildcard claims it",
			entries: []entryConfig{entry(deflate, "*.*"), entry(deflate, "secrets")},
			enabled: []schema.GroupResource{gr("", "secrets"), gr("", "configmaps")},
		},
		// Validation rejects an unknown algorithm at startup. If a caller skips
		// validation, NewPolicy still only compresses for Deflate, so it never
		// writes a frame no reader can decode.
		{
			name:     "an algorithm validation would reject selects nothing",
			entries:  []entryConfig{entry("Zstd", "configmaps")},
			disabled: []schema.GroupResource{gr("", "configmaps")},
		},
		{
			name:     "a secrets resource in another group is not the sensitive one",
			entries:  []entryConfig{entry(deflate, "*.example.com")},
			enabled:  []schema.GroupResource{gr("example.com", "secrets")},
			disabled: []schema.GroupResource{gr("", "secrets")},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := newPolicy(tc.entries...)
			for _, r := range tc.enabled {
				if !p.EnabledFor(r) {
					t.Errorf("EnabledFor(%s) = false, want true", r)
				}
			}
			for _, r := range tc.disabled {
				if p.EnabledFor(r) {
					t.Errorf("EnabledFor(%s) = true, want false", r)
				}
			}
		})
	}
}

func TestMinSizeFor(t *testing.T) {
	p := newPolicy(
		withMinSize(entry(deflate, "configmaps"), 256),
		withMinSize(entry(none, "events"), 4096),
		entry(deflate, "*.*"),
	)
	for _, c := range []struct {
		gr   schema.GroupResource
		want int
	}{
		{gr("", "configmaps"), 256},
		{gr("", "events"), 4096},
		{gr("", "pods"), DefaultMinSize},
		{SecretsGroupResource, DefaultMinSize},
	} {
		if got := p.MinSizeFor(c.gr); got != c.want {
			t.Errorf("MinSizeFor(%s) = %d, want %d", c.gr, got, c.want)
		}
	}
}

func TestNilPolicyIsInert(t *testing.T) {
	var p *Policy
	if p.EnabledFor(gr("", "configmaps")) || p.MinSizeFor(gr("", "configmaps")) != DefaultMinSize {
		t.Fatal("a nil policy must select nothing")
	}
	if p := NewPolicy(nil); p != nil {
		t.Fatalf("NewPolicy(nil) = %v, want nil", p)
	}
}
