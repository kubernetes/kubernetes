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

package compressionconfig

import (
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/apis/apiserver"
	"k8s.io/apiserver/pkg/apis/apiserver/validation"
	"k8s.io/apiserver/pkg/storage/value/compression"
)

const validConfig = `
apiVersion: apiserver.config.k8s.io/v1alpha1
kind: StorageCompressionConfiguration
resources:
- resources: ["events"]
  algorithm: None
- resources: ["configmaps"]
  algorithm: Deflate
  minSize: 2Ki
- resources: ["*.*"]
  algorithm: Deflate
`

type entryConfig = apiserver.CompressionResourceConfiguration

func writeConfig(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadCompressionConfig(t *testing.T) {
	policy, hash, err := LoadCompressionConfig(writeConfig(t, validConfig))
	if err != nil {
		t.Fatal(err)
	}
	configmaps, pods, events := schema.GroupResource{Resource: "configmaps"}, schema.GroupResource{Resource: "pods"}, schema.GroupResource{Resource: "events"}
	if !policy.EnabledFor(configmaps) || !policy.EnabledFor(pods) || policy.EnabledFor(events) || policy.EnabledFor(compression.SecretsGroupResource) {
		t.Error("the loaded policy doesn't select what the file says")
	}
	if got := policy.MinSizeFor(configmaps); got != 2048 {
		t.Errorf("MinSizeFor(configmaps) = %d, want 2048 from 2Ki", got)
	}
	if got := policy.MinSizeFor(pods); got != compression.DefaultMinSize {
		t.Errorf("MinSizeFor(pods) = %d, want the default %d when minSize is omitted", got, compression.DefaultMinSize)
	}
	if want := fmt.Sprintf("sha256:%x", sha256.Sum256([]byte(validConfig))); hash != want {
		t.Errorf("hash = %q, want %q", hash, want)
	}
	if _, other, err := LoadCompressionConfig(writeConfig(t, validConfig+"\n")); err != nil || other == hash {
		t.Errorf("a different file must hash differently: %q, %v", other, err)
	}
}

// TestLoadCompressionConfigSelectingNothing checks that a file with no entries
// loads and compresses nothing, so emptying the list turns compression off
// without failing startup.
func TestLoadCompressionConfigSelectingNothing(t *testing.T) {
	header := "apiVersion: apiserver.config.k8s.io/v1alpha1\nkind: StorageCompressionConfiguration\n"
	for name, content := range map[string]string{
		"an empty list":    header + "resources: []\n",
		"no resources key": header,
	} {
		t.Run(name, func(t *testing.T) {
			policy, hash, err := LoadCompressionConfig(writeConfig(t, content))
			if err != nil {
				t.Fatal(err)
			}
			if policy.EnabledFor(schema.GroupResource{Resource: "configmaps"}) {
				t.Error("a file with no entries must compress nothing")
			}
			if hash == "" {
				t.Error("the hash must still be reported, so replicas can be compared")
			}
		})
	}
}

func TestLoadCompressionConfigRejects(t *testing.T) {
	for _, tc := range []struct {
		name    string
		content string
		noFile  bool
		wantErr string
	}{
		{name: "a missing file", noFile: true, wantErr: "error reading"},
		{name: "an empty file", wantErr: "is empty"},
		{
			name:    "an unknown field",
			content: strings.Replace(validConfig, "minSize: 2Ki", "minsize: 2Ki", 1),
			wantErr: `unknown field "resources[1].minsize"`,
		},
		{
			name:    "another kind",
			content: "apiVersion: apiserver.config.k8s.io/v1alpha1\nkind: TracingConfiguration\n",
			wantErr: "got unexpected config type",
		},
		{
			name:    "a version that doesn't define the kind",
			content: strings.Replace(validConfig, "v1alpha1", "v1", 1),
			wantErr: "error decoding",
		},
		{
			name:    "a minSize that isn't a quantity",
			content: strings.Replace(validConfig, "minSize: 2Ki", "minSize: 2KB", 1),
			wantErr: "error decoding",
		},
		{
			name:    "a configuration that fails validation",
			content: strings.Replace(validConfig, "algorithm: None", "algorithm: Zstd", 1),
			wantErr: "resources[0].algorithm: Unsupported value",
		},
		{
			name:    "a masked selector",
			content: validConfig + "- resources: [\"pods\"]\n  algorithm: None\n",
			wantErr: `resources[3].resources[0]: Invalid value: "pods": masked by earlier rule "*.*"`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "missing.yaml")
			if !tc.noFile {
				path = writeConfig(t, tc.content)
			}
			_, _, err := LoadCompressionConfig(path)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) || !strings.Contains(err.Error(), path) {
				t.Fatalf("LoadCompressionConfig() = %v, want an error naming the file and containing %q", err, tc.wantErr)
			}
		})
	}
}

// firstMatch applies the documented rule literally, without CoveringSelectors:
// walk the entries in order and take the first one that selects gr. Wildcards
// never select core secrets.
func firstMatch(entries []entryConfig, gr schema.GroupResource) (entryConfig, bool) {
	for _, e := range entries {
		wildcard := false
		for _, raw := range e.Resources {
			sel := schema.ParseGroupResource(raw)
			switch {
			case sel == gr:
				return e, true
			case sel.Resource == "*" && (sel.Group == "*" || sel.Group == gr.Group):
				wildcard = true
			}
		}
		if wildcard && gr != compression.SecretsGroupResource {
			return e, true
		}
	}
	return entryConfig{}, false
}

// masked reports whether some selector is already decided by the selectors
// ahead of it, including those earlier in its own entry.
func masked(entries []entryConfig) bool {
	for i, e := range entries {
		for j, raw := range e.Resources {
			ahead := append(slices.Clone(entries[:i]), entryConfig{Resources: e.Resources[:j]})
			if _, ok := firstMatch(ahead, schema.ParseGroupResource(raw)); ok {
				return true
			}
		}
	}
	return false
}

// TestValidatedPolicyFollowsFirstMatch tries every configuration of one or two
// entries (one or two selectors each) and of three single-selector entries,
// drawn from a fixed selector pool. Validation and Policy share
// compression.CoveringSelectors, so this pins both to first-match-wins: masked
// configs fail validation, and accepted ones pick the entry firstMatch picks.
func TestValidatedPolicyFollowsFirstMatch(t *testing.T) {
	selectors := []string{"configmaps", "secrets", "*.", "*.*", "*.apps", "deployments.apps", "widgets.example.com"}
	var lists [][]string
	for _, a := range selectors {
		lists = append(lists, []string{a})
		for _, b := range selectors {
			if a != b {
				lists = append(lists, []string{a, b})
			}
		}
	}
	probes := []schema.GroupResource{
		{Resource: "configmaps"}, {Resource: "pods"}, compression.SecretsGroupResource, {Group: "example.com", Resource: "secrets"},
		{Group: "apps", Resource: "deployments"}, {Group: "apps", Resource: "replicasets"},
		{Group: "example.com", Resource: "widgets"}, {Group: "example.com", Resource: "gadgets"},
	}
	// Entry i gets minSize MinConfigurableMinSize+i, so MinSizeFor tells which entry won.
	build := func(lists ...[]string) []entryConfig {
		entries := make([]entryConfig, len(lists))
		for i, l := range lists {
			alg := apiserver.CompressionAlgorithmDeflate
			if i%2 == 1 {
				alg = apiserver.CompressionAlgorithmNone
			}
			minSize := resource.NewQuantity(int64(compression.MinConfigurableMinSize+i), resource.BinarySI)
			entries[i] = entryConfig{Resources: l, Algorithm: alg, MinSize: minSize}
		}
		return entries
	}
	var configs [][]entryConfig
	for _, a := range lists {
		configs = append(configs, build(a))
		for _, b := range lists {
			configs = append(configs, build(a, b))
		}
	}
	for _, a := range selectors {
		for _, b := range selectors {
			for _, c := range selectors {
				configs = append(configs, build([]string{a}, []string{b}, []string{c}))
			}
		}
	}

	accepted := 0
	for _, entries := range configs {
		name := fmt.Sprint(entries)
		config := &apiserver.StorageCompressionConfiguration{Resources: entries}
		errs := validation.ValidateStorageCompressionConfiguration(config)
		if masked(entries) && len(errs) == 0 {
			t.Fatalf("%s: a masked configuration passed validation", name)
		}
		if len(errs) > 0 {
			continue
		}
		accepted++
		p := compression.NewPolicy(config)
		for _, probe := range probes {
			want, ok := firstMatch(entries, probe)
			wantEnabled, wantMinSize := false, compression.DefaultMinSize
			if ok {
				wantEnabled, wantMinSize = want.Algorithm == apiserver.CompressionAlgorithmDeflate, int(want.MinSize.Value())
			}
			if got := p.EnabledFor(probe); got != wantEnabled {
				t.Fatalf("%s: EnabledFor(%s) = %v, want %v", name, probe, got, wantEnabled)
			}
			if got := p.MinSizeFor(probe); got != wantMinSize {
				t.Fatalf("%s: MinSizeFor(%s) = %d, want %d", name, probe, got, wantMinSize)
			}
		}
	}
	if accepted == 0 || accepted == len(configs) {
		t.Fatalf("%d of %d configurations accepted; the generator isn't exercising both outcomes", accepted, len(configs))
	}
}
