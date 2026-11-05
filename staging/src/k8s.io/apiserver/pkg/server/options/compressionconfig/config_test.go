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
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/validation/field"
	apiserverv1alpha1 "k8s.io/apiserver/pkg/apis/apiserver/v1alpha1"
	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/apiserver/pkg/storage/value/compression"
	"k8s.io/apiserver/pkg/storage/value/encrypt/identity"
)

const validConfig = `
apiVersion: apiserver.config.k8s.io/v1alpha1
kind: StorageCompressionConfiguration
compressionResources:
- resources: ["events"]
  algorithm: None
- resources: ["configmaps"]
  algorithm: Deflate
  minSize: 2Ki
- resources: ["*.*"]
  algorithm: Deflate
`

var (
	configmaps = gr("", "configmaps")
	pods       = gr("", "pods")
	events     = gr("", "events")
	// secrets is spelled out rather than taken from storagecompression, so the
	// firstMatch oracle doesn't share code with what it checks.
	secrets = gr("", "secrets")
)

const (
	deflate = apiserverv1alpha1.CompressionAlgorithmDeflate
	none    = apiserverv1alpha1.CompressionAlgorithmNone
)

type entryConfig = apiserverv1alpha1.CompressionResourceConfiguration

func gr(group, resource string) schema.GroupResource {
	return schema.GroupResource{Group: group, Resource: resource}
}

func entry(algorithm apiserverv1alpha1.CompressionAlgorithm, resources ...string) entryConfig {
	return entryConfig{Resources: resources, Algorithm: algorithm}
}

func withMinSize(e entryConfig, minSize int64) entryConfig {
	e.MinSize = resource.NewQuantity(minSize, resource.BinarySI)
	return e
}

func mustConfig(t *testing.T, entries ...entryConfig) *CompressionConfiguration {
	t.Helper()
	c, err := newCompressionConfiguration(&apiserverv1alpha1.StorageCompressionConfiguration{CompressionResources: entries}, "")
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func writeConfig(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadCompressionConfig(t *testing.T) {
	c, err := LoadCompressionConfig(writeConfig(t, validConfig))
	if err != nil {
		t.Fatal(err)
	}
	for gr, want := range map[schema.GroupResource]compression.Rule{
		configmaps: {Algorithm: compression.AlgorithmDeflate, MinSize: 2048},
		pods:       {Algorithm: compression.AlgorithmDeflate, MinSize: 1024},
		events:     {MinSize: 1024},
		secrets:    {},
	} {
		if got := c.RuleFor(gr); got != want {
			t.Errorf("RuleFor(%s) = %+v, want %+v", gr, got, want)
		}
	}
	if want := fmt.Sprintf("sha256:%x", sha256.Sum256([]byte(validConfig))); c.CompressionFileContentHash != want {
		t.Errorf("hash = %q, want %q", c.CompressionFileContentHash, want)
	}
	if other, err := LoadCompressionConfig(writeConfig(t, validConfig+"\n")); err != nil || other.CompressionFileContentHash == c.CompressionFileContentHash {
		t.Errorf("a different file must hash differently: %v", err)
	}
}

// TestLoadCompressionConfigSelectingNothing checks that a file with no entries
// loads and compresses nothing.
func TestLoadCompressionConfigSelectingNothing(t *testing.T) {
	header := "apiVersion: apiserver.config.k8s.io/v1alpha1\nkind: StorageCompressionConfiguration\n"
	for name, content := range map[string]string{
		"an empty list":               header + "compressionResources: []\n",
		"no compressionResources key": header,
	} {
		t.Run(name, func(t *testing.T) {
			c, err := LoadCompressionConfig(writeConfig(t, content))
			if err != nil {
				t.Fatal(err)
			}
			if got := c.RuleFor(configmaps); got != (compression.Rule{}) {
				t.Errorf("RuleFor(configmaps) = %+v, want the zero Rule", got)
			}
			if c.CompressionFileContentHash == "" {
				t.Error("the hash must still be reported, so replicas can be compared")
			}
		})
	}
}

// TestLoadCompressionConfigRejects checks that LoadCompressionConfig returns no
// config for every bad file, and an error that includes the file's path. Read
// and decode errors are the loader's own, so they're matched by message.
// Validation errors are matched by type, field and origin instead, so rewording
// a validation message doesn't break the test.
func TestLoadCompressionConfigRejects(t *testing.T) {
	root := field.NewPath("compressionResources")
	matcher := field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByDetailSubstring().RequireOriginWhenInvalid()
	for _, tc := range []struct {
		name      string
		content   string
		noFile    bool
		wantErr   string
		wantField field.ErrorList
	}{
		{name: "a missing file", noFile: true, wantErr: "error reading"},
		{name: "an empty file", wantErr: "is empty"},
		{
			name:    "an unknown field",
			content: strings.Replace(validConfig, "minSize: 2Ki", "minsize: 2Ki", 1),
			wantErr: `unknown field "compressionResources[1].minsize"`,
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
			name:      "an unknown algorithm",
			content:   strings.Replace(validConfig, "algorithm: None", "algorithm: Zstd", 1),
			wantField: field.ErrorList{field.NotSupported[string](root.Index(0).Child("algorithm"), nil, nil)},
		},
		{
			name:      "a minSize below the floor",
			content:   strings.Replace(validConfig, "minSize: 2Ki", "minSize: 63", 1),
			wantField: field.ErrorList{field.Invalid(root.Index(1).Child("minSize"), nil, "").WithOrigin("minimum")},
		},
		{
			name:    "a masked selector",
			content: validConfig + "- resources: [\"pods\"]\n  algorithm: None\n",
			wantField: field.ErrorList{
				field.Invalid(root.Index(3).Child("resources").Index(0), nil, "compressionResources[2].resources[0]").WithOrigin("masked"),
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "missing.yaml")
			if !tc.noFile {
				path = writeConfig(t, tc.content)
			}
			c, err := LoadCompressionConfig(path)
			if err == nil || !strings.Contains(err.Error(), path) {
				t.Fatalf("LoadCompressionConfig() = %v, want an error containing the file's path", err)
			}
			if tc.wantField != nil {
				got := fieldErrors(t, err)
				matcher.Test(t, tc.wantField, got)
				if len(got) != len(tc.wantField) {
					t.Errorf("got %d errors, want %d: %v", len(got), len(tc.wantField), err)
				}
			} else if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("LoadCompressionConfig() = %v, want an error containing %q", err, tc.wantErr)
			}
			if c != nil {
				t.Errorf("LoadCompressionConfig() returned %+v with its error, want nil", c)
			}
		})
	}
}

// fieldErrors unwraps LoadCompressionConfig's aggregate into its *field.Errors.
func fieldErrors(t *testing.T, err error) field.ErrorList {
	t.Helper()
	var agg utilerrors.Aggregate
	if !errors.As(err, &agg) {
		t.Fatalf("LoadCompressionConfig() = %v, want an aggregate of validation errors", err)
	}
	var errs field.ErrorList
	for _, e := range agg.Errors() {
		var fe *field.Error
		if !errors.As(e, &fe) {
			t.Fatalf("%v is not a *field.Error", e)
		}
		errs = append(errs, fe)
	}
	return errs
}

// TestValidateConfig checks that validateConfig reports a selector both
// ValidateSelectors and ValidateMasking catch only once. It reads the errors
// before ToAggregate, which would drop the repeat.
func TestValidateConfig(t *testing.T) {
	selector := func(i, j int) *field.Path {
		return field.NewPath("compressionResources").Index(i).Child("resources").Index(j)
	}
	got := validateConfig(&apiserverv1alpha1.StorageCompressionConfiguration{CompressionResources: []entryConfig{
		entry(deflate, "configmaps"),
		entry(deflate, "configmaps", "configmaps"),
	}})
	want := field.ErrorList{field.Duplicate(selector(1, 0), nil), field.Duplicate(selector(1, 1), nil)}
	field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByDetailSubstring().RequireOriginWhenInvalid().Test(t, want, got)
}

// TestLoadCompressionConfigReportsEachErrorOnce checks that each problem in a
// file gets exactly one error, whether the tags or the selector checks catch
// it. It can't see an error reported twice word for word, since ToAggregate
// drops exact repeats; TestValidateConfig catches those.
func TestLoadCompressionConfigReportsEachErrorOnce(t *testing.T) {
	content := `
apiVersion: apiserver.config.k8s.io/v1alpha1
kind: StorageCompressionConfiguration
compressionResources:
- resources: []
  algorithm: Deflate
- resources: ["configmaps"]
- resources: ["pods", "pods"]
  algorithm: Zstd
  minSize: 63
- resources: ["*.*"]
  algorithm: Deflate
  minSize: 1472Ki
- resources: ["services"]
  algorithm: None
`
	_, err := LoadCompressionConfig(writeConfig(t, content))
	root := field.NewPath("compressionResources")
	want := field.ErrorList{
		field.Required(root.Index(0).Child("resources"), ""),
		field.Required(root.Index(1).Child("algorithm"), ""),
		field.Duplicate(root.Index(2).Child("resources").Index(1), nil),
		field.NotSupported[string](root.Index(2).Child("algorithm"), nil, nil),
		field.Invalid(root.Index(2).Child("minSize"), nil, "").WithOrigin("minimum"),
		field.Invalid(root.Index(3).Child("minSize"), nil, "").WithOrigin("maximum"),
		field.Invalid(root.Index(4).Child("resources").Index(0), nil, "compressionResources[3].resources[0]").WithOrigin("masked"),
	}
	got := fieldErrors(t, err)
	field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByDetailSubstring().RequireOriginWhenInvalid().Test(t, want, got)
	// One want with an origin can match several errors, so compare the counts too.
	if len(got) != len(want) {
		t.Errorf("got %d errors, want %d: %v", len(got), len(want), err)
	}
}

func TestRuleFor(t *testing.T) {
	cases := []struct {
		name     string
		entries  []entryConfig
		enabled  []schema.GroupResource
		disabled []schema.GroupResource
	}{
		{
			name:     "no entries selects nothing",
			disabled: []schema.GroupResource{configmaps, secrets, gr("example.com", "widgets")},
		},
		{
			name:     "bare resource is core group",
			entries:  []entryConfig{entry(deflate, "configmaps")},
			enabled:  []schema.GroupResource{configmaps},
			disabled: []schema.GroupResource{gr("apps", "configmaps"), pods},
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
			disabled: []schema.GroupResource{configmaps, gr("other.com", "widgets")},
		},
		{
			name:     "core group wildcard",
			entries:  []entryConfig{entry(deflate, "*.")},
			enabled:  []schema.GroupResource{configmaps, pods},
			disabled: []schema.GroupResource{gr("apps", "deployments")},
		},
		{
			name:     "everything wildcard",
			entries:  []entryConfig{entry(deflate, "*.*")},
			enabled:  []schema.GroupResource{configmaps, gr("apps", "deployments"), gr("example.com", "widgets")},
			disabled: []schema.GroupResource{secrets},
		},
		{
			name:     "algorithm None selects nothing",
			entries:  []entryConfig{entry(none, "*.*")},
			disabled: []schema.GroupResource{configmaps, gr("apps", "deployments")},
		},
		{
			name:     "earlier None excludes a resource from a later wildcard",
			entries:  []entryConfig{entry(none, "events"), entry(deflate, "*.*")},
			enabled:  []schema.GroupResource{configmaps, gr("apps", "deployments")},
			disabled: []schema.GroupResource{events},
		},
		{
			name:     "earlier None excludes a whole group from a later wildcard",
			entries:  []entryConfig{entry(none, "*.coordination.k8s.io"), entry(deflate, "*.*")},
			enabled:  []schema.GroupResource{configmaps},
			disabled: []schema.GroupResource{gr("coordination.k8s.io", "leases")},
		},
		{
			name:     "an earlier Deflate wins over a later None",
			entries:  []entryConfig{entry(deflate, "configmaps"), entry(none, "*.")},
			enabled:  []schema.GroupResource{configmaps},
			disabled: []schema.GroupResource{pods},
		},
		{
			name:     "secrets named explicitly",
			entries:  []entryConfig{entry(deflate, "secrets")},
			enabled:  []schema.GroupResource{secrets},
			disabled: []schema.GroupResource{configmaps},
		},
		{
			name:     "secrets named with None",
			entries:  []entryConfig{entry(none, "secrets"), entry(deflate, "*.*")},
			enabled:  []schema.GroupResource{configmaps},
			disabled: []schema.GroupResource{secrets},
		},
		{
			name:     "the core group wildcard does not reach secrets",
			entries:  []entryConfig{entry(deflate, "*.")},
			disabled: []schema.GroupResource{secrets},
		},
		{
			name:    "secrets named after a wildcard still works, since no wildcard claims it",
			entries: []entryConfig{entry(deflate, "*.*"), entry(deflate, "secrets")},
			enabled: []schema.GroupResource{secrets, configmaps},
		},
		{
			name:     "a secrets resource in another group is not the sensitive one",
			entries:  []entryConfig{entry(deflate, "*.example.com")},
			enabled:  []schema.GroupResource{gr("example.com", "secrets")},
			disabled: []schema.GroupResource{secrets},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := mustConfig(t, tc.entries...)
			for _, r := range tc.enabled {
				if got := c.RuleFor(r).Algorithm; got != compression.AlgorithmDeflate {
					t.Errorf("RuleFor(%s).Algorithm = %v, want AlgorithmDeflate", r, got)
				}
			}
			for _, r := range tc.disabled {
				if got := c.RuleFor(r).Algorithm; got != compression.AlgorithmNone {
					t.Errorf("RuleFor(%s).Algorithm = %v, want AlgorithmNone", r, got)
				}
			}
		})
	}
}

// TestRuleForMinSize checks that each resource gets the minSize of the entry
// that decides it, so a named resource and a later wildcard can have different
// thresholds. An entry without minSize gets the 1Ki default, and a resource no
// entry selects gets the zero Rule.
func TestRuleForMinSize(t *testing.T) {
	fractional := entry(deflate, "deployments.apps")
	fractional.MinSize = new(resource.MustParse("100.5"))
	c := mustConfig(t,
		withMinSize(entry(deflate, "configmaps"), 256),
		withMinSize(entry(none, "events"), 4096),
		fractional,
		entry(deflate, "*.*"),
	)
	for gr, want := range map[schema.GroupResource]int{
		configmaps: 256,
		events:     4096,
		// A fractional minSize rounds up, so 100.5 compresses values of 101 bytes or more.
		gr("apps", "deployments"): 101,
		pods:                      1024,
		secrets:                   0,
	} {
		if got := c.RuleFor(gr).MinSize; got != want {
			t.Errorf("RuleFor(%s).MinSize = %d, want %d", gr, got, want)
		}
	}
}

// TestUnloadedConfigurationIsInert checks that RuleFor gives the zero Rule for
// a nil CompressionConfiguration, which a server without the flag has, and for
// a zero one built outside the loader.
func TestUnloadedConfigurationIsInert(t *testing.T) {
	for name, c := range map[string]*CompressionConfiguration{"nil": nil, "zero": {}} {
		if got := c.RuleFor(configmaps); got != (compression.Rule{}) {
			t.Errorf("%s: RuleFor(configmaps) = %+v, want the zero Rule", name, got)
		}
	}
}

// TestRulesReachTheTransformer wires a loaded file up the way the storage
// factory will (RuleFor per resource, then compression.WithCompression) and
// checks which values come out framed. An omitted minSize must act as 1Ki.
func TestRulesReachTheTransformer(t *testing.T) {
	loaded, err := LoadCompressionConfig(writeConfig(t, validConfig))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		config     *CompressionConfiguration
		gr         schema.GroupResource
		size       int
		wantFramed bool
	}{
		{loaded, configmaps, 2047, false},
		{loaded, configmaps, 2048, true},
		{loaded, pods, 1023, false},
		{loaded, pods, 1024, true},
		{loaded, events, 4096, false},
		{loaded, secrets, 4096, false},
		// No --storage-compression-config: the wiring still installs the transformer.
		{nil, configmaps, 4096, false},
	} {
		t.Run(fmt.Sprintf("%s/%d/loaded=%v", tc.gr, tc.size, tc.config != nil), func(t *testing.T) {
			tr := compression.WithCompression(identity.NewEncryptCheckTransformer(), tc.config.RuleFor(tc.gr), tc.gr)
			data := bytes.Repeat([]byte("a"), tc.size)
			dataCtx := value.DefaultContext("/registry/" + tc.gr.String())
			stored, err := tr.TransformToStorage(context.Background(), data, dataCtx)
			if err != nil {
				t.Fatal(err)
			}
			if framed := stored[0] == 0x00; framed != tc.wantFramed {
				t.Fatalf("framed = %v, want %v", framed, tc.wantFramed)
			}
			out, stale, err := tr.TransformFromStorage(context.Background(), stored, dataCtx)
			if err != nil || stale || !bytes.Equal(out, data) {
				t.Fatalf("round trip: stale %v, err %v", stale, err)
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
		if wildcard && gr != secrets {
			return e, true
		}
	}
	return entryConfig{}, false
}

// masked reports whether any selector is already matched by one before it, in
// an earlier entry or earlier in its own.
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

// TestRuleForFollowsFirstMatch checks validation and RuleFor against firstMatch
// over every config of one or two entries (one or two selectors each) and of
// three one-selector entries from a fixed pool. A masked config must fail
// newCompressionConfiguration, and an accepted one must give each probe the
// Rule of the entry firstMatch picks.
func TestRuleForFollowsFirstMatch(t *testing.T) {
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
		configmaps, pods, secrets, {Group: "example.com", Resource: "secrets"},
		{Group: "apps", Resource: "deployments"}, {Group: "apps", Resource: "replicasets"},
		{Group: "example.com", Resource: "widgets"}, {Group: "example.com", Resource: "gadgets"},
	}
	// Entry i gets minSize 64+i (64 is the floor), so the Rule shows which entry won.
	build := func(lists ...[]string) []entryConfig {
		entries := make([]entryConfig, len(lists))
		for i, l := range lists {
			alg := deflate
			if i%2 == 1 {
				alg = none
			}
			entries[i] = withMinSize(entry(alg, l...), int64(64+i))
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
		c, err := newCompressionConfiguration(&apiserverv1alpha1.StorageCompressionConfiguration{CompressionResources: entries}, "")
		if masked(entries) && err == nil {
			t.Fatalf("%s: a masked configuration passed validation", name)
		}
		if err != nil {
			continue
		}
		accepted++
		for _, probe := range probes {
			var want compression.Rule
			if e, ok := firstMatch(entries, probe); ok {
				want.MinSize = int(e.MinSize.Value())
				if e.Algorithm == deflate {
					want.Algorithm = compression.AlgorithmDeflate
				}
			}
			if got := c.RuleFor(probe); got != want {
				t.Fatalf("%s: RuleFor(%s) = %+v, want %+v", name, probe, got, want)
			}
		}
	}
	if accepted == 0 || accepted == len(configs) {
		t.Fatalf("%d of %d configurations accepted; the generator isn't exercising both outcomes", accepted, len(configs))
	}
}

// TestNewCompressionConfigurationValidates checks that the only constructor,
// newCompressionConfiguration, rejects a nil config and runs both the tags and
// the selector checks, so no CompressionConfiguration holds rules from an
// invalid config.
func TestNewCompressionConfigurationValidates(t *testing.T) {
	for name, config := range map[string]*apiserverv1alpha1.StorageCompressionConfiguration{
		"nil":                                nil,
		"missing algorithm, caught by a tag": {CompressionResources: []entryConfig{entry("", "configmaps")}},
		"masked selector, caught by the selector checks": {CompressionResources: []entryConfig{entry(deflate, "*.*"), entry(none, "configmaps")}},
	} {
		t.Run(name, func(t *testing.T) {
			c, err := newCompressionConfiguration(config, "")
			if err == nil || c != nil {
				t.Fatalf("newCompressionConfiguration() = %+v, %v, want only a validation error", c, err)
			}
		})
	}
}
