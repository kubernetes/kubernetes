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

package validation

import (
	"fmt"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/cel/environment"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/dynamic-resource-allocation/cel/sharingaffinity"
	resourceapi "k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/features"
)

func testResourceSliceWithSharingAffinity(name, poolName, driverName string, numExtractors int) *resourceapi.ResourceSlice {
	slice := testResourceSlice(name, poolName, driverName, 1)
	slice.Spec.Devices = nil
	for i := 0; i < numExtractors; i++ {
		slice.Spec.SharingAffinity = append(slice.Spec.SharingAffinity, resourceapi.SharingAffinityExtractor{
			Name:       fmt.Sprintf("key-%d", i),
			APIGroup:   "networking.example.com",
			Kind:       "NICConfig",
			Expression: `parameters.?subnetID.orValue("")`,
		})
	}
	return slice
}

// These fixtures measure the shared compiler rather than encoding estimator probe
// numbers. Two expensive expressions leave a small, exactly tunable remainder.
func sharingAffinityBudgetExpressions(t *testing.T, total uint64) []string {
	t.Helper()
	budget := uint64(resourceapi.SharingAffinityCELMaxCost)
	if total != budget && total != budget+1 {
		t.Fatalf("unsupported aggregate boundary %d", total)
	}
	cost := func(expr string) uint64 {
		t.Helper()
		if len(expr) > resourceapi.SharingAffinityCELExpressionMaxLength {
			t.Fatalf("cost fixture is %d bytes, exceeds expression limit", len(expr))
		}
		result := sharingaffinity.ValidateExpression(expr, environment.NewExpressions)
		if result.Error != nil {
			t.Fatalf("compile cost fixture: %v", result.Error)
		}
		return result.MaxCost
	}
	expensive := func(n int) string {
		return fmt.Sprintf("%q.matches(%q) ? 'x' : 'x'", strings.Repeat("a", 4000), strings.Repeat("a", n))
	}
	lo, hi := 1, 5000
	for lo < hi {
		mid := (lo + hi + 1) / 2
		if cost(expensive(mid)) <= (budget-10)/2 {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	base := expensive(lo)
	baseCost := cost(base)
	if baseCost == 0 || baseCost >= uint64(resourceapi.SharingAffinityCELMaxCost) {
		t.Fatalf("base cost %d must be positive and individually below the budget", baseCost)
	}
	remainder := budget - 2*baseCost
	padding := func(n int) string {
		return fmt.Sprintf("%q.matches('a') ? 'x' : 'x'", strings.Repeat("a", n))
	}
	lo, hi = 0, 10000
	for lo < hi {
		mid := (lo + hi) / 2
		if cost(padding(mid)) < remainder {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	filler := padding(lo)
	if cost(filler) != remainder {
		t.Fatalf("filler cost %d, want exactly %d", cost(filler), remainder)
	}
	if total == budget+1 {
		filler = "(" + filler + ") + ''"
		if cost(filler) != remainder+1 {
			t.Fatalf("padded filler cost %d, want exactly %d", cost(filler), remainder+1)
		}
	}
	expressions := []string{base, base, filler}
	var measured uint64
	for _, expr := range expressions {
		c := cost(expr)
		if c >= uint64(resourceapi.SharingAffinityCELMaxCost) {
			t.Fatalf("individual cost %d must be below the budget", c)
		}
		measured += c
	}
	if measured != total {
		t.Fatalf("measured aggregate cost %d, want exactly %d", measured, total)
	}
	return expressions
}

func TestValidateResourceSliceWithSharingAffinity(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.DRAConsumableCapacity: true, features.DRASharingAffinity: true,
	})

	goodName := "foo"
	driverName := "test.example.com"
	p := field.NewPath("spec", "sharingAffinity")
	entry := p.Index(0)
	const validExpression = `parameters.?subnetID.orValue("")`
	const dnsSubdomainError = "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"
	const dns1035LabelError = "a DNS-1035 label must consist of lower case alphanumeric characters or '-', start with an alphabetic character, and end with an alphanumeric character (e.g. 'my-name',  or 'abc-123', regex used for validation is '[a-z]([-a-z0-9]*[a-z0-9])?')"
	maxBytes := resourceapi.SharingAffinityCELExpressionMaxLength
	asciiAt := `"` + strings.Repeat("a", maxBytes-2) + `"`
	unicodeAt := `"` + strings.Repeat("é", (maxBytes-2)/2) + `"`
	unicodeOver := unicodeAt + " "
	if len(unicodeAt) != maxBytes || len(unicodeOver) != maxBytes+1 || utf8.RuneCountInString(unicodeOver) >= maxBytes {
		t.Fatal("invalid multibyte boundary fixtures")
	}
	withExtractor := func(extractor resourceapi.SharingAffinityExtractor) *resourceapi.ResourceSlice {
		slice := testResourceSliceWithSharingAffinity(goodName, goodName, driverName, 1)
		slice.Spec.SharingAffinity[0] = extractor
		return slice
	}
	scenarios := map[string]struct {
		slice        *resourceapi.ResourceSlice
		wantFailures field.ErrorList
	}{
		"invalid-sharing-affinity-with-devices": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec"), "{`devices`, `sharingAffinity`}", "only one of `devices`, `sharedCounters`, `sharingAffinity` is allowed"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.SharingAffinity = []resourceapi.SharingAffinityExtractor{
					{Name: "key", APIGroup: "*", Kind: "*", Expression: `"value"`},
				}
				return slice
			}(),
		},
		"invalid-sharing-affinity-with-shared-counters": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec"), "{`sharedCounters`, `sharingAffinity`}", "only one of `devices`, `sharedCounters`, `sharingAffinity` is allowed"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSliceWithSharedCounters(goodName, goodName, driverName, 1)
				slice.Spec.SharingAffinity = []resourceapi.SharingAffinityExtractor{
					{Name: "key", APIGroup: "*", Kind: "*", Expression: `"value"`},
				}
				return slice
			}(),
		},
		"invalid-sharing-affinity-with-devices-and-shared-counters": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec"), "{`devices`, `sharedCounters`, `sharingAffinity`}", "only one of `devices`, `sharedCounters`, `sharingAffinity` is allowed"),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSlice(goodName, goodName, driverName, 1)
				slice.Spec.SharedCounters = []resourceapi.CounterSet{
					{
						Name:     "counterset",
						Counters: testCounters(),
					},
				}
				slice.Spec.SharingAffinity = []resourceapi.SharingAffinityExtractor{
					{Name: "key", APIGroup: "*", Kind: "*", Expression: `"value"`},
				}
				return slice
			}(),
		},
		"valid-sharing-affinity-at-entry-limit": {
			slice: testResourceSliceWithSharingAffinity(goodName, goodName, driverName, resourceapi.SharingAffinityMaxEntries),
		},
		"invalid-sharing-affinity-over-entry-limit": {
			wantFailures: field.ErrorList{
				field.TooMany(p, resourceapi.SharingAffinityMaxEntries+1, resourceapi.SharingAffinityMaxEntries).WithOrigin("maxItems").MarkCoveredByDeclarative(),
			},
			slice: testResourceSliceWithSharingAffinity(goodName, goodName, driverName, resourceapi.SharingAffinityMaxEntries+1),
		},
		"invalid-sharing-affinity-with-duplicate-names": {
			wantFailures: field.ErrorList{
				field.Duplicate(p.Index(1), resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "b.example.com", Kind: "VFConfig", Expression: `"second"`}).MarkCoveredByDeclarative(),
			},
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSliceWithSharingAffinity(goodName, goodName, driverName, 1)
				slice.Spec.SharingAffinity = []resourceapi.SharingAffinityExtractor{
					{Name: "subnet", APIGroup: "a.example.com", Kind: "NICConfig", Expression: `"first"`},
					{Name: "subnet", APIGroup: "b.example.com", Kind: "VFConfig", Expression: `"second"`},
				}
				return slice
			}(),
		},
		"valid-sharing-affinity-with-identical-matchers-and-distinct-names": {
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSliceWithSharingAffinity(goodName, goodName, driverName, 1)
				slice.Spec.SharingAffinity = []resourceapi.SharingAffinityExtractor{
					{Name: "subnet", APIGroup: "*", Kind: "*", Expression: `"value"`},
					{Name: "pkey", APIGroup: "*", Kind: "*", Expression: `"value"`},
				}
				return slice
			}(),
		},
		"valid-sharing-affinity": {
			slice: withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: validExpression}),
		},
		"invalid-sharing-affinity-with-empty-name": {
			wantFailures: field.ErrorList{field.Required(entry.Child("name"), "").MarkCoveredByDeclarative()},
			slice:        withExtractor(resourceapi.SharingAffinityExtractor{APIGroup: "networking.example.com", Kind: "NICConfig", Expression: validExpression}),
		},
		"invalid-sharing-affinity-name-with-uppercase-letters": {
			wantFailures: field.ErrorList{
				field.Invalid(entry.Child("name"), "Subnet", "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')").WithOrigin("format=k8s-short-name").MarkCoveredByDeclarative(),
			},
			slice: withExtractor(resourceapi.SharingAffinityExtractor{Name: "Subnet", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: validExpression}),
		},
		"invalid-sharing-affinity-name-with-dot": {
			wantFailures: field.ErrorList{field.Invalid(entry.Child("name"), "sub.net", "must not contain dots").WithOrigin("format=k8s-short-name").MarkCoveredByDeclarative()},
			slice:        withExtractor(resourceapi.SharingAffinityExtractor{Name: "sub.net", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: validExpression}),
		},
		"valid-sharing-affinity-name-at-length-limit": {
			slice: withExtractor(resourceapi.SharingAffinityExtractor{Name: strings.Repeat("a", 63), APIGroup: "networking.example.com", Kind: "NICConfig", Expression: validExpression}),
		},
		"invalid-sharing-affinity-name-over-length-limit": {
			wantFailures: field.ErrorList{field.Invalid(entry.Child("name"), strings.Repeat("a", 64), "must be no more than 63 characters").WithOrigin("format=k8s-short-name").MarkCoveredByDeclarative()},
			slice:        withExtractor(resourceapi.SharingAffinityExtractor{Name: strings.Repeat("a", 64), APIGroup: "networking.example.com", Kind: "NICConfig", Expression: validExpression}),
		},
		"invalid-sharing-affinity-with-empty-apiGroup": {
			wantFailures: field.ErrorList{field.Required(entry.Child("apiGroup"), "").MarkCoveredByDeclarative()},
			slice:        withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", Kind: "NICConfig", Expression: validExpression}),
		},
		"valid-sharing-affinity-with-wildcard-apiGroup": {
			slice: withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "*", Kind: "NICConfig", Expression: validExpression}),
		},
		"invalid-sharing-affinity-apiGroup-with-uppercase-letters": {
			wantFailures: field.ErrorList{field.Invalid(entry.Child("apiGroup"), "Example.com", dnsSubdomainError)},
			slice:        withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "Example.com", Kind: "NICConfig", Expression: validExpression}),
		},
		"invalid-sharing-affinity-apiGroup-with-version": {
			wantFailures: field.ErrorList{field.Invalid(entry.Child("apiGroup"), "example.com/v1", dnsSubdomainError)},
			slice:        withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "example.com/v1", Kind: "NICConfig", Expression: validExpression}),
		},
		"invalid-sharing-affinity-apiGroup-with-partial-wildcard": {
			wantFailures: field.ErrorList{field.Invalid(entry.Child("apiGroup"), "*.example.com", dnsSubdomainError)},
			slice:        withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "*.example.com", Kind: "NICConfig", Expression: validExpression}),
		},
		"valid-sharing-affinity-apiGroup-at-length-limit": {
			slice: withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: strings.Repeat("a", 253), Kind: "NICConfig", Expression: validExpression}),
		},
		"invalid-sharing-affinity-apiGroup-over-length-limit": {
			wantFailures: field.ErrorList{field.TooLongCharacters(entry.Child("apiGroup"), "", 253).WithOrigin("maxLength").MarkCoveredByDeclarative()},
			slice:        withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: strings.Repeat("a", 254), Kind: "NICConfig", Expression: validExpression}),
		},
		"invalid-sharing-affinity-apiGroup-non-ascii-at-character-limit": {
			wantFailures: field.ErrorList{field.Invalid(entry.Child("apiGroup"), strings.Repeat("é", 253), "must be no more than 253 characters, "+dnsSubdomainError)},
			slice:        withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: strings.Repeat("é", 253), Kind: "NICConfig", Expression: validExpression}),
		},
		"invalid-sharing-affinity-apiGroup-non-ascii-over-character-limit": {
			wantFailures: field.ErrorList{field.TooLongCharacters(entry.Child("apiGroup"), "", 253).WithOrigin("maxLength").MarkCoveredByDeclarative()},
			slice:        withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: strings.Repeat("é", 254), Kind: "NICConfig", Expression: validExpression}),
		},
		"invalid-sharing-affinity-with-empty-kind": {
			wantFailures: field.ErrorList{field.Required(entry.Child("kind"), "").MarkCoveredByDeclarative()},
			slice:        withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Expression: validExpression}),
		},
		"valid-sharing-affinity-with-wildcard-kind": {
			slice: withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: "*", Expression: validExpression}),
		},
		"valid-sharing-affinity-with-lowercase-kind": {
			slice: withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: "nicconfig", Expression: validExpression}),
		},
		"invalid-sharing-affinity-kind-with-slash": {
			wantFailures: field.ErrorList{field.Invalid(entry.Child("kind"), "NIC/Config", dns1035LabelError)},
			slice:        withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: "NIC/Config", Expression: validExpression}),
		},
		"invalid-sharing-affinity-kind-with-dot": {
			wantFailures: field.ErrorList{field.Invalid(entry.Child("kind"), "NIC.Config", dns1035LabelError)},
			slice:        withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: "NIC.Config", Expression: validExpression}),
		},
		"invalid-sharing-affinity-kind-with-leading-digit": {
			wantFailures: field.ErrorList{field.Invalid(entry.Child("kind"), "1Config", dns1035LabelError)},
			slice:        withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: "1Config", Expression: validExpression}),
		},
		"valid-sharing-affinity-kind-at-length-limit": {
			slice: withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: strings.Repeat("A", 63), Expression: validExpression}),
		},
		"invalid-sharing-affinity-kind-over-length-limit": {
			wantFailures: field.ErrorList{field.TooLongCharacters(entry.Child("kind"), "", 63).WithOrigin("maxLength").MarkCoveredByDeclarative()},
			slice:        withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: strings.Repeat("A", 64), Expression: validExpression}),
		},
		"invalid-sharing-affinity-kind-non-ascii-at-character-limit": {
			wantFailures: field.ErrorList{field.Invalid(entry.Child("kind"), strings.Repeat("é", 63), "must be no more than 63 characters, "+dns1035LabelError)},
			slice:        withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: strings.Repeat("é", 63), Expression: validExpression}),
		},
		"invalid-sharing-affinity-kind-non-ascii-over-character-limit": {
			wantFailures: field.ErrorList{field.TooLongCharacters(entry.Child("kind"), "", 63).WithOrigin("maxLength").MarkCoveredByDeclarative()},
			slice:        withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: strings.Repeat("é", 64), Expression: validExpression}),
		},
		"invalid-sharing-affinity-with-empty-expression": {
			wantFailures: field.ErrorList{field.Required(entry.Child("expression"), "").MarkCoveredByDeclarative()},
			slice:        withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: "NICConfig"}),
		},
		"valid-sharing-affinity-expression-with-empty-string-result": {
			slice: withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: `""`}),
		},
		"valid-sharing-affinity-expression-with-dynamic-result": {
			slice: withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: `parameters.subnetID`}),
		},
		"valid-sharing-affinity-expression-with-kind-guard": {
			slice: withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: `has(parameters.kind) && parameters.kind == "NICConfig" ? parameters.?subnetID.orValue("") : ""`}),
		},
		"valid-sharing-affinity-expression-with-optional-field-access": {
			slice: withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: validExpression}),
		},
		"valid-sharing-affinity-expression-with-bindings-extension": {
			slice: withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: `cel.bind(s, parameters.?subnetID.orValue(""), s)`}),
		},
		"invalid-sharing-affinity-expression-with-undeclared-object": {
			wantFailures: field.ErrorList{
				field.Invalid(entry.Child("expression"), `object.subnetID`, "compile CEL expression: ERROR: <input>:1:1: undeclared reference to 'object' (in container '')\n | object.subnetID\n | ^"),
			},
			slice: withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: `object.subnetID`}),
		},
		"invalid-sharing-affinity-expression-with-undeclared-device": {
			wantFailures: field.ErrorList{
				field.Invalid(entry.Child("expression"), `device.name`, "compile CEL expression: ERROR: <input>:1:1: undeclared reference to 'device' (in container '')\n | device.name\n | ^"),
			},
			slice: withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: `device.name`}),
		},
		"invalid-sharing-affinity-expression-with-syntax-error": {
			wantFailures: field.ErrorList{
				field.Invalid(entry.Child("expression"), `parameters.`, "compile CEL expression: ERROR: <input>:1:12: Syntax error: no viable alternative at input '.'\n | parameters.\n | ...........^"),
			},
			slice: withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: `parameters.`}),
		},
		"invalid-sharing-affinity-expression-with-type-error": {
			wantFailures: field.ErrorList{
				field.Invalid(entry.Child("expression"), `1 + true`, "compile CEL expression: ERROR: <input>:1:3: found no matching overload for '_+_' applied to '(int, bool)'\n | 1 + true\n | ..^"),
			},
			slice: withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: `1 + true`}),
		},
		"invalid-sharing-affinity-expression-with-bool-result": {
			wantFailures: field.ErrorList{field.Invalid(entry.Child("expression"), `true`, "CEL expression must evaluate to string, not bool")},
			slice:        withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: `true`}),
		},
		"invalid-sharing-affinity-expression-with-int-result": {
			wantFailures: field.ErrorList{field.Invalid(entry.Child("expression"), `1`, "CEL expression must evaluate to string, not int")},
			slice:        withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: `1`}),
		},
		"invalid-sharing-affinity-expression-with-double-result": {
			wantFailures: field.ErrorList{field.Invalid(entry.Child("expression"), `1.5`, "CEL expression must evaluate to string, not double")},
			slice:        withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: `1.5`}),
		},
		"invalid-sharing-affinity-expression-with-bytes-result": {
			wantFailures: field.ErrorList{field.Invalid(entry.Child("expression"), `b"value"`, "CEL expression must evaluate to string, not bytes")},
			slice:        withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: `b"value"`}),
		},
		"invalid-sharing-affinity-expression-with-list-result": {
			wantFailures: field.ErrorList{field.Invalid(entry.Child("expression"), `["value"]`, "CEL expression must evaluate to string, not list(string)")},
			slice:        withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: `["value"]`}),
		},
		"invalid-sharing-affinity-expression-with-map-result": {
			wantFailures: field.ErrorList{field.Invalid(entry.Child("expression"), `{"key": "value"}`, "CEL expression must evaluate to string, not map(string, string)")},
			slice:        withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: `{"key": "value"}`}),
		},
		"invalid-sharing-affinity-expression-with-null-result": {
			wantFailures: field.ErrorList{field.Invalid(entry.Child("expression"), `null`, "CEL expression must evaluate to string, not null_type")},
			slice:        withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: `null`}),
		},
		"valid-sharing-affinity-ascii-expression-at-byte-limit": {
			slice: withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: asciiAt}),
		},
		"invalid-sharing-affinity-ascii-expression-over-byte-limit": {
			wantFailures: field.ErrorList{field.TooLong(entry.Child("expression"), "", maxBytes).WithOrigin("maxBytes").MarkCoveredByDeclarative()},
			slice:        withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: asciiAt + " "}),
		},
		"valid-sharing-affinity-multibyte-expression-at-byte-limit": {
			slice: withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: unicodeAt}),
		},
		"invalid-sharing-affinity-multibyte-expression-over-byte-limit": {
			wantFailures: field.ErrorList{field.TooLong(entry.Child("expression"), "", maxBytes).WithOrigin("maxBytes").MarkCoveredByDeclarative()},
			slice:        withExtractor(resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: unicodeOver}),
		},
		"invalid-sharing-affinity-with-all-required-fields-empty": {
			wantFailures: field.ErrorList{
				field.Required(entry.Child("name"), "").MarkCoveredByDeclarative(),
				field.Required(entry.Child("apiGroup"), "").MarkCoveredByDeclarative(),
				field.Required(entry.Child("kind"), "").MarkCoveredByDeclarative(),
				field.Required(entry.Child("expression"), "").MarkCoveredByDeclarative(),
			},
			slice: withExtractor(resourceapi.SharingAffinityExtractor{}),
		},
		"valid-sharing-affinity-with-wildcard-apiGroup-and-kind": {
			slice: func() *resourceapi.ResourceSlice {
				slice := testResourceSliceWithSharingAffinity(goodName, goodName, driverName, 1)
				slice.Spec.SharingAffinity[0].APIGroup = "*"
				slice.Spec.SharingAffinity[0].Kind = "*"
				return slice
			}(),
		},
	}

	for name, scenario := range scenarios {
		t.Run("create/"+name, func(t *testing.T) {
			errs := ValidateResourceSlice(scenario.slice.DeepCopy())
			if assertFailures(t, scenario.wantFailures, errs) {
				// assertFailures deliberately ignores error origins.
				field.ErrorMatcher{}.ByOrigin().Test(t, scenario.wantFailures, errs)
			}
		})
		t.Run("new-entry/"+name, func(t *testing.T) {
			oldResourceSlice := scenario.slice.DeepCopy()
			oldResourceSlice.Spec.SharingAffinity = nil
			oldResourceSlice.Spec.Devices = nil
			oldResourceSlice.Spec.SharedCounters = nil
			oldResourceSlice.ResourceVersion = "1"
			updated := scenario.slice.DeepCopy()
			updated.ResourceVersion = "1"
			errs := ValidateResourceSliceUpdate(updated, oldResourceSlice)
			if assertFailures(t, scenario.wantFailures, errs) {
				// assertFailures deliberately ignores error origins.
				field.ErrorMatcher{}.ByOrigin().Test(t, scenario.wantFailures, errs)
			}
		})
	}
}

func TestValidateResourceSlice_SharingAffinityCost(t *testing.T) {
	for _, tc := range []struct {
		name string
		over uint64
	}{
		{"aggregate-exact", 0},
		{"aggregate-one-over", 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.DRAConsumableCapacity: true, features.DRASharingAffinity: true,
			})
			expressions := sharingAffinityBudgetExpressions(t, uint64(resourceapi.SharingAffinityCELMaxCost)+tc.over)
			slice := testResourceSliceWithSharingAffinity("slice", "pool", "driver.example.com", len(expressions))
			for i, expr := range expressions {
				slice.Spec.SharingAffinity[i].Expression = expr
			}
			var want field.ErrorList
			if tc.over != 0 {
				want = field.ErrorList{field.Forbidden(field.NewPath("spec", "sharingAffinity"), "")}
			}
			field.ErrorMatcher{}.ByType().ByField().Test(t, want, ValidateResourceSlice(slice))
			old := slice.DeepCopy()
			old.Spec.SharingAffinity = old.Spec.SharingAffinity[:1]
			old.ResourceVersion, slice.ResourceVersion = "1", "1"
			field.ErrorMatcher{}.ByType().ByField().Test(t, want, ValidateResourceSliceUpdate(slice, old))
		})
	}
	t.Run("individual-cost-limit", func(t *testing.T) {
		featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
			features.DRAConsumableCapacity: true, features.DRASharingAffinity: true,
		})
		slice := testResourceSliceWithSharingAffinity("slice", "pool", "driver.example.com", 1)
		slice.Spec.SharingAffinity[0].Expression = fmt.Sprintf("cel.bind(s, %q, (s+s).matches(s+s) ? 'yes' : 'no')", strings.Repeat("a", 4000))
		compiled := sharingaffinity.ValidateExpression(slice.Spec.SharingAffinity[0].Expression, environment.NewExpressions)
		if compiled.Error != nil || compiled.MaxCost <= uint64(resourceapi.SharingAffinityCELMaxCost) {
			t.Fatalf("expected a compilable over-budget fixture, got cost %d, error %v", compiled.MaxCost, compiled.Error)
		}
		want := field.ErrorList{
			field.Forbidden(field.NewPath("spec", "sharingAffinity").Index(0).Child("expression"), ""),
			field.Forbidden(field.NewPath("spec", "sharingAffinity"), ""),
		}
		field.ErrorMatcher{}.ByType().ByField().Test(t, want, ValidateResourceSlice(slice))
		old := slice.DeepCopy()
		old.Spec.SharingAffinity[0].Expression = `""`
		old.ResourceVersion, slice.ResourceVersion = "1", "1"
		field.ErrorMatcher{}.ByType().ByField().Test(t, want, ValidateResourceSliceUpdate(slice, old))
	})
}

func TestValidateResourceSliceUpdateWithSharingAffinity(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.DRAConsumableCapacity: true, features.DRASharingAffinity: true,
	})

	name := "valid"
	p := field.NewPath("spec", "sharingAffinity")
	validResourceSlice := testResourceSliceWithSharingAffinity(name, name, name, 2)
	// Over-limit stored values model limits tightened after persistence.
	invalidResourceSliceWithLongExpression := validResourceSlice.DeepCopy()
	invalidResourceSliceWithLongExpression.Spec.SharingAffinity[0].Expression = `"` + strings.Repeat("a", resourceapi.SharingAffinityCELExpressionMaxLength+1) + `"`

	expressions := sharingAffinityBudgetExpressions(t, uint64(resourceapi.SharingAffinityCELMaxCost))
	validResourceSliceAtAggregateCostLimit := testResourceSliceWithSharingAffinity(name, name, name, len(expressions))
	for i, expression := range expressions {
		validResourceSliceAtAggregateCostLimit.Spec.SharingAffinity[i].APIGroup = "a.example.com"
		validResourceSliceAtAggregateCostLimit.Spec.SharingAffinity[i].Expression = expression
	}
	if errs := ValidateResourceSlice(validResourceSliceAtAggregateCostLimit); len(errs) != 0 {
		t.Fatalf("expected a valid slice at the aggregate cost limit, got %v", errs)
	}

	overBudgetExpressions := sharingAffinityBudgetExpressions(t, uint64(resourceapi.SharingAffinityCELMaxCost)+1)
	invalidResourceSliceWithOverBudgetExpressions := validResourceSliceAtAggregateCostLimit.DeepCopy()
	for i, expression := range overBudgetExpressions {
		invalidResourceSliceWithOverBudgetExpressions.Spec.SharingAffinity[i].Expression = expression
	}

	scenarios := map[string]struct {
		oldResourceSlice *resourceapi.ResourceSlice
		update           func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice
		wantFailures     field.ErrorList
	}{
		"valid-update-reorders-sharing-affinity": {
			oldResourceSlice: validResourceSlice,
			update: func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				slices.Reverse(slice.Spec.SharingAffinity)
				return slice
			},
		},

		"invalid-update-clears-sharing-affinity-name": {
			wantFailures: field.ErrorList{
				field.Required(p.Index(0).Child("name"), "").MarkCoveredByDeclarative(),
			},
			oldResourceSlice: validResourceSlice,
			update: func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				slice.Spec.SharingAffinity[0].Name = ""
				return slice
			},
		},
		"invalid-update-clears-sharing-affinity-kind": {
			wantFailures: field.ErrorList{
				field.Required(p.Index(0).Child("kind"), "").MarkCoveredByDeclarative(),
			},
			oldResourceSlice: validResourceSlice,
			update: func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				slice.Spec.SharingAffinity[0].Kind = ""
				return slice
			},
		},
		"invalid-update-clears-sharing-affinity-expression": {
			wantFailures: field.ErrorList{
				field.Required(p.Index(0).Child("expression"), "").MarkCoveredByDeclarative(),
			},
			oldResourceSlice: validResourceSlice,
			update: func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				slice.Spec.SharingAffinity[0].Expression = ""
				return slice
			},
		},
		"invalid-update-clears-sharing-affinity-apiGroup": {
			wantFailures: field.ErrorList{
				field.Required(p.Index(0).Child("apiGroup"), "").MarkCoveredByDeclarative(),
			},
			oldResourceSlice: validResourceSlice,
			update: func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				slice.Spec.SharingAffinity[0].APIGroup = ""
				return slice
			},
		},

		"invalid-update-renames-sharing-affinity-to-invalid-name": {
			wantFailures: field.ErrorList{
				field.Invalid(p.Index(0).Child("name"), "invalid.name", "must not contain dots").WithOrigin("format=k8s-short-name").MarkCoveredByDeclarative(),
			},
			oldResourceSlice: validResourceSlice,
			update: func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				slice.Spec.SharingAffinity[0].Name = "invalid.name"
				return slice
			},
		},
		"invalid-update-renames-sharing-affinity-to-duplicate-name": {
			wantFailures: field.ErrorList{
				field.Duplicate(p.Index(1), resourceapi.SharingAffinityExtractor{Name: "key-0", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: `parameters.?subnetID.orValue("")`}).MarkCoveredByDeclarative(),
			},
			oldResourceSlice: validResourceSlice,
			update: func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				slice.Spec.SharingAffinity[1].Name = slice.Spec.SharingAffinity[0].Name
				return slice
			},
		},
		"invalid-update-sets-sharing-affinity-expression-to-bool": {
			wantFailures: field.ErrorList{
				field.Invalid(p.Index(0).Child("expression"), "true", "CEL expression must evaluate to string, not bool"),
			},
			oldResourceSlice: validResourceSlice,
			update: func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				slice.Spec.SharingAffinity[0].Expression = "true"
				return slice
			},
		},
		"invalid-update-introduces-sharing-affinity-syntax-error": {
			wantFailures: field.ErrorList{
				field.Invalid(p.Index(0).Child("expression"), "parameters.", "compile CEL expression: ERROR: <input>:1:12: Syntax error: no viable alternative at input '.'\n | parameters.\n | ...........^"),
			},
			oldResourceSlice: validResourceSlice,
			update: func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				slice.Spec.SharingAffinity[0].Expression = "parameters."
				return slice
			},
		},

		"invalid-update-grows-sharing-affinity-over-limit": {
			wantFailures: field.ErrorList{
				field.TooMany(p, resourceapi.SharingAffinityMaxEntries+1, resourceapi.SharingAffinityMaxEntries).WithOrigin("maxItems").MarkCoveredByDeclarative(),
			},
			oldResourceSlice: validResourceSlice,
			update: func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				slice.Spec.SharingAffinity = testResourceSliceWithSharingAffinity(name, name, name, resourceapi.SharingAffinityMaxEntries+1).Spec.SharingAffinity
				return slice
			},
		},
		"invalid-update-grows-sharing-affinity-kind-over-limit": {
			wantFailures: field.ErrorList{
				field.TooLongCharacters(p.Index(0).Child("kind"), "", 63).WithOrigin("maxLength").MarkCoveredByDeclarative(),
			},
			oldResourceSlice: validResourceSlice,
			update: func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				slice.Spec.SharingAffinity[0].Kind = strings.Repeat("A", 64)
				return slice
			},
		},
		"invalid-update-grows-sharing-affinity-expression-over-limit": {
			wantFailures: field.ErrorList{
				field.TooLong(p.Index(0).Child("expression"), "", resourceapi.SharingAffinityCELExpressionMaxLength).WithOrigin("maxBytes").MarkCoveredByDeclarative(),
			},
			oldResourceSlice: validResourceSlice,
			update: func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				slice.Spec.SharingAffinity[0].Expression = invalidResourceSliceWithLongExpression.Spec.SharingAffinity[0].Expression
				return slice
			},
		},
		"invalid-update-grows-sharing-affinity-apiGroup-over-limit": {
			wantFailures: field.ErrorList{
				field.TooLongCharacters(p.Index(0).Child("apiGroup"), "", 253).WithOrigin("maxLength").MarkCoveredByDeclarative(),
			},
			oldResourceSlice: validResourceSlice,
			update: func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				slice.Spec.SharingAffinity[0].APIGroup = strings.Repeat("b", 254)
				return slice
			},
		},

		"valid-ratcheting-unchanged-oversized-sharing-affinity": {
			oldResourceSlice: testResourceSliceWithSharingAffinity(name, name, name, resourceapi.SharingAffinityMaxEntries+1),
			update: func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				slice.Labels = map[string]string{"foo": "bar"}
				return slice
			},
		},
		"valid-ratcheting-unchanged-sharing-affinity-overlength-expression-sibling-edit": {
			oldResourceSlice: invalidResourceSliceWithLongExpression,
			update: func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				slice.Spec.SharingAffinity[1].Expression = `"changed"`
				return slice
			},
		},
		"invalid-update-revalidates-sharing-affinity-expression-length-after-rename": {
			wantFailures: field.ErrorList{
				field.TooLong(p.Index(0).Child("expression"), "", resourceapi.SharingAffinityCELExpressionMaxLength).WithOrigin("maxBytes").MarkCoveredByDeclarative(),
			},
			oldResourceSlice: invalidResourceSliceWithLongExpression,
			update: func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				slice.Spec.SharingAffinity[0].Name = "renamed"
				return slice
			},
		},

		"valid-ratcheting-unchanged-sharing-affinity-over-budget-aggregate-cost": {
			oldResourceSlice: invalidResourceSliceWithOverBudgetExpressions,
			update: func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				slice.Spec.Pool.Generation++
				return slice
			},
		},
		"invalid-update-revalidates-sharing-affinity-aggregate-cost-after-apiGroup-change": {
			wantFailures: field.ErrorList{
				field.Forbidden(p, "too complex, total cost of sharing affinity CEL expressions in the slice exceeds cost limit"),
			},
			oldResourceSlice: invalidResourceSliceWithOverBudgetExpressions,
			update: func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				slice.Spec.SharingAffinity[0].APIGroup = "*"
				return slice
			},
		},
		"invalid-update-changes-sharing-affinity-expression-to-exceed-aggregate-cost-limit": {
			wantFailures: field.ErrorList{
				field.Forbidden(p, "too complex, total cost of sharing affinity CEL expressions in the slice exceeds cost limit"),
			},
			oldResourceSlice: validResourceSliceAtAggregateCostLimit,
			update: func(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
				last := len(slice.Spec.SharingAffinity) - 1
				slice.Spec.SharingAffinity[last].Expression = overBudgetExpressions[last]
				return slice
			},
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			oldResourceSlice := scenario.oldResourceSlice.DeepCopy()
			oldResourceSlice.ResourceVersion = "1"
			errs := ValidateResourceSliceUpdate(scenario.update(oldResourceSlice.DeepCopy()), oldResourceSlice)
			if assertFailures(t, scenario.wantFailures, errs) {
				// assertFailures deliberately ignores error origins.
				field.ErrorMatcher{}.ByOrigin().Test(t, scenario.wantFailures, errs)
			}
		})
	}
}
