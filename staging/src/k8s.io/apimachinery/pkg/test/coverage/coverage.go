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

package coverage

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"sync"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// Rule is one declared field-validation error: error type plus optional origin.
type Rule struct {
	ErrorType string
	Origin    string
}

// FieldRules maps field path → declared rules for a single Kind.
type FieldRules map[string][]Rule

// ruleKey is the flat identity of one declared-or-observed rule across the
// whole test process.
type ruleKey struct {
	gvk       schema.GroupVersionKind
	path      string
	errorType string
	origin    string
}

// indexKeyRe normalizes runtime path subscripts ("[0]", "[my-key]") to "[*]"
// so observed paths line up with the canonical paths from validation-gen.
var indexKeyRe = regexp.MustCompile(`\[[^\]]+\]`)

var (
	rulesMu  sync.Mutex
	declared = sets.New[ruleKey]()
	observed = sets.New[ruleKey]()
)

// RegisterDeclaredRules records the rules declared for one GVK by
// validation-gen.
func RegisterDeclaredRules(gvk schema.GroupVersionKind, rules FieldRules) {
	rulesMu.Lock()
	defer rulesMu.Unlock()
	for path, rs := range rules {
		for _, r := range rs {
			declared.Insert(ruleKey{gvk: gvk, path: path, errorType: r.ErrorType, origin: r.Origin})
		}
	}
}

// RecordObservedRules marks every error in errs as observed for the given GVK.
// Idempotent per (GVK, path, errorType, origin). Callers compute the GVK from
// whatever context they have (e.g., a runtime scheme + request info).
func RecordObservedRules(gvk schema.GroupVersionKind, errs field.ErrorList) {
	rulesMu.Lock()
	defer rulesMu.Unlock()
	// Runtime errors carry the actual subscript ("spec.items[3]",
	// "metadata.labels[my-key]"); declared paths use "[*]". Replace every
	// "[...]" with "[*]" so the lookup in AssertDeclarativeCoverage matches.
	for _, e := range errs {
		path := indexKeyRe.ReplaceAllString(e.Field, "[*]")
		observed.Insert(ruleKey{gvk: gvk, path: path, errorType: string(e.Type), origin: e.Origin})
	}
}

// AssertDeclarativeCoverage returns nil if every rule registered via
// RegisterDeclaredRules was observed at least once during this test process;
// otherwise an error listing the uncovered rules.
func AssertDeclarativeCoverage() error {
	rulesMu.Lock()
	uncovered := declared.Difference(observed).UnsortedList()
	rulesMu.Unlock()
	if len(uncovered) == 0 {
		return nil
	}
	return fmt.Errorf("%d uncovered declarative-validation rules:\n%s", len(uncovered), formatUncovered(uncovered))
}

// formatUncovered renders uncovered rules grouped by GVK, one rule per line.
// GVKs are sorted by group/version/kind; rules within a GVK by path/errorType/origin.
//
// Example:
//
//	example.com/v1, Kind=Widget:
//	  spec.name  FieldValueRequired
//	  spec.name  FieldValueInvalid origin="format=dns-label"
func formatUncovered(uncovered []ruleKey) string {
	gvksToRules := map[schema.GroupVersionKind][]ruleKey{}
	for _, k := range uncovered {
		gvksToRules[k.gvk] = append(gvksToRules[k.gvk], k)
	}
	gvks := make([]schema.GroupVersionKind, 0, len(gvksToRules))
	for gvk := range gvksToRules {
		gvks = append(gvks, gvk)
	}
	sort.Slice(gvks, func(i, j int) bool {
		if gvks[i].Group != gvks[j].Group {
			return gvks[i].Group < gvks[j].Group
		}
		if gvks[i].Version != gvks[j].Version {
			return gvks[i].Version < gvks[j].Version
		}
		return gvks[i].Kind < gvks[j].Kind
	})
	var sb strings.Builder
	for _, gvk := range gvks {
		gv := gvk.Group + "/" + gvk.Version
		if gvk.Group == "" {
			gv = gvk.Version
		}
		fmt.Fprintf(&sb, "%s, Kind=%s:\n", gv, gvk.Kind)
		keys := gvksToRules[gvk]
		sort.Slice(keys, func(i, j int) bool {
			if keys[i].path != keys[j].path {
				return keys[i].path < keys[j].path
			}
			if keys[i].errorType != keys[j].errorType {
				return keys[i].errorType < keys[j].errorType
			}
			return keys[i].origin < keys[j].origin
		})
		for _, k := range keys {
			if k.origin != "" {
				fmt.Fprintf(&sb, "  %s  %s origin=%q\n", k.path, k.errorType, k.origin)
			} else {
				fmt.Fprintf(&sb, "  %s  %s\n", k.path, k.errorType)
			}
		}
	}
	return sb.String()
}
