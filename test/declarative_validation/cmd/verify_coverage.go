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

package cmd

import (
	"cmp"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	"github.com/spf13/cobra"

	"sigs.k8s.io/yaml"

	"k8s.io/code-generator/pkg/guardrails"
	"k8s.io/klog/v2"
)

var (
	expectedRulesFile string
	actualRulesDir    string
	allowlistFile     string
)

// NewVerifyCoverageCommand returns the cobra command for "verify-coverage".
func NewVerifyCoverageCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "verify-coverage",
		Short: "Diff declared declarative-validation rules against rules observed during tests",
		Long: `Compares the declared rules JSON (output of validation-gen --report-rules)
against the per-package observation files written by declarative_validation_test.go
runs (via VALIDATION_RULES_REPORT_DIR). Exits 0 if every declared rule has at
least one matching observation, 1 otherwise (with the uncovered set printed
to stderr). An optional YAML allowlist may suppress intentional gaps.`,
		Run: verifyCoverageFunc,
	}
	cmd.Flags().StringVar(&expectedRulesFile, "expected-rules", "",
		"path to the declared rules JSON (output of `validation-gen --report-rules`)")
	cmd.Flags().StringVar(&actualRulesDir, "actual-rules-dir", "",
		"directory of per-package observation JSON files written via VALIDATION_RULES_REPORT_DIR")
	cmd.Flags().StringVar(&allowlistFile, "allowlist", "",
		"optional YAML file listing rules to suppress from the uncovered output")
	_ = cmd.MarkFlagRequired("expected-rules")
	_ = cmd.MarkFlagRequired("actual-rules-dir")
	return cmd
}

func verifyCoverageFunc(_ *cobra.Command, _ []string) {
	expected, err := loadFile(expectedRulesFile)
	if err != nil {
		klog.Fatalf("Error reading --expected-rules: %v", err)
	}
	actual, err := loadDir(actualRulesDir)
	if err != nil {
		klog.Fatalf("Error reading --actual-rules-dir: %v", err)
	}
	allow, err := loadAllowlist(allowlistFile)
	if err != nil {
		klog.Fatalf("Error reading --allowlist: %v", err)
	}
	uncovered := filterAllowed(diff(expected, actual), allow)

	total := 0
	for _, r := range uncovered {
		for _, fields := range r.Kinds {
			for _, rules := range fields {
				total += len(rules)
			}
		}
	}
	if total == 0 {
		fmt.Println("All declared declarative-validation rules are covered by tests.")
		os.Exit(0)
	}

	fmt.Fprintf(os.Stderr, "Uncovered declarative-validation rules: %d\n", total)
	slices.SortFunc(uncovered, func(a, b guardrails.Report) int {
		if c := cmp.Compare(a.Group, b.Group); c != 0 {
			return c
		}
		return cmp.Compare(a.Version, b.Version)
	})
	for _, r := range uncovered {
		for _, kind := range slices.Sorted(maps.Keys(r.Kinds)) {
			fmt.Fprintf(os.Stderr, "\n%s/%s, Kind=%s:\n", r.Group, r.Version, kind)
			for _, path := range slices.Sorted(maps.Keys(r.Kinds[kind])) {
				for _, rule := range r.Kinds[kind][path] {
					suffix := rule.ErrorType
					if rule.Origin != "" {
						suffix = fmt.Sprintf("%s origin=%q", rule.ErrorType, rule.Origin)
					}
					fmt.Fprintf(os.Stderr, "  %s  %s\n", path, suffix)
				}
			}
		}
	}
	os.Exit(1)
}

// indexKeyRe normalizes runtime path subscripts ("[0]", "[my-key]") to "[*]"
// so observed paths from runtime field.Error.Field line up with the canonical
// paths emitted by validation-gen --report-rules.
var indexKeyRe = regexp.MustCompile(`\[[^\]]+\]`)

// allowEntry suppresses rules from the verifier's output. Any non-empty field
// acts as a literal-equality filter; empty fields match any value. Reason is
// required and free-form.
type allowEntry struct {
	Group     string `json:"group,omitempty"`
	Version   string `json:"version,omitempty"`
	Kind      string `json:"kind,omitempty"`
	Path      string `json:"path,omitempty"`
	ErrorType string `json:"errorType,omitempty"`
	Origin    string `json:"origin,omitempty"`
	Reason    string `json:"reason"`
}

func (a allowEntry) matches(group, version, kind, path string, r guardrails.Rule) bool {
	if a.Group != "" && a.Group != group {
		return false
	}
	if a.Version != "" && a.Version != version {
		return false
	}
	if a.Kind != "" && a.Kind != kind {
		return false
	}
	if a.Path != "" && a.Path != path {
		return false
	}
	if a.ErrorType != "" && a.ErrorType != r.ErrorType {
		return false
	}
	if a.Origin != "" && a.Origin != r.Origin {
		return false
	}
	return true
}

// loadAllowlist reads a YAML file as []allowEntry. Returns nil if path is
// empty (allowlist is optional).
func loadAllowlist(path string) ([]allowEntry, error) {
	if path == "" {
		return nil, nil
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var entries []allowEntry
	if err := yaml.Unmarshal(b, &entries); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return entries, nil
}

// loadFile reads one JSON file as []guardrails.Report.
func loadFile(path string) ([]guardrails.Report, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var r []guardrails.Report
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return r, nil
}

// loadDir reads every *.json file in dir and merges them by (Group, Version),
// union-merging Kinds maps and dedup'ing rules at each path. Used to merge
// the per-package observation files written by declarative_validation_test.go
// runs into a single set for diffing against the declared rules.
func loadDir(dir string) ([]guardrails.Report, error) {
	matches, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil {
		return nil, err
	}
	merged := map[string]*guardrails.Report{}
	for _, file := range matches {
		part, err := loadFile(file)
		if err != nil {
			return nil, err
		}
		for i := range part {
			p := &part[i]
			key := p.Group + "|" + p.Version
			r, ok := merged[key]
			if !ok {
				r = &guardrails.Report{Group: p.Group, Version: p.Version, Kinds: map[string]map[string][]guardrails.Rule{}}
				merged[key] = r
			}
			for kind, fields := range p.Kinds {
				existing, ok := r.Kinds[kind]
				if !ok {
					r.Kinds[kind] = fields
					continue
				}
				for path, rules := range fields {
					for _, rule := range rules {
						if !slices.Contains(existing[path], rule) {
							existing[path] = append(existing[path], rule)
						}
					}
				}
			}
		}
	}
	out := make([]guardrails.Report, 0, len(merged))
	for _, r := range merged {
		out = append(out, *r)
	}
	return out, nil
}

// filterReports returns a deep copy of reports keeping only rules for which
// keep returns true. Empty rule slices, paths, kinds, and reports are dropped.
func filterReports(reports []guardrails.Report, keep func(group, version, kind, path string, r guardrails.Rule) bool) []guardrails.Report {
	var out []guardrails.Report
	for _, r := range reports {
		kept := map[string]map[string][]guardrails.Rule{}
		for kind, fields := range r.Kinds {
			for path, rules := range fields {
				var ks []guardrails.Rule
				for _, rule := range rules {
					if keep(r.Group, r.Version, kind, path, rule) {
						ks = append(ks, rule)
					}
				}
				if len(ks) > 0 {
					if kept[kind] == nil {
						kept[kind] = map[string][]guardrails.Rule{}
					}
					kept[kind][path] = ks
				}
			}
		}
		if len(kept) > 0 {
			out = append(out, guardrails.Report{Group: r.Group, Version: r.Version, Kinds: kept})
		}
	}
	return out
}

// diff returns the subset of expected with no matching observation in actual,
// after normalizing actual paths via indexKeyRe. The result has the same
// shape as expected; empty kinds/paths/rules are dropped. A rule matches if
// (group, version, kind, path, errorType, origin) all coincide.
func diff(expected, actual []guardrails.Report) []guardrails.Report {
	type ruleKey struct{ gv, kind, path, errorType, origin string }
	seen := map[ruleKey]struct{}{}
	for _, r := range actual {
		gv := r.Group + "/" + r.Version
		for kind, fields := range r.Kinds {
			for path, rules := range fields {
				norm := indexKeyRe.ReplaceAllString(path, "[*]")
				for _, rule := range rules {
					seen[ruleKey{gv, kind, norm, rule.ErrorType, rule.Origin}] = struct{}{}
					// Slice-level rules (e.g. +k8s:unique=set) are declared at
					// "foo" but the runtime emits at the offending index,
					// normalized to "foo[*]". Register both so the declared
					// "foo" rule matches.
					if stripped, ok := strings.CutSuffix(norm, "[*]"); ok {
						seen[ruleKey{gv, kind, stripped, rule.ErrorType, rule.Origin}] = struct{}{}
					}
				}
			}
		}
	}
	return filterReports(expected, func(group, version, kind, path string, r guardrails.Rule) bool {
		_, ok := seen[ruleKey{group + "/" + version, kind, path, r.ErrorType, r.Origin}]
		return !ok
	})
}

// filterAllowed returns reports with rules matched by any allow entry removed.
func filterAllowed(reports []guardrails.Report, allow []allowEntry) []guardrails.Report {
	if len(allow) == 0 {
		return reports
	}
	return filterReports(reports, func(group, version, kind, path string, r guardrails.Rule) bool {
		for _, a := range allow {
			if a.matches(group, version, kind, path, r) {
				return false
			}
		}
		return true
	})
}
