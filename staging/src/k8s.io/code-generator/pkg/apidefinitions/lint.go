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

package apidefinitions

import (
	"errors"
	"fmt"
	"path"
	"reflect"
	"regexp"
	"slices"
	"sort"
	"strings"

	"github.com/spf13/pflag"

	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/types"
)

const (
	// LintRuleKnownTagsOnly rejects any +k8s: tag in the package
	// that are not recognized as a known Kubernetes tag.
	LintRuleKnownTagsOnly = "known-tags-only"

	// LintRuleExplicitDisablement requires that all generators
	// either be enabled API definition packages where the
	// generator is typically needed, or that a tag is used
	// to disable the generator explicitly.
	// When set, this prevents generators from being accidentally
	// omitted in API definition packages where they are typically
	// needed.
	LintRuleExplicitDisablement = "require-explicit-disablement"
)

// Option provides configuration options for package identification.
type Option func(*identifyState)

type identifyState struct {
	lintRules []string
}

// WithLintRules enables one or more tag-checking lint rules.
func WithLintRules(names ...string) Option {
	return func(s *identifyState) {
		if s.lintRules == nil {
			s.lintRules = append(s.lintRules, names...)
		}
	}
}

type LintArgs struct {
	// LintRules names tag-checking lint rules to enable; see
	// apipkg.LintRule* constants for the supported names. Off by
	// default to preserve backward compatibility with third-party generator tags.
	LintRules []string
}

var allLintRules = []string{LintRuleKnownTagsOnly, LintRuleExplicitDisablement}

func AddFlags(args *LintArgs, fs *pflag.FlagSet) {
	supportedLinters := allLintRules
	fs.StringSliceVar(&args.LintRules, "lint-rules", args.LintRules,
		fmt.Sprintf("Comma-separated list of tag-checking lint rules to enable. Supports: %s",
			strings.Join(supportedLinters, ",")))
}

func ValidateFlags(lintRules []string) error {
	for _, rule := range lintRules {
		if !slices.Contains(allLintRules, rule) {
			return fmt.Errorf("unrecognized rule in --lint-rules: %s", rule)
		}
	}
	return nil
}

func applyLintRules(st *identifyState, pkg *types.Package) error {
	for _, rule := range st.lintRules {
		switch rule {
		case LintRuleKnownTagsOnly:
			if err := checkKnownTags(pkg); err != nil {
				return err
			}
		case LintRuleExplicitDisablement:
			if err := checkExplicitDisablement(pkg); err != nil {
				return err
			}
		default:
			return fmt.Errorf("unrecognized lint-rule: %s", rule)
		}
	}
	return nil
}

// checkKnownTags enforces LintRuleKnownTagsOnly. It only checks tags
// in the +k8s:*-gen[...] family; non-generator +k8s: tags pass
// through untouched.
func checkKnownTags(pkg *types.Package) error {
	if pkg == nil {
		return nil
	}
	var unknown []string
	for name := range codetags.Extract("+", pkg.Comments) {
		if !strings.HasPrefix(name, "k8s:") || !strings.Contains(name, "-gen") {
			continue
		}
		base := name
		if i := strings.Index(name[len("k8s:"):], ":"); i >= 0 {
			base = name[:len("k8s:")+i]
		}
		if !allKnownTags.Has(base) {
			unknown = append(unknown, "+"+name)
		}
	}
	if len(unknown) == 0 {
		return nil
	}

	sort.Strings(unknown)
	allowed := allKnownTags.UnsortedList()
	slices.Sort(allowed)
	return fmt.Errorf("%s/doc.go: unrecognized +k8s: tag(s): %s\nrecognized generator tags: %s",
		pkg.Dir, strings.Join(unknown, ", "), strings.Join(allowed, ", "))
}

// checkExplicitDisablement enforces LintRuleExplicitDisablement.
func checkExplicitDisablement(pkg *types.Package) error {
	if pkg == nil {
		return nil
	}
	roles := classifyPackage(pkg)
	expected := expectedGenerators(roles)
	if len(expected) == 0 {
		return nil
	}
	tagged := codetags.Extract("+", pkg.Comments)
	var missing []Spec
	for _, spec := range expected {
		if _, ok := tagged[spec.ActivationTag]; !ok {
			missing = append(missing, spec)
		}
	}
	if len(missing) == 0 {
		return nil
	}
	sort.Slice(missing, func(i, j int) bool { return missing[i].ActivationTag < missing[j].ActivationTag })

	var sb strings.Builder
	fmt.Fprintf(&sb, "%s/doc.go: %s.\n", pkg.Dir, roles.describePackage(pkg.Dir))
	sb.WriteString("Set explicit configuration or disablement for these generators in doc.go:")
	for _, spec := range missing {
		fmt.Fprintf(&sb, "\n  +%s=<%s>", spec.ActivationTag, valueHint(spec))
	}
	return errors.New(sb.String())
}

// valueHint describes the accepted shape of an ActivationTag value, for
// inclusion in user-facing errors.
func valueHint(spec Spec) string {
	switch spec.ValueMode {
	case ConversionPeerList:
		return "peer-package or false"
	case TypeFilterList:
		return "TypeMeta or false"
	case Package:
		return "package or false"
	case Boolean:
		return "true or false"
	default:
		return "value or false"
	}
}

func expectedGenerators(roles packageRoles) []Spec {
	var expected []Spec
	if roles.isExternalVersion {
		expected = append(expected, Deepcopy)
	}
	if roles.isInternalVersion {
		expected = append(expected, Conversion, Defaulter, Validation)
	}
	return expected
}

type packageRoles struct {
	// isExternalGroup identifies an API group package containing
	// versioned API types (e.g. staging/src/k8s.io/api/apps).
	isExternalGroup bool

	// isInternalGroup identifies an API group package containing
	// API type internals (e.g. pkg/apis/apps). The "internal type" (aka Hub type)
	// often lives here.
	isInternalGroup bool

	// isExternalVersion identifies a package containing versioned API types
	// (e.g. staging/src/k8s.io/api/apps/v1).
	isExternalVersion bool

	// isInternalVersion identifies a per-version internals package containing
	// conversion / defaulting / validation for hub types in the parent
	// directory (e.g. pkg/apis/apps/v1).
	isInternalVersion bool
}

func (r packageRoles) describePackage(pkgDir string) string {
	var parts []string
	if r.isExternalGroup {
		parts = append(parts, "API group package with versioned API types in subpackages")
	}
	if r.isInternalGroup {
		parts = append(parts, "API group package containing hub API types")
	}
	if r.isExternalVersion {
		parts = append(parts, "package containing versioned API types")
	}
	if r.isInternalVersion {
		desc := "package containing per-version internals (conversion, defaulting, validation)"
		if parent := path.Dir(pkgDir); parent != "" && parent != "." && parent != "/" {
			desc += " for hub types at " + parent
		}
		parts = append(parts, desc)
	}
	if len(parts) == 0 {
		return "package has no recognized API role"
	}
	return "this package looks like: " + strings.Join(parts, "; ")
}

// classifyPackage looks for evidence of package roles.
//
// This may return false negatives as there is insufficient
// information to identify the intent of all packages.
// For example, a package directory with no tags or types might be in
// a directory that is expected to contain API types, but which
// will not be detected by this function.
//
// This must ONLY be used for linting.
func classifyPackage(pkg *types.Package) packageRoles {
	// TODO: We may introduce API declaration files in package in the future.
	// This would enable package classification that is more precise and
	// does not require the heuristics used here.
	if pkg == nil || !looksLikeAPIPackage(pkg) {
		return packageRoles{}
	}

	roles := packageRoles{}
	isVersion := isVersionPath(pkg.Path)
	external := looksLikeExternalVersionPackage(pkg)
	if isVersion {
		if external {
			roles.isExternalVersion = true
		} else if looksLikeInternalVersionPackage(pkg) {
			roles.isInternalVersion = true
		}
	} else {
		if external {
			roles.isExternalGroup = true
		} else {
			roles.isInternalGroup = true
		}
	}
	return roles
}

func looksLikeAPIPackage(pkg *types.Package) bool {
	if _, ok, _ := GroupNameForPackage(pkg.Comments); ok {
		return true
	}
	for name := range codetags.Extract("+", pkg.Comments) {
		if allKnownTags.Has(name) {
			return true
		}
	}
	return false
}

func looksLikeInternalVersionPackage(pkg *types.Package) bool {
	raw := codetags.Extract("+", pkg.Comments)
	for _, s := range []Spec{Defaulter, Validation} {
		if _, ok := raw[s.ActivationTag]; ok {
			return true
		}
		if s.InputTag != "" {
			if _, ok := raw[s.InputTag]; ok {
				return true
			}
		}
	}
	return false
}

func looksLikeExternalVersionPackage(pkg *types.Package) bool {
	for _, t := range pkg.Types {
		for _, m := range t.Members {
			if m.Name == "TypeMeta" && m.Embedded {
				tag, exists := reflect.StructTag(m.Tags).Lookup("json")
				name := strings.Split(tag, ",")[0]
				if exists && name == "" {
					return true
				}
			}
		}
	}
	return false
}

var versionPathRE = regexp.MustCompile(`^v\d+([a-z]+\d+)?$`)

// isVersionPath reports whether the trailing path segment looks like
// an API version (v1, v1beta2, v2alpha1, ...).
func isVersionPath(p string) bool {
	if p == "" {
		return false
	}
	return versionPathRE.MatchString(path.Base(p))
}
