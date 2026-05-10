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

package main

import (
	"cmp"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/code-generator/cmd/validation-gen/validators"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/namer"
	"k8s.io/gengo/v2/types"
	"sigs.k8s.io/yaml"
)

// Package symbols referenced in emitted test fixtures. fmtPkgSymbols (shared
// with validation.go) covers Fprintln.
var (
	schemaPkg               = "k8s.io/apimachinery/pkg/runtime/schema"
	schemaPkgSymbols        = mkPkgNames(schemaPkg, "GroupVersionKind")
	osPkgSymbols            = mkPkgNames("os", "Stderr", "Exit")
	testingSymbols          = mkPkgNames("testing", "M")
	runtimeTestPkg          = "k8s.io/apimachinery/pkg/test/coverage"
	runtimeRegisterSymbols  = mkPkgNames(runtimeTestPkg, "RegisterDeclaredRules", "FieldRules")
	runtimeAssertCovSymbols = mkPkgNames(runtimeTestPkg, "AssertDeclarativeCoverage")
)

// apiVersionRe matches Kubernetes-style API versions: v1, v1alpha1, v2beta3.
// Used to detect whether an input package is a versioned API package (vs. a
// helper or main package, which we skip).
var apiVersionRe = regexp.MustCompile(`^v\d+(alpha\d+|beta\d+)?$`)

// rule is one declared field-validation error.
type rule struct {
	ErrorType string
	Origin    string
}

// fieldRules maps field path → declared rules for a single Kind.
type fieldRules map[string][]rule

// report is one GVK's declared rules. Used as in-memory scaffolding while
// emitting per-Kind test fixtures; not serialized.
type report struct {
	Group   string
	Version string
	Kind    string
	Rules   fieldRules
}

// allowlistEntry filters a declared rule out of coverage fixture generation.
// Every field is required: apiVersion matches by literal equality (in the
// "<group>/<version>" form, or just "<version>" for core); kind/path/
// errorType/origin match literally or wildcard with "*". Empty is rejected
// at load time so an entry can never silently overexclude.
type allowlistEntry struct {
	APIVersion string `json:"apiVersion"`
	Kind       string `json:"kind"`
	Path       string `json:"path"`
	ErrorType  string `json:"errorType"`
	Origin     string `json:"origin"`
	Reason     string `json:"reason"`
}

// groupVersion splits e.APIVersion into (group, version). "v1" → ("", "v1");
// "apps/v1" → ("apps", "v1").
func (e *allowlistEntry) groupVersion() (string, string) {
	if i := strings.IndexByte(e.APIVersion, '/'); i >= 0 {
		return e.APIVersion[:i], e.APIVersion[i+1:]
	}
	return "", e.APIVersion
}

// matches reports whether the entry filters out the given rule. apiVersion
// must match exactly; the remaining fields match by literal equality or
// wildcard with "*".
func (e *allowlistEntry) matches(group, version, kind, path string, rule rule) bool {
	g, v := e.groupVersion()
	return g == group &&
		v == version &&
		(e.Kind == "*" || e.Kind == kind) &&
		(e.Path == "*" || e.Path == path) &&
		(e.ErrorType == "*" || e.ErrorType == rule.ErrorType) &&
		(e.Origin == "*" || e.Origin == rule.Origin)
}

// filterReports returns reports with allowlisted rules removed. Reports
// left with no rules are dropped. Returns reports unchanged if allowlist is empty.
func filterReports(reports []*report, allowlist []allowlistEntry) []*report {
	if len(allowlist) == 0 {
		return reports
	}
	out := make([]*report, 0, len(reports))
	for _, r := range reports {
		filtered := fieldRules{}
		for path, rs := range r.Rules {
			for _, rule := range rs {
				if slices.ContainsFunc(allowlist, func(e allowlistEntry) bool {
					return e.matches(r.Group, r.Version, r.Kind, path, rule)
				}) {
					continue
				}
				filtered[path] = append(filtered[path], rule)
			}
		}
		if len(filtered) == 0 {
			continue
		}
		out = append(out, &report{
			Group: r.Group, Version: r.Version, Kind: r.Kind, Rules: filtered,
		})
	}
	return out
}

// loadAllowlist reads the YAML file at path. Returns nil when path is empty.
// Errors on read/parse failure or any entry missing the required reason.
func loadAllowlist(path string) ([]allowlistEntry, error) {
	if path == "" {
		return nil, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading allowlist %q: %w", path, err)
	}
	var entries []allowlistEntry
	if err := yaml.Unmarshal(data, &entries); err != nil {
		return nil, fmt.Errorf("parsing allowlist %q: %w", path, err)
	}
	for i, e := range entries {
		for _, f := range []struct{ name, val string }{
			{"apiVersion", e.APIVersion},
			{"kind", e.Kind},
			{"path", e.Path},
			{"errorType", e.ErrorType},
			{"origin", e.Origin},
			{"reason", e.Reason},
		} {
			if strings.TrimSpace(f.val) == "" {
				return nil, fmt.Errorf("allowlist %q entry %d: %s is required (use %q to wildcard)", path, i, f.name, "*")
			}
		}
	}
	return entries, nil
}

// perVersionTestGen emits one <prefix><version>_test.go for one (Group, Version, Kind):
// an init() that registers the declared FieldRules with the runtime testing
// package. Uses SnippetWriter so imports are namer-tracked.
type perVersionTestGen struct {
	generator.GoGenerator
	outputPackage string
	imports       namer.ImportTracker
	report        *report
}

func newPerVersionTestGen(outputPackage, filePrefix string, report *report) generator.Generator {
	return &perVersionTestGen{
		GoGenerator: generator.GoGenerator{
			OutputFilename: filePrefix + report.Version + "_test.go",
		},
		outputPackage: outputPackage,
		imports:       generator.NewImportTrackerForPackage(outputPackage),
		report:        report,
	}
}

func (g *perVersionTestGen) Namers(*generator.Context) namer.NameSystems {
	return namer.NameSystems{"raw": namer.NewRawNamer(g.outputPackage, g.imports)}
}

func (g *perVersionTestGen) Imports(*generator.Context) []string { return g.imports.ImportLines() }

func (g *perVersionTestGen) Filter(*generator.Context, *types.Type) bool { return false }

func (g *perVersionTestGen) Init(c *generator.Context, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "$", "$")
	args := generator.Args{
		"schema":  mkSymbolArgs(c, schemaPkgSymbols),
		"runtime": mkSymbolArgs(c, runtimeRegisterSymbols),
		"group":   strconv.Quote(g.report.Group),
		"version": strconv.Quote(g.report.Version),
		"kind":    strconv.Quote(g.report.Kind),
	}
	sw.Do("func init() {\n", nil)
	sw.Do("    $.runtime.RegisterDeclaredRules|raw$(\n", args)
	sw.Do("        $.schema.GroupVersionKind|raw${Group: $.group$, Version: $.version$, Kind: $.kind$},\n", args)
	sw.Do("        $.runtime.FieldRules|raw${\n", args)

	paths := make([]string, 0, len(g.report.Rules))
	for p := range g.report.Rules {
		paths = append(paths, p)
	}
	slices.Sort(paths)
	for _, path := range paths {
		rules := slices.Clone(g.report.Rules[path])
		slices.SortFunc(rules, func(a, b rule) int {
			if c := cmp.Compare(a.ErrorType, b.ErrorType); c != 0 {
				return c
			}
			return cmp.Compare(a.Origin, b.Origin)
		})
		sw.Do("            $.path$: {\n", generator.Args{"path": strconv.Quote(path)})
		for _, r := range rules {
			ruleArgs := generator.Args{"errorType": strconv.Quote(r.ErrorType)}
			if r.Origin != "" {
				ruleArgs["origin"] = strconv.Quote(r.Origin)
				sw.Do("                {ErrorType: $.errorType$, Origin: $.origin$},\n", ruleArgs)
			} else {
				sw.Do("                {ErrorType: $.errorType$},\n", ruleArgs)
			}
		}
		sw.Do("            },\n", nil)
	}
	sw.Do("        },\n    )\n}\n", nil)
	return sw.Error()
}

// mainTestGen emits <prefix>main_test.go for one Kind: an apiVersions slice and a
// TestMain that asserts coverage after the package's tests run.
type mainTestGen struct {
	generator.GoGenerator
	outputPackage string
	imports       namer.ImportTracker
	versions      []string
}

func newMainTestGen(outputPackage, filePrefix string, versions []string) generator.Generator {
	return &mainTestGen{
		GoGenerator: generator.GoGenerator{
			OutputFilename: filePrefix + "main_test.go",
		},
		outputPackage: outputPackage,
		imports:       generator.NewImportTrackerForPackage(outputPackage),
		versions:      versions,
	}
}

func (g *mainTestGen) Namers(*generator.Context) namer.NameSystems {
	return namer.NameSystems{"raw": namer.NewRawNamer(g.outputPackage, g.imports)}
}

func (g *mainTestGen) Imports(*generator.Context) []string { return g.imports.ImportLines() }

func (g *mainTestGen) Filter(*generator.Context, *types.Type) bool { return false }

func (g *mainTestGen) Init(c *generator.Context, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "$", "$")
	args := generator.Args{
		"testing": mkSymbolArgs(c, testingSymbols),
		"fmt":     mkSymbolArgs(c, fmtPkgSymbols),
		"os":      mkSymbolArgs(c, osPkgSymbols),
		"runtime": mkSymbolArgs(c, runtimeAssertCovSymbols),
	}
	versions := slices.Clone(g.versions)
	slices.Sort(versions)
	sw.Do("var apiVersions = []string{", nil)
	for i, v := range versions {
		if i > 0 {
			sw.Do(", ", nil)
		}
		sw.Do("$.v$", generator.Args{"v": strconv.Quote(v)})
	}
	sw.Do("}\n\n", nil)
	sw.Do("func TestMain(m *$.testing.M|raw$) {\n", args)
	sw.Do("    code := m.Run()\n", nil)
	sw.Do("    if err := $.runtime.AssertDeclarativeCoverage|raw$(); err != nil {\n", args)
	sw.Do("        $.fmt.Fprintln|raw$($.os.Stderr|raw$, err)\n", args)
	sw.Do("        if code == 0 {\n            code = 1\n        }\n    }\n", nil)
	sw.Do("    $.os.Exit|raw$(code)\n}\n", args)
	return sw.Error()
}

// collectReports appends a *report into groupKindReports for each
// Kind in pkg with at least one declared rule. Records all packages; testTargets
// filters out non-API packages (empty Version) at emit time.
func collectReports(pkg *types.Package, rootTypes []*types.Type, td *typeDiscoverer, groupKindReports map[schema.GroupKind][]*report) {
	// Derive (group, version) from the input package:
	//   group:   the GroupName const if defined (the established API
	//            convention); otherwise the package path as a fallback.
	//   version: the package name when it looks like an API version
	//            (vN[alphaM|betaM]); empty otherwise — testTargets uses the
	//            empty version to skip non-API packages.
	var group, version string
	if c, ok := pkg.Constants["GroupName"]; ok && c.ConstValue != nil {
		group = *c.ConstValue
	} else {
		group = pkg.Path
	}
	if apiVersionRe.MatchString(pkg.Name) {
		version = pkg.Name
	}

	for _, t := range rootTypes {
		rules := collectRules(td.typeNodes[t])
		if len(rules) == 0 {
			continue
		}
		kind := t.Name.Name
		gk := schema.GroupKind{Group: group, Kind: kind}
		groupKindReports[gk] = append(groupKindReports[gk], &report{
			Group:   group,
			Version: version,
			Kind:    kind,
			Rules:   rules,
		})
	}
}

// testTargets returns one SimpleTarget per Kind in groupKindReports. Each
// target's directory is <testOutputRoot>/<short-group>/<lowercase(kind)>/
// and contains one <prefix><version>_test.go per version plus a
// shared <prefix>main_test.go. Skips Kinds with empty Version
// (non-API packages) and Kinds whose every rule is allowlisted.
func testTargets(testOutputRoot, filePrefix string, groupKindReports map[schema.GroupKind][]*report, allowlist []allowlistEntry, boilerplate []byte) []generator.Target {
	if testOutputRoot == "" || len(groupKindReports) == 0 {
		return nil
	}
	out := make([]generator.Target, 0, len(groupKindReports))
	for _, reports := range groupKindReports {
		// reports is non-empty by construction in collectReports.
		first := reports[0]
		if first.Version == "" {
			continue // not a real API package
		}
		reports = filterReports(reports, allowlist)
		if len(reports) == 0 {
			continue // every rule was allowlisted away
		}
		lowerKind := strings.ToLower(first.Kind)
		// Short group name (first DNS label) as the directory; "core" for the
		// empty legacy group — matches client-gen / informer-gen convention.
		group := first.Group
		if group == "" {
			group = "core"
		}
		pkgDir := filepath.Join(testOutputRoot, strings.Split(group, ".")[0], lowerKind)

		out = append(out, &generator.SimpleTarget{
			PkgName:       lowerKind,
			PkgPath:       pkgDir, // informational; gengo writes to PkgDir
			PkgDir:        pkgDir,
			HeaderComment: boilerplate,
			GeneratorsFunc: func(*generator.Context) []generator.Generator {
				gens := make([]generator.Generator, 0, len(reports)+1) // +1 for main_test.go
				versions := make([]string, 0, len(reports))
				for _, report := range reports {
					gens = append(gens, newPerVersionTestGen(pkgDir, filePrefix, report))
					versions = append(versions, report.Version)
				}
				gens = append(gens, newMainTestGen(pkgDir, filePrefix, versions))
				return gens
			},
		})
	}
	return out
}

// collectRules walks node's type tree and returns its declared FieldRules.
func collectRules(node *typeNode) fieldRules {
	if node == nil {
		return nil
	}
	rules := fieldRules{}
	seen := map[*typeNode]bool{}

	record := func(path string, fns []validators.FunctionGen) {
		for _, fn := range fns {
			recordRules(rules, path, fn, "")
		}
	}

	var walkNode func(*typeNode, string)
	var walkChild func(*childNode, string)
	walkNode = func(n *typeNode, path string) {
		if n == nil || seen[n] {
			return
		}
		seen[n] = true
		defer delete(seen, n)

		record(path, n.typeValidations.Functions)
		if n.valueType == nil {
			return
		}
		record(joinPath(path, "[*]"), n.typeValIterations.Functions)
		record(path, n.typeKeyIterations.Functions) // keys validate at parent path

		for _, fld := range n.fields {
			walkChild(fld, joinPath(path, fld.jsonName))
		}
		if n.elem != nil {
			walkChild(n.elem, joinPath(path, "[*]"))
		}
		if n.key != nil {
			walkChild(n.key, path)
		}
		if n.underlying != nil {
			walkChild(n.underlying, path)
		}
	}
	walkChild = func(c *childNode, path string) {
		if c == nil {
			return
		}
		record(path, c.fieldValidations.Functions)
		record(joinPath(path, "[*]"), c.fieldValIterations.Functions)
		record(path, c.fieldKeyIterations.Functions)
		walkNode(c.node, path)
	}
	walkNode(node, "")
	return rules
}

// recordRules descends fg, accumulating Wrapper/MultiWrapperFunction
// PathFragments into suffix, and records (basePath+suffix, Rule) at each
// emitting leaf (Emits != nil).
func recordRules(rules fieldRules, basePath string, fg validators.FunctionGen, suffix string) {
	if fg.Emits != nil {
		path := basePath + suffix + fg.Emits.PathFragment
		rules[path] = append(rules[path], rule{
			ErrorType: string(fg.Emits.Type),
			Origin:    fg.Emits.Origin,
		})
		return
	}
	for _, arg := range fg.Args {
		switch a := arg.(type) {
		case validators.WrapperFunction:
			recordRules(rules, basePath, a.Function, suffix+a.PathFragment)
		case validators.MultiWrapperFunction:
			for _, child := range a.Functions {
				recordRules(rules, basePath, child, suffix+a.PathFragment)
			}
		}
	}
}

// joinPath joins seg onto base; empty seg is a no-op so inline-embedded
// struct fields (jsonName "") stay transparent.
func joinPath(base, seg string) string {
	if seg == "" {
		return base
	}
	if base == "" {
		return seg
	}
	if strings.HasPrefix(seg, "[") {
		return base + seg
	}
	return base + "." + seg
}
