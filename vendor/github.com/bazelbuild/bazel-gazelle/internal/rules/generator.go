/* Copyright 2016 The Bazel Authors. All rights reserved.

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

package rules

import (
	"fmt"
	"log"
	"path"
	"strings"

	"github.com/bazelbuild/bazel-gazelle/internal/config"
	"github.com/bazelbuild/bazel-gazelle/internal/label"
	"github.com/bazelbuild/bazel-gazelle/internal/packages"
	bf "github.com/bazelbuild/buildtools/build"
)

// NewGenerator returns a new instance of Generator.
// "oldFile" is the existing build file. May be nil.
func NewGenerator(c *config.Config, l *label.Labeler, oldFile *bf.File) *Generator {
	shouldSetVisibility := oldFile == nil || !hasDefaultVisibility(oldFile)
	return &Generator{c: c, l: l, shouldSetVisibility: shouldSetVisibility}
}

// Generator generates Bazel build rules for Go build targets.
type Generator struct {
	c                   *config.Config
	l                   *label.Labeler
	shouldSetVisibility bool
}

// GenerateRules generates a list of rules for targets in "pkg". It also returns
// a list of empty rules that may be deleted from an existing file.
func (g *Generator) GenerateRules(pkg *packages.Package) (rules []bf.Expr, empty []bf.Expr, err error) {
	var rs []bf.Expr

	protoLibName, protoRules := g.generateProto(pkg)
	rs = append(rs, protoRules...)

	libName, libRule := g.generateLib(pkg, protoLibName)
	rs = append(rs, libRule)

	rs = append(rs,
		g.generateBin(pkg, libName),
		g.generateTest(pkg, libName, false),
		g.generateTest(pkg, "", true))

	for _, r := range rs {
		if isEmpty(r) {
			empty = append(empty, r)
		} else {
			rules = append(rules, r)
		}
	}

	return rules, empty, nil
}

func (g *Generator) generateProto(pkg *packages.Package) (string, []bf.Expr) {
	if g.c.ProtoMode == config.DisableProtoMode {
		// Don't create or delete proto rules in this mode. Any existing rules
		// are likely hand-written.
		return "", nil
	}

	filegroupName := config.DefaultProtosName
	protoName := g.l.ProtoLabel(pkg.Rel, pkg.Name).Name
	goProtoName := g.l.GoProtoLabel(pkg.Rel, pkg.Name).Name

	if g.c.ProtoMode == config.LegacyProtoMode {
		if !pkg.Proto.HasProto() {
			return "", []bf.Expr{EmptyRule("filegroup", filegroupName)}
		}
		attrs := []KeyValue{
			{Key: "name", Value: filegroupName},
			{Key: "srcs", Value: pkg.Proto.Sources},
		}
		if g.shouldSetVisibility {
			attrs = append(attrs, KeyValue{"visibility", []string{checkInternalVisibility(pkg.Rel, "//visibility:public")}})
		}
		return "", []bf.Expr{NewRule("filegroup", attrs)}
	}

	if !pkg.Proto.HasProto() {
		return "", []bf.Expr{
			EmptyRule("filegroup", filegroupName),
			EmptyRule("proto_library", protoName),
			EmptyRule("go_proto_library", goProtoName),
		}
	}

	var rules []bf.Expr
	visibility := []string{checkInternalVisibility(pkg.Rel, "//visibility:public")}
	protoAttrs := []KeyValue{
		{"name", protoName},
		{"srcs", pkg.Proto.Sources},
	}
	if g.shouldSetVisibility {
		protoAttrs = append(protoAttrs, KeyValue{"visibility", visibility})
	}
	imports := pkg.Proto.Imports
	if !imports.IsEmpty() {
		protoAttrs = append(protoAttrs, KeyValue{config.GazelleImportsKey, imports})
	}
	rules = append(rules, NewRule("proto_library", protoAttrs))

	goProtoAttrs := []KeyValue{
		{"name", goProtoName},
		{"proto", ":" + protoName},
		{"importpath", pkg.ImportPath},
	}
	if pkg.Proto.HasServices {
		goProtoAttrs = append(goProtoAttrs, KeyValue{"compilers", []string{"@io_bazel_rules_go//proto:go_grpc"}})
	}
	if g.shouldSetVisibility {
		goProtoAttrs = append(goProtoAttrs, KeyValue{"visibility", visibility})
	}
	if !imports.IsEmpty() {
		goProtoAttrs = append(goProtoAttrs, KeyValue{config.GazelleImportsKey, imports})
	}
	rules = append(rules, NewRule("go_proto_library", goProtoAttrs))

	return goProtoName, rules
}

func (g *Generator) generateBin(pkg *packages.Package, library string) bf.Expr {
	name := g.l.BinaryLabel(pkg.Rel).Name
	if !pkg.IsCommand() || pkg.Binary.Sources.IsEmpty() && library == "" {
		return EmptyRule("go_binary", name)
	}
	visibility := checkInternalVisibility(pkg.Rel, "//visibility:public")
	attrs := g.commonAttrs(pkg.Rel, name, visibility, pkg.Binary)
	if library != "" {
		attrs = append(attrs, KeyValue{"embed", []string{":" + library}})
	}
	return NewRule("go_binary", attrs)
}

func (g *Generator) generateLib(pkg *packages.Package, goProtoName string) (string, *bf.CallExpr) {
	name := g.l.LibraryLabel(pkg.Rel).Name
	if !pkg.Library.HasGo() && goProtoName == "" {
		return "", EmptyRule("go_library", name)
	}
	var visibility string
	if pkg.IsCommand() {
		// Libraries made for a go_binary should not be exposed to the public.
		visibility = "//visibility:private"
	} else {
		visibility = checkInternalVisibility(pkg.Rel, "//visibility:public")
	}

	attrs := g.commonAttrs(pkg.Rel, name, visibility, pkg.Library)
	attrs = append(attrs, KeyValue{"importpath", pkg.ImportPath})
	if goProtoName != "" {
		attrs = append(attrs, KeyValue{"embed", []string{":" + goProtoName}})
	}

	rule := NewRule("go_library", attrs)
	return name, rule
}

// hasDefaultVisibility returns whether oldFile contains a "package" rule with
// a "default_visibility" attribute. Rules generated by Gazelle should not
// have their own visibility attributes if this is the case.
func hasDefaultVisibility(oldFile *bf.File) bool {
	for _, s := range oldFile.Stmt {
		c, ok := s.(*bf.CallExpr)
		if !ok {
			continue
		}
		r := bf.Rule{Call: c}
		if r.Kind() == "package" && r.Attr("default_visibility") != nil {
			return true
		}
	}
	return false
}

// checkInternalVisibility overrides the given visibility if the package is
// internal.
func checkInternalVisibility(rel, visibility string) string {
	if i := strings.LastIndex(rel, "/internal/"); i >= 0 {
		visibility = fmt.Sprintf("//%s:__subpackages__", rel[:i])
	} else if strings.HasPrefix(rel, "internal/") {
		visibility = "//:__subpackages__"
	}
	return visibility
}

func (g *Generator) generateTest(pkg *packages.Package, library string, isXTest bool) bf.Expr {
	name := g.l.TestLabel(pkg.Rel, isXTest).Name
	target := pkg.Test
	if isXTest {
		target = pkg.XTest
	}
	if !target.HasGo() {
		return EmptyRule("go_test", name)
	}
	attrs := g.commonAttrs(pkg.Rel, name, "", target)
	if library != "" {
		attrs = append(attrs, KeyValue{"embed", []string{":" + library}})
	}
	if pkg.HasTestdata {
		glob := GlobValue{Patterns: []string{"testdata/**"}}
		attrs = append(attrs, KeyValue{"data", glob})
	}
	return NewRule("go_test", attrs)
}

func (g *Generator) commonAttrs(pkgRel, name, visibility string, target packages.GoTarget) []KeyValue {
	attrs := []KeyValue{{"name", name}}
	if !target.Sources.IsEmpty() {
		attrs = append(attrs, KeyValue{"srcs", target.Sources})
	}
	if target.Cgo {
		attrs = append(attrs, KeyValue{"cgo", true})
	}
	if !target.CLinkOpts.IsEmpty() {
		attrs = append(attrs, KeyValue{"clinkopts", g.options(target.CLinkOpts, pkgRel)})
	}
	if !target.COpts.IsEmpty() {
		attrs = append(attrs, KeyValue{"copts", g.options(target.COpts, pkgRel)})
	}
	if g.shouldSetVisibility && visibility != "" {
		attrs = append(attrs, KeyValue{"visibility", []string{visibility}})
	}
	imports := target.Imports
	if !imports.IsEmpty() {
		attrs = append(attrs, KeyValue{config.GazelleImportsKey, imports})
	}
	return attrs
}

var (
	// shortOptPrefixes are strings that come at the beginning of an option
	// argument that includes a path, e.g., -Ifoo/bar.
	shortOptPrefixes = []string{"-I", "-L", "-F"}

	// longOptPrefixes are separate arguments that come before a path argument,
	// e.g., -iquote foo/bar.
	longOptPrefixes = []string{"-I", "-L", "-F", "-iquote", "-isystem"}
)

// options transforms package-relative paths in cgo options into repository-
// root-relative paths that Bazel can understand. For example, if a cgo file
// in //foo declares an include flag in its copts: "-Ibar", this method
// will transform that flag into "-Ifoo/bar".
func (g *Generator) options(opts packages.PlatformStrings, pkgRel string) packages.PlatformStrings {
	fixPath := func(opt string) string {
		if strings.HasPrefix(opt, "/") {
			return opt
		}
		return path.Clean(path.Join(pkgRel, opt))
	}

	fixGroups := func(groups []string) ([]string, error) {
		fixedGroups := make([]string, len(groups))
		for i, group := range groups {
			opts := strings.Split(group, packages.OptSeparator)
			fixedOpts := make([]string, len(opts))
			isPath := false
			for j, opt := range opts {
				if isPath {
					opt = fixPath(opt)
					isPath = false
					goto next
				}

				for _, short := range shortOptPrefixes {
					if strings.HasPrefix(opt, short) && len(opt) > len(short) {
						opt = short + fixPath(opt[len(short):])
						goto next
					}
				}

				for _, long := range longOptPrefixes {
					if opt == long {
						isPath = true
						goto next
					}
				}

			next:
				fixedOpts[j] = escapeOption(opt)
			}
			fixedGroups[i] = strings.Join(fixedOpts, " ")
		}

		return fixedGroups, nil
	}

	opts, errs := opts.MapSlice(fixGroups)
	if errs != nil {
		log.Panicf("unexpected error when transforming options with pkg %q: %v", pkgRel, errs)
	}
	return opts
}

func escapeOption(opt string) string {
	return strings.NewReplacer(
		`\`, `\\`,
		`'`, `\'`,
		`"`, `\"`,
		` `, `\ `,
		"\t", "\\\t",
		"\n", "\\\n",
		"\r", "\\\r",
	).Replace(opt)
}

func isEmpty(r bf.Expr) bool {
	c, ok := r.(*bf.CallExpr)
	return ok && len(c.List) == 1 // name
}
