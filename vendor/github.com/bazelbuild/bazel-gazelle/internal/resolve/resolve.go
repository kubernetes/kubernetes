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

package resolve

import (
	"fmt"
	"go/build"
	"log"
	"path"
	"strings"

	"github.com/bazelbuild/bazel-gazelle/internal/config"
	"github.com/bazelbuild/bazel-gazelle/internal/label"
	"github.com/bazelbuild/bazel-gazelle/internal/pathtools"
	"github.com/bazelbuild/bazel-gazelle/internal/repos"
	bf "github.com/bazelbuild/buildtools/build"
)

// Resolver resolves import strings in source files (import paths in Go,
// import statements in protos) into Bazel labels.
type Resolver struct {
	c        *config.Config
	l        *label.Labeler
	ix       *RuleIndex
	external nonlocalResolver
}

// nonlocalResolver resolves import paths outside of the current repository's
// prefix. Once we have smarter import path resolution, this shouldn't
// be necessary, and we can remove this abstraction.
type nonlocalResolver interface {
	resolve(imp string) (label.Label, error)
}

func NewResolver(c *config.Config, l *label.Labeler, ix *RuleIndex, rc *repos.RemoteCache) *Resolver {
	var e nonlocalResolver
	switch c.DepMode {
	case config.ExternalMode:
		e = newExternalResolver(l, rc)
	case config.VendorMode:
		e = newVendoredResolver(l)
	}

	return &Resolver{
		c:        c,
		l:        l,
		ix:       ix,
		external: e,
	}
}

// ResolveRule copies and modifies a generated rule e by replacing the import
// paths in the "_gazelle_imports" attribute with labels in a "deps"
// attribute. This may be safely called on expressions that aren't Go rules
// (the original expression will be returned). Any existing "deps" attribute
// is deleted, so it may be necessary to merge the result.
func (r *Resolver) ResolveRule(e bf.Expr, pkgRel string) bf.Expr {
	call, ok := e.(*bf.CallExpr)
	if !ok {
		return e
	}
	rule := bf.Rule{Call: call}
	from := label.New("", pkgRel, rule.Name())

	var resolve func(imp string, from label.Label) (label.Label, error)
	var embeds []label.Label
	switch rule.Kind() {
	case "go_library", "go_binary", "go_test":
		resolve = r.resolveGo
		embeds = getEmbedsGo(call, from)
	case "proto_library":
		resolve = r.resolveProto
	case "go_proto_library", "go_grpc_library":
		resolve = r.resolveGoProto
		embeds = getEmbedsGo(call, from)
	default:
		return e
	}

	resolved := *call
	resolved.List = append([]bf.Expr{}, call.List...)
	rule.Call = &resolved

	imports := rule.Attr(config.GazelleImportsKey)
	rule.DelAttr(config.GazelleImportsKey)
	rule.DelAttr("deps")
	deps := mapExprStrings(imports, func(imp string) string {
		label, err := resolve(imp, from)
		if err != nil {
			switch err.(type) {
			case standardImportError, selfImportError:
				return ""
			default:
				log.Print(err)
				return ""
			}
		}
		for _, e := range embeds {
			if label.Equal(e) {
				return ""
			}
		}
		label.Relative = label.Repo == "" && label.Pkg == pkgRel
		return label.String()
	})
	if deps != nil {
		rule.SetAttr("deps", deps)
	}

	return &resolved
}

type standardImportError struct {
	imp string
}

func (e standardImportError) Error() string {
	return fmt.Sprintf("import path %q is in the standard library", e.imp)
}

// mapExprStrings applies a function f to the strings in e and returns a new
// expression with the results. Scalar strings, lists, dicts, selects, and
// concatenations are supported.
func mapExprStrings(e bf.Expr, f func(string) string) bf.Expr {
	if e == nil {
		return nil
	}
	switch expr := e.(type) {
	case *bf.StringExpr:
		s := f(expr.Value)
		if s == "" {
			return nil
		}
		ret := *expr
		ret.Value = s
		return &ret

	case *bf.ListExpr:
		var list []bf.Expr
		for _, elem := range expr.List {
			elem = mapExprStrings(elem, f)
			if elem != nil {
				list = append(list, elem)
			}
		}
		if len(list) == 0 && len(expr.List) > 0 {
			return nil
		}
		ret := *expr
		ret.List = list
		return &ret

	case *bf.DictExpr:
		var cases []bf.Expr
		isEmpty := true
		for _, kv := range expr.List {
			keyval, ok := kv.(*bf.KeyValueExpr)
			if !ok {
				log.Panicf("unexpected expression in generated imports dict: %#v", kv)
			}
			value := mapExprStrings(keyval.Value, f)
			if value != nil {
				cases = append(cases, &bf.KeyValueExpr{Key: keyval.Key, Value: value})
				if key, ok := keyval.Key.(*bf.StringExpr); !ok || key.Value != "//conditions:default" {
					isEmpty = false
				}
			}
		}
		if isEmpty {
			return nil
		}
		ret := *expr
		ret.List = cases
		return &ret

	case *bf.CallExpr:
		if x, ok := expr.X.(*bf.LiteralExpr); !ok || x.Token != "select" || len(expr.List) != 1 {
			log.Panicf("unexpected call expression in generated imports: %#v", e)
		}
		arg := mapExprStrings(expr.List[0], f)
		if arg == nil {
			return nil
		}
		call := *expr
		call.List[0] = arg
		return &call

	case *bf.BinaryExpr:
		x := mapExprStrings(expr.X, f)
		y := mapExprStrings(expr.Y, f)
		if x == nil {
			return y
		}
		if y == nil {
			return x
		}
		binop := *expr
		binop.X = x
		binop.Y = y
		return &binop

	default:
		log.Panicf("unexpected expression in generated imports: %#v", e)
		return nil
	}
}

// resolveGo resolves an import path from a Go source file to a label.
// pkgRel is the path to the Go package relative to the repository root; it
// is used to resolve relative imports.
func (r *Resolver) resolveGo(imp string, from label.Label) (label.Label, error) {
	if build.IsLocalImport(imp) {
		cleanRel := path.Clean(path.Join(from.Pkg, imp))
		if build.IsLocalImport(cleanRel) {
			return label.NoLabel, fmt.Errorf("relative import path %q from %q points outside of repository", imp, from.Pkg)
		}
		imp = path.Join(r.c.GoPrefix, cleanRel)
	}

	if IsStandard(imp) {
		return label.NoLabel, standardImportError{imp}
	}

	if l := resolveWellKnownGo(imp); !l.Equal(label.NoLabel) {
		return l, nil
	}

	if l, err := r.ix.findLabelByImport(importSpec{config.GoLang, imp}, config.GoLang, from); err != nil {
		if _, ok := err.(ruleNotFoundError); !ok {
			return label.NoLabel, err
		}
	} else {
		return l, nil
	}

	if pathtools.HasPrefix(imp, r.c.GoPrefix) {
		return r.l.LibraryLabel(pathtools.TrimPrefix(imp, r.c.GoPrefix)), nil
	}

	return r.external.resolve(imp)
}

// resolveProto resolves an import statement in a .proto file to a label
// for a proto_library rule.
func (r *Resolver) resolveProto(imp string, from label.Label) (label.Label, error) {
	if !strings.HasSuffix(imp, ".proto") {
		return label.NoLabel, fmt.Errorf("can't import non-proto: %q", imp)
	}
	if isWellKnownProto(imp) {
		name := path.Base(imp[:len(imp)-len(".proto")]) + "_proto"
		return label.New(config.WellKnownTypesProtoRepo, "", name), nil
	}

	if l, err := r.ix.findLabelByImport(importSpec{config.ProtoLang, imp}, config.ProtoLang, from); err != nil {
		if _, ok := err.(ruleNotFoundError); !ok {
			return label.NoLabel, err
		}
	} else {
		return l, nil
	}

	rel := path.Dir(imp)
	if rel == "." {
		rel = ""
	}
	name := pathtools.RelBaseName(rel, r.c.GoPrefix, r.c.RepoRoot)
	return r.l.ProtoLabel(rel, name), nil
}

// resolveGoProto resolves an import statement in a .proto file to a
// label for a go_library rule that embeds the corresponding go_proto_library.
func (r *Resolver) resolveGoProto(imp string, from label.Label) (label.Label, error) {
	if !strings.HasSuffix(imp, ".proto") {
		return label.NoLabel, fmt.Errorf("can't import non-proto: %q", imp)
	}
	stem := imp[:len(imp)-len(".proto")]

	if isWellKnownProto(stem) {
		return label.NoLabel, standardImportError{imp}
	}

	if l, err := r.ix.findLabelByImport(importSpec{config.ProtoLang, imp}, config.GoLang, from); err != nil {
		if _, ok := err.(ruleNotFoundError); !ok {
			return label.NoLabel, err
		}
	} else {
		return l, err
	}

	// As a fallback, guess the label based on the proto file name. We assume
	// all proto files in a directory belong to the same package, and the
	// package name matches the directory base name. We also assume that protos
	// in the vendor directory must refer to something else in vendor.
	rel := path.Dir(imp)
	if rel == "." {
		rel = ""
	}
	if from.Pkg == "vendor" || strings.HasPrefix(from.Pkg, "vendor/") {
		rel = path.Join("vendor", rel)
	}
	return r.l.LibraryLabel(rel), nil
}

func getEmbedsGo(call *bf.CallExpr, from label.Label) []label.Label {
	rule := bf.Rule{Call: call}
	embedStrings := rule.AttrStrings("embed")
	embedLabels := make([]label.Label, 0, len(embedStrings))
	for _, s := range embedStrings {
		l, err := label.Parse(s)
		if err != nil {
			continue
		}
		l = l.Abs(from.Repo, from.Pkg)
		embedLabels = append(embedLabels, l)
	}
	return embedLabels
}

// IsStandard returns whether a package is in the standard library.
func IsStandard(imp string) bool {
	return stdPackages[imp]
}

func isWellKnownProto(imp string) bool {
	return pathtools.HasPrefix(imp, config.WellKnownTypesProtoPrefix) && pathtools.TrimPrefix(imp, config.WellKnownTypesProtoPrefix) == path.Base(imp)
}

func resolveWellKnownGo(imp string) label.Label {
	// keep in sync with @io_bazel_rules_go//proto/wkt:well_known_types.bzl
	// TODO(jayconrod): in well_known_types.bzl, write the import paths and
	// targets in a public dict. Import it here, and use it to generate this code.
	switch imp {
	case "github.com/golang/protobuf/ptypes/any",
		"github.com/golang/protobuf/ptypes/api",
		"github.com/golang/protobuf/protoc-gen-go/descriptor",
		"github.com/golang/protobuf/ptypes/duration",
		"github.com/golang/protobuf/ptypes/empty",
		"google.golang.org/genproto/protobuf/field_mask",
		"google.golang.org/genproto/protobuf/source_context",
		"github.com/golang/protobuf/ptypes/struct",
		"github.com/golang/protobuf/ptypes/timestamp",
		"github.com/golang/protobuf/ptypes/wrappers":
		return label.Label{
			Repo: config.RulesGoRepoName,
			Pkg:  config.WellKnownTypesPkg,
			Name: path.Base(imp) + "_go_proto",
		}
	case "github.com/golang/protobuf/protoc-gen-go/plugin":
		return label.Label{
			Repo: config.RulesGoRepoName,
			Pkg:  config.WellKnownTypesPkg,
			Name: "compiler_plugin_go_proto",
		}
	case "google.golang.org/genproto/protobuf/ptype":
		return label.Label{
			Repo: config.RulesGoRepoName,
			Pkg:  config.WellKnownTypesPkg,
			Name: "type_go_proto",
		}
	}
	return label.NoLabel
}

func isWellKnownGo(imp string) bool {
	prefix := config.WellKnownTypesGoPrefix + "/ptypes/"
	return strings.HasPrefix(imp, prefix) && strings.TrimPrefix(imp, prefix) == path.Base(imp)
}
