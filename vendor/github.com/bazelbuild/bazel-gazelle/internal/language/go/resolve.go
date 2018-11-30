/* Copyright 2018 The Bazel Authors. All rights reserved.

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

package golang

import (
	"errors"
	"fmt"
	"go/build"
	"log"
	"path"
	"strings"

	"github.com/bazelbuild/bazel-gazelle/internal/config"
	"github.com/bazelbuild/bazel-gazelle/internal/label"
	"github.com/bazelbuild/bazel-gazelle/internal/language/proto"
	"github.com/bazelbuild/bazel-gazelle/internal/pathtools"
	"github.com/bazelbuild/bazel-gazelle/internal/repos"
	"github.com/bazelbuild/bazel-gazelle/internal/resolve"
	"github.com/bazelbuild/bazel-gazelle/internal/rule"
)

func (_ *goLang) Imports(_ *config.Config, r *rule.Rule, f *rule.File) []resolve.ImportSpec {
	if !isGoLibrary(r.Kind()) {
		return nil
	}
	if importPath := r.AttrString("importpath"); importPath == "" {
		return []resolve.ImportSpec{}
	} else {
		return []resolve.ImportSpec{{goName, importPath}}
	}
}

func (_ *goLang) Embeds(r *rule.Rule, from label.Label) []label.Label {
	embedStrings := r.AttrStrings("embed")
	if isGoProtoLibrary(r.Kind()) {
		embedStrings = append(embedStrings, r.AttrString("proto"))
	}
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

func (gl *goLang) Resolve(c *config.Config, ix *resolve.RuleIndex, rc *repos.RemoteCache, r *rule.Rule, from label.Label) {
	importsRaw := r.PrivateAttr(config.GazelleImportsKey)
	if importsRaw == nil {
		// may not be set in tests.
		return
	}
	imports := importsRaw.(rule.PlatformStrings)
	r.DelAttr("deps")
	resolve := resolveGo
	if r.Kind() == "go_proto_library" {
		resolve = resolveProto
	}
	deps, errs := imports.Map(func(imp string) (string, error) {
		l, err := resolve(c, ix, rc, r, imp, from)
		if err == skipImportError {
			return "", nil
		} else if err != nil {
			return "", err
		}
		for _, embed := range gl.Embeds(r, from) {
			if embed.Equal(l) {
				return "", nil
			}
		}
		l = l.Rel(from.Repo, from.Pkg)
		return l.String(), nil
	})
	for _, err := range errs {
		log.Print(err)
	}
	if !deps.IsEmpty() {
		if r.Kind() == "go_proto_library" {
			// protos may import the same library multiple times by different names,
			// so we need to de-duplicate them. Protos are not platform-specific,
			// so it's safe to just flatten them.
			r.SetAttr("deps", deps.Flat())
		} else {
			r.SetAttr("deps", deps)
		}
	}
}

var (
	skipImportError = errors.New("std or self import")
	notFoundError   = errors.New("rule not found")
)

func resolveGo(c *config.Config, ix *resolve.RuleIndex, rc *repos.RemoteCache, r *rule.Rule, imp string, from label.Label) (label.Label, error) {
	gc := getGoConfig(c)
	pc := proto.GetProtoConfig(c)
	if build.IsLocalImport(imp) {
		cleanRel := path.Clean(path.Join(from.Pkg, imp))
		if build.IsLocalImport(cleanRel) {
			return label.NoLabel, fmt.Errorf("relative import path %q from %q points outside of repository", imp, from.Pkg)
		}
		imp = path.Join(gc.prefix, cleanRel)
	}

	if isStandard(imp) {
		return label.NoLabel, skipImportError
	}

	if l, ok := resolve.FindRuleWithOverride(c, resolve.ImportSpec{Lang: "go", Imp: imp}, "go"); ok {
		return l, nil
	}

	if pc.Mode.ShouldUseKnownImports() {
		// These are commonly used libraries that depend on Well Known Types.
		// They depend on the generated versions of these protos to avoid conflicts.
		// However, since protoc-gen-go depends on these libraries, we generate
		// its rules in disable_global mode (to avoid cyclic dependency), so the
		// "go_default_library" versions of these libraries depend on the
		// pre-generated versions of the proto libraries.
		switch imp {
		case "github.com/golang/protobuf/jsonpb":
			return label.New("com_github_golang_protobuf", "jsonpb", "go_default_library_gen"), nil
		case "github.com/golang/protobuf/descriptor":
			return label.New("com_github_golang_protobuf", "descriptor", "go_default_library_gen"), nil
		case "github.com/golang/protobuf/ptypes":
			return label.New("com_github_golang_protobuf", "ptypes", "go_default_library_gen"), nil
		case "google.golang.org/grpc":
			return label.New("org_golang_google_grpc", "", "go_default_library"), nil
		}
		if l, ok := knownGoProtoImports[imp]; ok {
			return l, nil
		}
	}

	if l, err := resolveWithIndexGo(ix, imp, from); err == nil || err == skipImportError {
		return l, err
	} else if err != notFoundError {
		return label.NoLabel, err
	}

	if pathtools.HasPrefix(imp, gc.prefix) {
		pkg := path.Join(gc.prefixRel, pathtools.TrimPrefix(imp, gc.prefix))
		return label.New("", pkg, config.DefaultLibName), nil
	}

	if gc.depMode == externalMode {
		return resolveExternal(rc, imp)
	} else {
		return resolveVendored(rc, imp)
	}
}

// isStandard returns whether a package is in the standard library.
func isStandard(imp string) bool {
	return stdPackages[imp]
}

func resolveWithIndexGo(ix *resolve.RuleIndex, imp string, from label.Label) (label.Label, error) {
	matches := ix.FindRulesByImport(resolve.ImportSpec{Lang: "go", Imp: imp}, "go")
	var bestMatch resolve.FindResult
	var bestMatchIsVendored bool
	var bestMatchVendorRoot string
	var matchError error

	for _, m := range matches {
		// Apply vendoring logic for Go libraries. A library in a vendor directory
		// is only visible in the parent tree. Vendored libraries supercede
		// non-vendored libraries, and libraries closer to from.Pkg supercede
		// those further up the tree.
		isVendored := false
		vendorRoot := ""
		parts := strings.Split(m.Label.Pkg, "/")
		for i := len(parts) - 1; i >= 0; i-- {
			if parts[i] == "vendor" {
				isVendored = true
				vendorRoot = strings.Join(parts[:i], "/")
				break
			}
		}
		if isVendored {
		}
		if isVendored && !label.New(m.Label.Repo, vendorRoot, "").Contains(from) {
			// vendor directory not visible
			continue
		}
		if bestMatch.Label.Equal(label.NoLabel) || isVendored && (!bestMatchIsVendored || len(vendorRoot) > len(bestMatchVendorRoot)) {
			// Current match is better
			bestMatch = m
			bestMatchIsVendored = isVendored
			bestMatchVendorRoot = vendorRoot
			matchError = nil
		} else if (!isVendored && bestMatchIsVendored) || (isVendored && len(vendorRoot) < len(bestMatchVendorRoot)) {
			// Current match is worse
		} else {
			// Match is ambiguous
			matchError = fmt.Errorf("multiple rules (%s and %s) may be imported with %q from %s", bestMatch.Label, m.Label, imp, from)
		}
	}
	if matchError != nil {
		return label.NoLabel, matchError
	}
	if bestMatch.Label.Equal(label.NoLabel) {
		return label.NoLabel, notFoundError
	}
	if bestMatch.IsSelfImport(from) {
		return label.NoLabel, skipImportError
	}
	return bestMatch.Label, nil
}

func resolveExternal(rc *repos.RemoteCache, imp string) (label.Label, error) {
	prefix, repo, err := rc.Root(imp)
	if err != nil {
		return label.NoLabel, err
	}

	var pkg string
	if imp != prefix {
		pkg = pathtools.TrimPrefix(imp, prefix)
	}

	return label.New(repo, pkg, config.DefaultLibName), nil
}

func resolveVendored(rc *repos.RemoteCache, imp string) (label.Label, error) {
	return label.New("", path.Join("vendor", imp), config.DefaultLibName), nil
}

func resolveProto(c *config.Config, ix *resolve.RuleIndex, rc *repos.RemoteCache, r *rule.Rule, imp string, from label.Label) (label.Label, error) {
	pc := proto.GetProtoConfig(c)

	if wellKnownProtos[imp] {
		return label.NoLabel, skipImportError
	}

	if l, ok := resolve.FindRuleWithOverride(c, resolve.ImportSpec{Lang: "proto", Imp: imp}, "go"); ok {
		return l, nil
	}

	if l, ok := knownProtoImports[imp]; ok && pc.Mode.ShouldUseKnownImports() {
		if l.Equal(from) {
			return label.NoLabel, skipImportError
		} else {
			return l, nil
		}
	}

	if l, err := resolveWithIndexProto(ix, imp, from); err == nil || err == skipImportError {
		return l, err
	} else if err != notFoundError {
		return label.NoLabel, err
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
	return label.New("", rel, config.DefaultLibName), nil
}

// wellKnownProtos is the set of proto sets for which we don't need to add
// an explicit dependency in go_proto_library.
// TODO(jayconrod): generate from
// @io_bazel_rules_go//proto/wkt:WELL_KNOWN_TYPE_PACKAGES
var wellKnownProtos = map[string]bool{
	"google/protobuf/any.proto":             true,
	"google/protobuf/api.proto":             true,
	"google/protobuf/compiler_plugin.proto": true,
	"google/protobuf/descriptor.proto":      true,
	"google/protobuf/duration.proto":        true,
	"google/protobuf/empty.proto":           true,
	"google/protobuf/field_mask.proto":      true,
	"google/protobuf/source_context.proto":  true,
	"google/protobuf/struct.proto":          true,
	"google/protobuf/timestamp.proto":       true,
	"google/protobuf/type.proto":            true,
	"google/protobuf/wrappers.proto":        true,
}

func resolveWithIndexProto(ix *resolve.RuleIndex, imp string, from label.Label) (label.Label, error) {
	matches := ix.FindRulesByImport(resolve.ImportSpec{Lang: "proto", Imp: imp}, "go")
	if len(matches) == 0 {
		return label.NoLabel, notFoundError
	}
	if len(matches) > 1 {
		return label.NoLabel, fmt.Errorf("multiple rules (%s and %s) may be imported with %q from %s", matches[0].Label, matches[1].Label, imp, from)
	}
	if matches[0].IsSelfImport(from) {
		return label.NoLabel, skipImportError
	}
	return matches[0].Label, nil
}

func isGoLibrary(kind string) bool {
	return kind == "go_library" || isGoProtoLibrary(kind)
}

func isGoProtoLibrary(kind string) bool {
	return kind == "go_proto_library" || kind == "go_grpc_library"
}
