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

package proto

import (
	"errors"
	"fmt"
	"log"
	"path"
	"sort"
	"strings"

	"github.com/bazelbuild/bazel-gazelle/config"
	"github.com/bazelbuild/bazel-gazelle/label"
	"github.com/bazelbuild/bazel-gazelle/repo"
	"github.com/bazelbuild/bazel-gazelle/resolve"
	"github.com/bazelbuild/bazel-gazelle/rule"
)

func (_ *protoLang) Imports(c *config.Config, r *rule.Rule, f *rule.File) []resolve.ImportSpec {
	rel := f.Pkg
	srcs := r.AttrStrings("srcs")
	imports := make([]resolve.ImportSpec, len(srcs))
	pc := GetProtoConfig(c)
	prefix := rel
	if pc.stripImportPrefix != "" {
		prefix = strings.TrimPrefix(rel, pc.stripImportPrefix[1:])
		if rel == prefix {
			return nil
		}
	}
	if pc.importPrefix != "" {
		prefix = path.Join(pc.importPrefix, prefix)
	}
	for i, src := range srcs {
		imports[i] = resolve.ImportSpec{Lang: "proto", Imp: path.Join(prefix, src)}
	}
	return imports
}

func (_ *protoLang) Embeds(r *rule.Rule, from label.Label) []label.Label {
	return nil
}

func (_ *protoLang) Resolve(c *config.Config, ix *resolve.RuleIndex, rc *repo.RemoteCache, r *rule.Rule, importsRaw interface{}, from label.Label) {
	if importsRaw == nil {
		// may not be set in tests.
		return
	}
	imports := importsRaw.([]string)
	r.DelAttr("deps")
	depSet := make(map[string]bool)
	for _, imp := range imports {
		l, err := resolveProto(c, ix, r, imp, from)
		if err == skipImportError {
			continue
		} else if err != nil {
			log.Print(err)
		} else {
			l = l.Rel(from.Repo, from.Pkg)
			depSet[l.String()] = true
		}
	}
	if len(depSet) > 0 {
		deps := make([]string, 0, len(depSet))
		for dep := range depSet {
			deps = append(deps, dep)
		}
		sort.Strings(deps)
		r.SetAttr("deps", deps)
	}
}

var (
	skipImportError = errors.New("std import")
	notFoundError   = errors.New("not found")
)

func resolveProto(c *config.Config, ix *resolve.RuleIndex, r *rule.Rule, imp string, from label.Label) (label.Label, error) {
	pc := GetProtoConfig(c)
	if !strings.HasSuffix(imp, ".proto") {
		return label.NoLabel, fmt.Errorf("can't import non-proto: %q", imp)
	}

	if l, ok := resolve.FindRuleWithOverride(c, resolve.ImportSpec{Imp: imp, Lang: "proto"}, "proto"); ok {
		return l, nil
	}

	if l, ok := knownImports[imp]; ok && pc.Mode.ShouldUseKnownImports() {
		if l.Equal(from) {
			return label.NoLabel, skipImportError
		} else {
			return l, nil
		}
	}

	if l, err := resolveWithIndex(ix, imp, from); err == nil || err == skipImportError {
		return l, err
	} else if err != notFoundError {
		return label.NoLabel, err
	}

	rel := path.Dir(imp)
	if rel == "." {
		rel = ""
	}
	name := RuleName(rel)
	return label.New("", rel, name), nil
}

func resolveWithIndex(ix *resolve.RuleIndex, imp string, from label.Label) (label.Label, error) {
	matches := ix.FindRulesByImport(resolve.ImportSpec{Lang: "proto", Imp: imp}, "proto")
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
