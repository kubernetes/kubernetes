/* Copyright 2017 The Bazel Authors. All rights reserved.

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
	"log"
	"path"
	"path/filepath"
	"strings"

	"github.com/bazelbuild/bazel-gazelle/internal/config"
	"github.com/bazelbuild/bazel-gazelle/internal/label"
	bf "github.com/bazelbuild/buildtools/build"
)

// RuleIndex is a table of rules in a workspace, indexed by label and by
// import path. Used by Resolver to map import paths to labels.
type RuleIndex struct {
	rules     []*ruleRecord
	labelMap  map[label.Label]*ruleRecord
	importMap map[importSpec][]*ruleRecord
}

// ruleRecord contains information about a rule relevant to import indexing.
type ruleRecord struct {
	rule       bf.Rule
	label      label.Label
	lang       config.Language
	importedAs []importSpec
	embedded   bool
}

// importSpec describes a package to be imported. Language is specified, since
// different languages have different formats for their imports.
type importSpec struct {
	lang config.Language
	imp  string
}

func NewRuleIndex() *RuleIndex {
	return &RuleIndex{
		labelMap: make(map[label.Label]*ruleRecord),
	}
}

// AddRulesFromFile adds existing rules to the index from file
// (which must not be nil).
func (ix *RuleIndex) AddRulesFromFile(c *config.Config, file *bf.File) {
	buildRel, err := filepath.Rel(c.RepoRoot, file.Path)
	if err != nil {
		log.Panicf("file not in repo: %s", file.Path)
	}
	buildRel = path.Dir(filepath.ToSlash(buildRel))
	if buildRel == "." || buildRel == "/" {
		buildRel = ""
	}

	for _, stmt := range file.Stmt {
		if call, ok := stmt.(*bf.CallExpr); ok {
			ix.addRule(call, c.GoPrefix, buildRel)
		}
	}
}

func (ix *RuleIndex) addRule(call *bf.CallExpr, goPrefix, buildRel string) {
	rule := bf.Rule{Call: call}
	record := &ruleRecord{
		rule:  rule,
		label: label.New("", buildRel, rule.Name()),
	}

	if _, ok := ix.labelMap[record.label]; ok {
		log.Printf("multiple rules found with label %s", record.label)
		return
	}

	kind := rule.Kind()
	switch {
	case isGoLibrary(kind):
		record.lang = config.GoLang
		if imp := rule.AttrString("importpath"); imp != "" {
			record.importedAs = []importSpec{{lang: config.GoLang, imp: imp}}
		}
		// Additional proto imports may be added in Finish.

	case kind == "proto_library":
		record.lang = config.ProtoLang
		for _, s := range findSources(rule, buildRel, ".proto") {
			record.importedAs = append(record.importedAs, importSpec{lang: config.ProtoLang, imp: s})
		}

	default:
		return
	}

	ix.rules = append(ix.rules, record)
	ix.labelMap[record.label] = record
}

// Finish constructs the import index and performs any other necessary indexing
// actions after all rules have been added. This step is necessary because
// a rule may be indexed differently based on what rules are added later.
//
// This function must be called after all AddRulesFromFile calls but before any
// findRuleByImport calls.
func (ix *RuleIndex) Finish() {
	ix.skipGoEmbds()
	ix.buildImportIndex()
}

// skipGoEmbeds sets the embedded flag on Go library rules that are imported
// by other Go library rules with the same import path. Note that embedded
// rules may still be imported with non-Go imports. For example, a
// go_proto_library may be imported with either a Go import path or a proto
// path. If the library is embedded, only the proto path will be indexed.
func (ix *RuleIndex) skipGoEmbds() {
	for _, r := range ix.rules {
		if !isGoLibrary(r.rule.Kind()) {
			continue
		}
		importpath := r.rule.AttrString("importpath")

		var embedLabels []label.Label
		if embedList, ok := r.rule.Attr("embed").(*bf.ListExpr); ok {
			for _, embedElem := range embedList.List {
				embedStr, ok := embedElem.(*bf.StringExpr)
				if !ok {
					continue
				}
				embedLabel, err := label.Parse(embedStr.Value)
				if err != nil {
					continue
				}
				embedLabels = append(embedLabels, embedLabel)
			}
		}
		if libraryStr, ok := r.rule.Attr("library").(*bf.StringExpr); ok {
			if libraryLabel, err := label.Parse(libraryStr.Value); err == nil {
				embedLabels = append(embedLabels, libraryLabel)
			}
		}

		for _, l := range embedLabels {
			embed, ok := ix.findRuleByLabel(l, r.label)
			if !ok {
				continue
			}
			if embed.rule.AttrString("importpath") != importpath {
				continue
			}
			embed.embedded = true
		}
	}
}

// buildImportIndex constructs the map used by findRuleByImport.
func (ix *RuleIndex) buildImportIndex() {
	ix.importMap = make(map[importSpec][]*ruleRecord)
	for _, r := range ix.rules {
		if isGoProtoLibrary(r.rule.Kind()) {
			protoImports := findGoProtoSources(ix, r)
			r.importedAs = append(r.importedAs, protoImports...)
		}
		for _, imp := range r.importedAs {
			if imp.lang == config.GoLang && r.embedded {
				continue
			}
			ix.importMap[imp] = append(ix.importMap[imp], r)
		}
	}
}

type ruleNotFoundError struct {
	from label.Label
	imp  string
}

func (e ruleNotFoundError) Error() string {
	return fmt.Sprintf("no rule found for import %q, needed in %s", e.imp, e.from)
}

type selfImportError struct {
	from label.Label
	imp  string
}

func (e selfImportError) Error() string {
	return fmt.Sprintf("rule %s imports itself with path %q", e.from, e.imp)
}

func (ix *RuleIndex) findRuleByLabel(label label.Label, from label.Label) (*ruleRecord, bool) {
	label = label.Abs(from.Repo, from.Pkg)
	r, ok := ix.labelMap[label]
	return r, ok
}

// findRuleByImport attempts to resolve an import string to a rule record.
// imp is the import to resolve (which includes the target language). lang is
// the language of the rule with the dependency (for example, in
// go_proto_library, imp will have ProtoLang and lang will be GoLang).
// from is the rule which is doing the dependency. This is used to check
// vendoring visibility and to check for self-imports.
//
// Any number of rules may provide the same import. If no rules provide the
// import, ruleNotFoundError is returned. If a rule imports itself,
// selfImportError is returned. If multiple rules provide the import, this
// function will attempt to choose one based on Go vendoring logic.  In
// ambiguous cases, an error is returned.
func (ix *RuleIndex) findRuleByImport(imp importSpec, lang config.Language, from label.Label) (*ruleRecord, error) {
	matches := ix.importMap[imp]
	var bestMatch *ruleRecord
	var bestMatchIsVendored bool
	var bestMatchVendorRoot string
	var matchError error
	for _, m := range matches {
		if m.lang != lang {
			continue
		}

		switch imp.lang {
		case config.GoLang:
			// Apply vendoring logic for Go libraries. A library in a vendor directory
			// is only visible in the parent tree. Vendored libraries supercede
			// non-vendored libraries, and libraries closer to from.Pkg supercede
			// those further up the tree.
			isVendored := false
			vendorRoot := ""
			parts := strings.Split(m.label.Pkg, "/")
			for i := len(parts) - 1; i >= 0; i-- {
				if parts[i] == "vendor" {
					isVendored = true
					vendorRoot = strings.Join(parts[:i], "/")
					break
				}
			}
			if isVendored && !label.New(m.label.Repo, vendorRoot, "").Contains(from) {
				// vendor directory not visible
				continue
			}
			if bestMatch == nil || isVendored && (!bestMatchIsVendored || len(vendorRoot) > len(bestMatchVendorRoot)) {
				// Current match is better
				bestMatch = m
				bestMatchIsVendored = isVendored
				bestMatchVendorRoot = vendorRoot
				matchError = nil
			} else if (!isVendored && bestMatchIsVendored) || (isVendored && len(vendorRoot) < len(bestMatchVendorRoot)) {
				// Current match is worse
			} else {
				// Match is ambiguous
				matchError = fmt.Errorf("multiple rules (%s and %s) may be imported with %q from %s", bestMatch.label, m.label, imp.imp, from)
			}

		default:
			if bestMatch == nil {
				bestMatch = m
			} else {
				matchError = fmt.Errorf("multiple rules (%s and %s) may be imported with %q from %s", bestMatch.label, m.label, imp.imp, from)
			}
		}
	}
	if matchError != nil {
		return nil, matchError
	}
	if bestMatch == nil {
		return nil, ruleNotFoundError{from, imp.imp}
	}
	if bestMatch.label.Equal(from) {
		return nil, selfImportError{from, imp.imp}
	}

	if imp.lang == config.ProtoLang && lang == config.GoLang {
		importpath := bestMatch.rule.AttrString("importpath")
		if betterMatch, err := ix.findRuleByImport(importSpec{config.GoLang, importpath}, config.GoLang, from); err == nil {
			return betterMatch, nil
		}
	}

	return bestMatch, nil
}

func (ix *RuleIndex) findLabelByImport(imp importSpec, lang config.Language, from label.Label) (label.Label, error) {
	r, err := ix.findRuleByImport(imp, lang, from)
	if err != nil {
		return label.NoLabel, err
	}
	return r.label, nil
}

func findGoProtoSources(ix *RuleIndex, r *ruleRecord) []importSpec {
	protoLabel, err := label.Parse(r.rule.AttrString("proto"))
	if err != nil {
		return nil
	}
	proto, ok := ix.findRuleByLabel(protoLabel, r.label)
	if !ok {
		return nil
	}
	var importedAs []importSpec
	for _, source := range findSources(proto.rule, proto.label.Pkg, ".proto") {
		importedAs = append(importedAs, importSpec{lang: config.ProtoLang, imp: source})
	}
	return importedAs
}

func findSources(r bf.Rule, buildRel, ext string) []string {
	srcsExpr := r.Attr("srcs")
	srcsList, ok := srcsExpr.(*bf.ListExpr)
	if !ok {
		return nil
	}
	var srcs []string
	for _, srcExpr := range srcsList.List {
		src, ok := srcExpr.(*bf.StringExpr)
		if !ok {
			continue
		}
		label, err := label.Parse(src.Value)
		if err != nil || !label.Relative || !strings.HasSuffix(label.Name, ext) {
			continue
		}
		srcs = append(srcs, path.Join(buildRel, label.Name))
	}
	return srcs
}

func isGoLibrary(kind string) bool {
	return kind == "go_library" || isGoProtoLibrary(kind)
}

func isGoProtoLibrary(kind string) bool {
	return kind == "go_proto_library" || kind == "go_grpc_library"
}
