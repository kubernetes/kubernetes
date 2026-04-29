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
	"strings"

	"k8s.io/code-generator/cmd/validation-gen/validators"
	"k8s.io/code-generator/pkg/guardrails"
	"k8s.io/gengo/v2/types"
)

// collectRules walks the typeNode tree at node and returns a field-path → rules
// map. Empty if node has no declared rules.
func collectRules(node *typeNode) map[string][]guardrails.Rule {
	if node == nil {
		return nil
	}
	rulesByPath := map[string][]guardrails.Rule{}
	seen := map[*typeNode]bool{}

	record := func(path string, fns []validators.FunctionGen) {
		for _, fn := range fns {
			recordEmission(rulesByPath, path, fn, "")
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
	return rulesByPath
}

// recordEmission descends fg, accumulating each Wrapper/MultiWrapperFunction
// PathFragment into suffix, and records (basePath+suffix, Rule) once an
// emitting function (Emits != nil) is reached.
func recordEmission(rulesByPath map[string][]guardrails.Rule, basePath string, fg validators.FunctionGen, suffix string) {
	if fg.Emits != nil {
		path := basePath + suffix
		rulesByPath[path] = append(rulesByPath[path], guardrails.Rule{
			ErrorType: string(fg.Emits.Type),
			Origin:    fg.Emits.Origin,
		})
		return
	}
	for _, arg := range fg.Args {
		switch a := arg.(type) {
		case validators.WrapperFunction:
			recordEmission(rulesByPath, basePath, a.Function, suffix+a.PathFragment)
		case validators.MultiWrapperFunction:
			for _, child := range a.Functions {
				recordEmission(rulesByPath, basePath, child, suffix+a.PathFragment)
			}
		}
	}
}

// joinPath joins seg onto base. Empty seg is a no-op so inline-embedded
// struct fields (whose jsonName is "") stay transparent.
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

// gvFor returns the (group, version) for a Go package. Group is read from the
// package's GroupName const (set in every k8s.io/api/<group>/<version>/register.go);
// pkg.Path is used as a last-resort identifier for non-API packages.
// Version is parsed from the import path when it matches that layout.
func gvFor(pkg *types.Package) (string, string) {
	version := ""
	if relPath, ok := strings.CutPrefix(pkg.Path, "k8s.io/api/"); ok {
		if parts := strings.Split(relPath, "/"); len(parts) == 2 {
			version = parts[1]
		}
	}

	// ConstValue holds the un-quoted string literal for string-typed consts.
	if c, ok := pkg.Constants["GroupName"]; ok && c.ConstValue != nil {
		return *c.ConstValue, version
	}

	return pkg.Path, ""
}
