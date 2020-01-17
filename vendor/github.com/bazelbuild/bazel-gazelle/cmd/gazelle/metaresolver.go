/* Copyright 2019 The Bazel Authors. All rights reserved.

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
	"github.com/bazelbuild/bazel-gazelle/config"
	"github.com/bazelbuild/bazel-gazelle/resolve"
	"github.com/bazelbuild/bazel-gazelle/rule"
)

// metaResolver provides a rule.Resolver for any rule.Rule.
type metaResolver struct {
	// builtins provides a map of the language kinds to their resolver.
	builtins map[string]resolve.Resolver

	// mappedKinds provides a list of replacements used by File.Pkg.
	mappedKinds map[string][]config.MappedKind
}

func newMetaResolver() *metaResolver {
	return &metaResolver{
		builtins:    make(map[string]resolve.Resolver),
		mappedKinds: make(map[string][]config.MappedKind),
	}
}

// AddBuiltin registers a builtin kind with its info.
func (mr *metaResolver) AddBuiltin(kindName string, resolver resolve.Resolver) {
	mr.builtins[kindName] = resolver
}

// MappedKind records the fact that the given mapping was applied while
// processing the given package.
func (mr *metaResolver) MappedKind(pkgRel string, kind config.MappedKind) {
	mr.mappedKinds[pkgRel] = append(mr.mappedKinds[pkgRel], kind)
}

// Resolver returns a resolver for the given rule and package, and a bool
// indicating whether one was found. Empty string may be passed for pkgRel,
// which results in consulting the builtin kinds only.
func (mr metaResolver) Resolver(r *rule.Rule, pkgRel string) resolve.Resolver {
	for _, mappedKind := range mr.mappedKinds[pkgRel] {
		if mappedKind.KindName == r.Kind() {
			return mr.builtins[mappedKind.FromKind]
		}
	}
	return mr.builtins[r.Kind()]
}
