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

import "github.com/bazelbuild/bazel-gazelle/internal/rule"

var goKinds = map[string]rule.KindInfo{
	"filegroup": {
		NonEmptyAttrs:  map[string]bool{"srcs": true},
		MergeableAttrs: map[string]bool{"srcs": true},
	},
	"go_binary": {
		MatchAny: true,
		NonEmptyAttrs: map[string]bool{
			"deps":  true,
			"embed": true,
			"srcs":  true,
		},
		SubstituteAttrs: map[string]bool{"embed": true},
		MergeableAttrs: map[string]bool{
			"cgo":       true,
			"clinkopts": true,
			"copts":     true,
			"embed":     true,
			"srcs":      true,
		},
		ResolveAttrs: map[string]bool{"deps": true},
	},
	"go_library": {
		MatchAttrs: []string{"importpath"},
		NonEmptyAttrs: map[string]bool{
			"deps":  true,
			"embed": true,
			"srcs":  true,
		},
		SubstituteAttrs: map[string]bool{
			"embed": true,
		},
		MergeableAttrs: map[string]bool{
			"cgo":        true,
			"clinkopts":  true,
			"copts":      true,
			"embed":      true,
			"importmap":  true,
			"importpath": true,
			"srcs":       true,
		},
		ResolveAttrs: map[string]bool{"deps": true},
	},
	"go_proto_library": {
		MatchAttrs: []string{"importpath"},
		NonEmptyAttrs: map[string]bool{
			"deps":  true,
			"embed": true,
			"proto": true,
			"srcs":  true,
		},
		SubstituteAttrs: map[string]bool{"proto": true},
		MergeableAttrs: map[string]bool{
			"srcs":       true,
			"importpath": true,
			"importmap":  true,
			"cgo":        true,
			"clinkopts":  true,
			"copts":      true,
			"embed":      true,
			"proto":      true,
		},
		ResolveAttrs: map[string]bool{"deps": true},
	},
	"go_repository": {
		MatchAttrs:    []string{"importpath"},
		NonEmptyAttrs: nil, // never empty
		MergeableAttrs: map[string]bool{
			"commit":       true,
			"importpath":   true,
			"remote":       true,
			"sha256":       true,
			"strip_prefix": true,
			"tag":          true,
			"type":         true,
			"urls":         true,
			"vcs":          true,
		},
	},
	"go_test": {
		NonEmptyAttrs: map[string]bool{
			"deps":  true,
			"embed": true,
			"srcs":  true,
		},
		MergeableAttrs: map[string]bool{
			"cgo":       true,
			"clinkopts": true,
			"copts":     true,
			"embed":     true,
			"srcs":      true,
		},
		ResolveAttrs: map[string]bool{"deps": true},
	},
}

var goLoads = []rule.LoadInfo{
	{
		Name: "@io_bazel_rules_go//go:def.bzl",
		Symbols: []string{
			"cgo_library",
			"go_binary",
			"go_library",
			"go_prefix",
			"go_repository",
			"go_test",
		},
	}, {
		Name: "@io_bazel_rules_go//proto:def.bzl",
		Symbols: []string{
			"go_grpc_library",
			"go_proto_library",
		},
	}, {
		Name: "@bazel_gazelle//:deps.bzl",
		Symbols: []string{
			"go_repository",
		},
		After: []string{
			"go_rules_dependencies",
			"go_register_toolchains",
			"gazelle_dependencies",
		},
	},
}

func (_ *goLang) Kinds() map[string]rule.KindInfo { return goKinds }
func (_ *goLang) Loads() []rule.LoadInfo          { return goLoads }
