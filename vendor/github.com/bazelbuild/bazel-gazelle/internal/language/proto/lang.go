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

// Package proto provides support for protocol buffer rules.
// It generates proto_library rules only (not go_proto_library or any other
// language-specific implementations).
//
// Configuration
//
// Configuration is largely controlled by Mode. In disable mode, proto rules are
// left alone (neither generated nor deleted). In legacy mode, filegroups are
// emitted containing protos. In default mode, proto_library rules are
// emitted. The proto mode may be set with the -proto command line flag or the
// "# gazelle:proto" directive.
//
// The configuration is largely public, and other languages may depend on it.
// For example, go uses Mode to determine whether to generate go_proto_library
// rules and ignore static .pb.go files.
//
// Rule generation
//
// Currently, Gazelle generates at most one proto_library per directory. Protos
// in the same package are grouped together into a proto_library. If there are
// sources for multiple packages, the package name that matches the directory
// name will be chosen; if there is no such package, an error will be printed.
// We expect to provide support for multiple proto_libraries in the future
// when Go has support for multiple packages and we have better rule matching.
// The generated proto_library will be named after the directory, not the
// proto or the package. For example, for foo/bar/baz.proto, a proto_library
// rule will be generated named //foo/bar:bar_proto.
//
// Dependency resolution
//
// proto_library rules are indexed by their srcs attribute. Gazelle attempts
// to resolve proto imports (e.g., import foo/bar/bar.proto) to the
// proto_library that contains the named source file
// (e.g., //foo/bar:bar_proto). If no indexed proto_library provides the source
// file, Gazelle will guess a label, following conventions.
//
// No attempt is made to resolve protos to rules in external repositories,
// since there's no indication that a proto import comes from an external
// repository. In the future, build files in external repos will be indexed,
// so we can support this (#12).
//
// Gazelle has special cases for Well Known Types (i.e., imports of the form
// google/protobuf/*.proto). These are resolved to rules in
// @com_google_protobuf.
package proto

import "github.com/bazelbuild/bazel-gazelle/internal/language"

const protoName = "proto"

type protoLang struct{}

func (_ *protoLang) Name() string { return protoName }

func New() language.Language {
	return &protoLang{}
}
