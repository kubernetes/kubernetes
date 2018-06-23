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

package config

import (
	"log"
	"path"
	"regexp"
	"strings"

	bf "github.com/bazelbuild/buildtools/build"
)

// Directive is a key-value pair extracted from a top-level comment in
// a build file. Directives have the following format:
//
//     # gazelle:key value
//
// Keys may not contain spaces. Values may be empty and may contain spaces,
// but surrounding space is trimmed.
type Directive struct {
	Key, Value string
}

// Top-level directives apply to the whole package or build file. They must
// appear before the first statement.
var knownTopLevelDirectives = map[string]bool{
	"build_file_name":  true,
	"build_tags":       true,
	"exclude":          true,
	"ignore":           true,
	"importmap_prefix": true,
	"repo":             true,
	"prefix":           true,
	"proto":            true,
}

// TODO(jayconrod): annotation directives will apply to an individual rule.
// They must appear in the block of comments above that rule.

// ParseDirectives scans f for Gazelle directives. The full list of directives
// is returned. Errors are reported for unrecognized directives and directives
// out of place (after the first statement).
func ParseDirectives(f *bf.File) []Directive {
	var directives []Directive
	parseComment := func(com bf.Comment) {
		match := directiveRe.FindStringSubmatch(com.Token)
		if match == nil {
			return
		}
		key, value := match[1], match[2]
		if _, ok := knownTopLevelDirectives[key]; !ok {
			log.Printf("%s:%d: unknown directive: %s", f.Path, com.Start.Line, com.Token)
			return
		}
		directives = append(directives, Directive{key, value})
	}

	for _, s := range f.Stmt {
		coms := s.Comment()
		for _, com := range coms.Before {
			parseComment(com)
		}
		for _, com := range coms.After {
			parseComment(com)
		}
	}
	return directives
}

var directiveRe = regexp.MustCompile(`^#\s*gazelle:(\w+)\s*(.*?)\s*$`)

// ApplyDirectives applies directives that modify the configuration to a copy of
// c, which is returned. If there are no configuration directives, c is returned
// unmodified.
func ApplyDirectives(c *Config, directives []Directive, rel string) *Config {
	modified := *c
	didModify := false
	for _, d := range directives {
		switch d.Key {
		case "build_file_name":
			modified.ValidBuildFileNames = strings.Split(d.Value, ",")
			didModify = true
		case "build_tags":
			if err := modified.SetBuildTags(d.Value); err != nil {
				log.Print(err)
				modified.GenericTags = c.GenericTags
			} else {
				modified.PreprocessTags()
				didModify = true
			}
		case "importmap_prefix":
			if err := CheckPrefix(d.Value); err != nil {
				log.Print(err)
				continue
			}
			modified.GoImportMapPrefix = d.Value
			modified.GoImportMapPrefixRel = rel
			didModify = true
		case "prefix":
			if err := CheckPrefix(d.Value); err != nil {
				log.Print(err)
				continue
			}
			modified.GoPrefix = d.Value
			modified.GoPrefixRel = rel
			didModify = true
		case "proto":
			protoMode, err := ProtoModeFromString(d.Value)
			if err != nil {
				log.Print(err)
				continue
			}
			modified.ProtoMode = protoMode
			modified.ProtoModeExplicit = true
			didModify = true
		}
	}
	if !didModify {
		return c
	}
	return &modified
}

// InferProtoMode sets Config.ProtoMode, based on the contents of f.  If the
// proto mode is already set to something other than the default, or if the mode
// is set explicitly in directives, this function does not change it. If the
// legacy go_proto_library.bzl is loaded, or if this is the Well Known Types
// repository, legacy mode is used. If go_proto_library is loaded from another
// file, proto rule generation is disabled.
func InferProtoMode(c *Config, rel string, f *bf.File, directives []Directive) *Config {
	if c.ProtoMode != DefaultProtoMode || c.ProtoModeExplicit {
		return c
	}
	for _, d := range directives {
		if d.Key == "proto" {
			return c
		}
	}
	if c.GoPrefix == WellKnownTypesGoPrefix {
		// Use legacy mode in this repo. We don't need proto_library or
		// go_proto_library, since we get that from @com_google_protobuf.
		// Legacy rules still refer to .proto files in here, which need are
		// exposed by filegroup. go_library rules from .pb.go files will be
		// generated, which are depended upon by the new rules.
		modified := *c
		modified.ProtoMode = LegacyProtoMode
		return &modified
	}
	if path.Base(rel) == "vendor" {
		modified := *c
		modified.ProtoMode = DisableProtoMode
		return &modified
	}
	if f == nil {
		return c
	}
	mode := DefaultProtoMode
	for _, stmt := range f.Stmt {
		c, ok := stmt.(*bf.CallExpr)
		if !ok {
			continue
		}
		x, ok := c.X.(*bf.LiteralExpr)
		if !ok || x.Token != "load" || len(c.List) == 0 {
			continue
		}
		name, ok := c.List[0].(*bf.StringExpr)
		if !ok {
			continue
		}
		if name.Value == "@io_bazel_rules_go//proto:def.bzl" {
			break
		}
		if name.Value == "@io_bazel_rules_go//proto:go_proto_library.bzl" {
			mode = LegacyProtoMode
			break
		}
		for _, arg := range c.List[1:] {
			if sym, ok := arg.(*bf.StringExpr); ok && sym.Value == "go_proto_library" {
				mode = DisableProtoMode
				break
			}
			kwarg, ok := arg.(*bf.BinaryExpr)
			if !ok || kwarg.Op != "=" {
				continue
			}
			if key, ok := kwarg.X.(*bf.LiteralExpr); ok && key.Token == "go_proto_library" {
				mode = DisableProtoMode
				break
			}
		}
	}
	if mode == DefaultProtoMode || c.ProtoMode == mode || c.ShouldFix && mode == LegacyProtoMode {
		return c
	}
	modified := *c
	modified.ProtoMode = mode
	return &modified
}
