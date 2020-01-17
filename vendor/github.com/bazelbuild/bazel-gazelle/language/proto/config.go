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
	"flag"
	"fmt"
	"log"
	"path"
	"strings"

	"github.com/bazelbuild/bazel-gazelle/config"
	"github.com/bazelbuild/bazel-gazelle/rule"
)

// ProtoConfig contains configuration values related to protos.
//
// This type is public because other languages need to generate rules based
// on protos, so this configuration may be relevant to them.
type ProtoConfig struct {
	// Mode determines how rules are generated for protos.
	Mode Mode

	// ModeExplicit indicates whether the proto mode was set explicitly.
	ModeExplicit bool

	// GoPrefix is the current Go prefix (the Go extension may set this in the
	// root directory only). Used to generate proto rule names in the root
	// directory when there are no proto files or the proto package name
	// can't be determined.
	// TODO(jayconrod): deprecate and remove Go-specific behavior.
	GoPrefix string

	// groupOption is an option name that Gazelle will use to group .proto
	// files into proto_library rules. If unset, the proto package name is used.
	groupOption string

	// stripImportPrefix The prefix to strip from the paths of the .proto files.
	// If set, Gazelle will apply this value to the strip_import_prefix attribute
	// within the proto_library_rule.
	stripImportPrefix string

	// importPrefix The prefix to add to the paths of the .proto files.
	// If set, Gazelle will apply this value to the import_prefix attribute
	// within the proto_library_rule.
	importPrefix string
}

// GetProtoConfig returns the proto language configuration. If the proto
// extension was not run, it will return nil.
func GetProtoConfig(c *config.Config) *ProtoConfig {
	pc := c.Exts[protoName]
	if pc == nil {
		return nil
	}
	return pc.(*ProtoConfig)
}

// Mode determines how proto rules are generated.
type Mode int

const (
	// DefaultMode generates proto_library rules. Other languages should generate
	// library rules based on these (e.g., go_proto_library) and should ignore
	// checked-in generated files (e.g., .pb.go files) when there is a .proto
	// file with a similar name.
	DefaultMode Mode = iota

	// DisableMode ignores .proto files and generates empty proto_library rules.
	// Checked-in generated files (e.g., .pb.go files) should be treated as
	// normal sources.
	DisableMode

	// DisableGlobalMode is similar to DisableMode, but it also prevents
	// the use of special cases in dependency resolution for well known types
	// and Google APIs.
	DisableGlobalMode

	// LegacyMode generates filegroups for .proto files if .pb.go files are
	// present in the same directory.
	LegacyMode

	// PackageMode generates a proto_library for each set of .proto files with
	// the same package name in each directory.
	PackageMode
)

func ModeFromString(s string) (Mode, error) {
	switch s {
	case "default":
		return DefaultMode, nil
	case "disable":
		return DisableMode, nil
	case "disable_global":
		return DisableGlobalMode, nil
	case "legacy":
		return LegacyMode, nil
	case "package":
		return PackageMode, nil
	default:
		return 0, fmt.Errorf("unrecognized proto mode: %q", s)
	}
}

func (m Mode) String() string {
	switch m {
	case DefaultMode:
		return "default"
	case DisableMode:
		return "disable"
	case DisableGlobalMode:
		return "disable_global"
	case LegacyMode:
		return "legacy"
	case PackageMode:
		return "package"
	default:
		log.Panicf("unknown mode %d", m)
		return ""
	}
}

func (m Mode) ShouldGenerateRules() bool {
	switch m {
	case DisableMode, DisableGlobalMode, LegacyMode:
		return false
	default:
		return true
	}
}

func (m Mode) ShouldIncludePregeneratedFiles() bool {
	switch m {
	case DisableMode, DisableGlobalMode, LegacyMode:
		return true
	default:
		return false
	}
}

func (m Mode) ShouldUseKnownImports() bool {
	return m != DisableGlobalMode
}

type modeFlag struct {
	mode *Mode
}

func (f *modeFlag) Set(value string) error {
	if mode, err := ModeFromString(value); err != nil {
		return err
	} else {
		*f.mode = mode
		return nil
	}
}

func (f *modeFlag) String() string {
	var mode Mode
	if f != nil && f.mode != nil {
		mode = *f.mode
	}
	return mode.String()
}

func (_ *protoLang) RegisterFlags(fs *flag.FlagSet, cmd string, c *config.Config) {
	pc := &ProtoConfig{}
	c.Exts[protoName] = pc

	// Note: the -proto flag does not set the ModeExplicit flag. We want to
	// be able to switch to DisableMode in vendor directories, even when
	// this is set for compatibility with older versions.
	fs.Var(&modeFlag{&pc.Mode}, "proto", "default: generates a proto_library rule for one package\n\tpackage: generates a proto_library rule for for each package\n\tdisable: does not touch proto rules\n\tdisable_global: does not touch proto rules and does not use special cases for protos in dependency resolution")
	fs.StringVar(&pc.groupOption, "proto_group", "", "option name used to group .proto files into proto_library rules")
	fs.StringVar(&pc.importPrefix, "proto_import_prefix", "", "When set, .proto source files in the srcs attribute of the rule are accessible at their path with this prefix appended on.")
}

func (_ *protoLang) CheckFlags(fs *flag.FlagSet, c *config.Config) error {
	return nil
}

func (_ *protoLang) KnownDirectives() []string {
	return []string{"proto", "proto_group", "proto_strip_import_prefix", "proto_import_prefix"}
}

func (_ *protoLang) Configure(c *config.Config, rel string, f *rule.File) {
	pc := &ProtoConfig{}
	*pc = *GetProtoConfig(c)
	c.Exts[protoName] = pc
	if f != nil {
		for _, d := range f.Directives {
			switch d.Key {
			case "proto":
				mode, err := ModeFromString(d.Value)
				if err != nil {
					log.Print(err)
					continue
				}
				pc.Mode = mode
				pc.ModeExplicit = true
			case "proto_group":
				pc.groupOption = d.Value
			case "proto_strip_import_prefix":
				pc.stripImportPrefix = d.Value
				if rel != "" {
					if err := checkStripImportPrefix(pc.stripImportPrefix, rel); err != nil {
						log.Print(err)
					}
				}
			case "proto_import_prefix":
				pc.importPrefix = d.Value
			}
		}
	}
	inferProtoMode(c, rel, f)
}

// inferProtoMode sets ProtoConfig.Mode based on the directory name and the
// contents of f. If the proto mode is set explicitly, this function does not
// change it. If this is a vendor directory, or go_proto_library is loaded from
// another file, proto rule generation is disabled.
//
// TODO(jayconrod): this logic is archaic, now that rules are generated by
// separate language extensions. Proto rule generation should be independent
// from Go.
func inferProtoMode(c *config.Config, rel string, f *rule.File) {
	pc := GetProtoConfig(c)
	if pc.Mode != DefaultMode || pc.ModeExplicit {
		return
	}
	if pc.GoPrefix == wellKnownTypesGoPrefix {
		pc.Mode = LegacyMode
		return
	}
	if path.Base(rel) == "vendor" {
		pc.Mode = DisableMode
		return
	}
	if f == nil {
		return
	}
	mode := DefaultMode
outer:
	for _, l := range f.Loads {
		name := l.Name()
		if name == "@io_bazel_rules_go//proto:def.bzl" {
			break
		}
		if name == "@io_bazel_rules_go//proto:go_proto_library.bzl" {
			mode = LegacyMode
			break
		}
		for _, sym := range l.Symbols() {
			if sym == "go_proto_library" {
				mode = DisableMode
				break outer
			}
		}
	}
	if mode == DefaultMode || pc.Mode == mode || c.ShouldFix && mode == LegacyMode {
		return
	}
	pc.Mode = mode
}

func checkStripImportPrefix(prefix, rel string) error {
	if !strings.HasPrefix(prefix, "/") || !strings.HasPrefix(rel, prefix[1:]) {
		return fmt.Errorf("invalid proto_strip_import_prefix %q at %s", prefix, rel)
	}
	return nil
}
