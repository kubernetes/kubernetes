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
	"flag"
	"fmt"
	"go/build"
	"path/filepath"
	"strings"

	"github.com/bazelbuild/bazel-gazelle/internal/rule"
	"github.com/bazelbuild/bazel-gazelle/internal/wspace"
)

// Config holds information about how Gazelle should run. This is based on
// command line arguments, directives, other hints in build files.
//
// A Config applies to a single directory. A Config is created for the
// repository root directory, then copied and modified for each subdirectory.
//
// Config itself contains only general information. Most configuration
// information is language-specific and is stored in Exts. This information
// is modified by extensions that implement Configurer.
type Config struct {
	// RepoRoot is the absolute, canonical path to the root directory of the
	// repository with all symlinks resolved.
	RepoRoot string

	// RepoName is the name of the repository.
	RepoName string

	// ReadBuildFilesDir is the absolute path to a directory where
	// build files should be read from instead of RepoRoot.
	ReadBuildFilesDir string

	// WriteBuildFilesDir is the absolute path to a directory where
	// build files should be written to instead of RepoRoot.
	WriteBuildFilesDir string

	// ValidBuildFileNames is a list of base names that are considered valid
	// build files. Some repositories may have files named "BUILD" that are not
	// used by Bazel and should be ignored. Must contain at least one string.
	ValidBuildFileNames []string

	// ShouldFix determines whether Gazelle attempts to remove and replace
	// usage of deprecated rules.
	ShouldFix bool

	// Exts is a set of configurable extensions. Generally, each language
	// has its own set of extensions, but other modules may provide their own
	// extensions as well. Values in here may be populated by command line
	// arguments, directives in build files, or other mechanisms.
	Exts map[string]interface{}
}

func New() *Config {
	return &Config{
		ValidBuildFileNames: DefaultValidBuildFileNames,
		Exts:                make(map[string]interface{}),
	}
}

// Clone creates a copy of the configuration for use in a subdirectory.
// Note that the Exts map is copied, but its contents are not.
// Configurer.Configure should do this, if needed.
func (c *Config) Clone() *Config {
	cc := *c
	cc.Exts = make(map[string]interface{})
	for k, v := range c.Exts {
		cc.Exts[k] = v
	}
	return &cc
}

var DefaultValidBuildFileNames = []string{"BUILD.bazel", "BUILD"}

func (c *Config) IsValidBuildFileName(name string) bool {
	for _, n := range c.ValidBuildFileNames {
		if name == n {
			return true
		}
	}
	return false
}

func (c *Config) DefaultBuildFileName() string {
	return c.ValidBuildFileNames[0]
}

// Configurer is the interface for language or library-specific configuration
// extensions. Most (ideally all) modifications to Config should happen
// via this interface.
type Configurer interface {
	// RegisterFlags registers command-line flags used by the extension. This
	// method is called once with the root configuration when Gazelle
	// starts. RegisterFlags may set an initial values in Config.Exts. When flags
	// are set, they should modify these values.
	RegisterFlags(fs *flag.FlagSet, cmd string, c *Config)

	// CheckFlags validates the configuration after command line flags are parsed.
	// This is called once with the root configuration when Gazelle starts.
	// CheckFlags may set default values in flags or make implied changes.
	CheckFlags(fs *flag.FlagSet, c *Config) error

	// KnownDirectives returns a list of directive keys that this Configurer can
	// interpret. Gazelle prints errors for directives that are not recoginized by
	// any Configurer.
	KnownDirectives() []string

	// Configure modifies the configuration using directives and other information
	// extracted from a build file. Configure is called in each directory.
	//
	// c is the configuration for the current directory. It starts out as a copy
	// of the configuration for the parent directory.
	//
	// rel is the slash-separated relative path from the repository root to
	// the current directory. It is "" for the root directory itself.
	//
	// f is the build file for the current directory or nil if there is no
	// existing build file.
	Configure(c *Config, rel string, f *rule.File)
}

// CommonConfigurer handles language-agnostic command-line flags and directives,
// i.e., those that apply to Config itself and not to Config.Exts.
type CommonConfigurer struct {
	repoRoot, buildFileNames, readBuildFilesDir, writeBuildFilesDir string
}

func (cc *CommonConfigurer) RegisterFlags(fs *flag.FlagSet, cmd string, c *Config) {
	fs.StringVar(&cc.repoRoot, "repo_root", "", "path to a directory which corresponds to go_prefix, otherwise gazelle searches for it.")
	fs.StringVar(&cc.buildFileNames, "build_file_name", strings.Join(DefaultValidBuildFileNames, ","), "comma-separated list of valid build file names.\nThe first element of the list is the name of output build files to generate.")
	fs.StringVar(&cc.readBuildFilesDir, "experimental_read_build_files_dir", "", "path to a directory where build files should be read from (instead of -repo_root)")
	fs.StringVar(&cc.writeBuildFilesDir, "experimental_write_build_files_dir", "", "path to a directory where build files should be written to (instead of -repo_root)")
}

func (cc *CommonConfigurer) CheckFlags(fs *flag.FlagSet, c *Config) error {
	var err error
	if cc.repoRoot == "" {
		cc.repoRoot, err = wspace.Find(".")
		if err != nil {
			return fmt.Errorf("-repo_root not specified, and WORKSPACE cannot be found: %v", err)
		}
	}
	c.RepoRoot, err = filepath.Abs(cc.repoRoot)
	if err != nil {
		return fmt.Errorf("%s: failed to find absolute path of repo root: %v", cc.repoRoot, err)
	}
	c.RepoRoot, err = filepath.EvalSymlinks(c.RepoRoot)
	if err != nil {
		return fmt.Errorf("%s: failed to resolve symlinks: %v", cc.repoRoot, err)
	}
	c.ValidBuildFileNames = strings.Split(cc.buildFileNames, ",")
	if cc.readBuildFilesDir != "" {
		c.ReadBuildFilesDir, err = filepath.Abs(cc.readBuildFilesDir)
		if err != nil {
			return fmt.Errorf("%s: failed to find absolute path of -read_build_files_dir: %v", cc.readBuildFilesDir, err)
		}
	}
	if cc.writeBuildFilesDir != "" {
		c.WriteBuildFilesDir, err = filepath.Abs(cc.writeBuildFilesDir)
		if err != nil {
			return fmt.Errorf("%s: failed to find absolute path of -write_build_files_dir: %v", cc.writeBuildFilesDir, err)
		}
	}

	return nil
}

func (cc *CommonConfigurer) KnownDirectives() []string {
	return []string{"build_file_name"}
}

func (cc *CommonConfigurer) Configure(c *Config, rel string, f *rule.File) {
	if f == nil {
		return
	}
	for _, d := range f.Directives {
		if d.Key == "build_file_name" {
			c.ValidBuildFileNames = strings.Split(d.Value, ",")
		}
	}
}

// CheckPrefix checks that a string may be used as a prefix. We forbid local
// (relative) imports and those beginning with "/". We allow the empty string,
// but generated rules must not have an empty importpath.
func CheckPrefix(prefix string) error {
	if strings.HasPrefix(prefix, "/") || build.IsLocalImport(prefix) {
		return fmt.Errorf("invalid prefix: %q", prefix)
	}
	return nil
}

// DependencyMode determines how imports of packages outside of the prefix
// are resolved.
type DependencyMode int

const (
	// ExternalMode indicates imports should be resolved to external dependencies
	// (declared in WORKSPACE).
	ExternalMode DependencyMode = iota

	// VendorMode indicates imports should be resolved to libraries in the
	// vendor directory.
	VendorMode
)

// DependencyModeFromString converts a string from the command line
// to a DependencyMode. Valid strings are "external", "vendor". An error will
// be returned for an invalid string.
func DependencyModeFromString(s string) (DependencyMode, error) {
	switch s {
	case "external":
		return ExternalMode, nil
	case "vendored":
		return VendorMode, nil
	default:
		return 0, fmt.Errorf("unrecognized dependency mode: %q", s)
	}
}

// ProtoMode determines how proto rules are generated.
type ProtoMode int

const (
	// DefaultProtoMode generates proto_library and new grpc_proto_library rules.
	// .pb.go files are excluded when there is a .proto file with a similar name.
	DefaultProtoMode ProtoMode = iota

	// DisableProtoMode ignores .proto files. .pb.go files are treated
	// as normal sources.
	DisableProtoMode

	// LegacyProtoMode generates filegroups for .proto files if .pb.go files
	// are present in the same directory.
	LegacyProtoMode
)

func ProtoModeFromString(s string) (ProtoMode, error) {
	switch s {
	case "default":
		return DefaultProtoMode, nil
	case "disable":
		return DisableProtoMode, nil
	case "legacy":
		return LegacyProtoMode, nil
	default:
		return 0, fmt.Errorf("unrecognized proto mode: %q", s)
	}
}
