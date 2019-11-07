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

// Package config provides extensible configuration for Gazelle libraries.
//
// Packages may define Configurers which add support for new command-line
// options and directive comments in build files. Note that the
// language.Language interface embeds Configurer, so each language extension
// has the opportunity
//
// When Gazelle walks the directory trees in a repository, it calls the
// Configure method of each Configurer to produce a Config object.
// Config objects are passed as arguments to most functions in Gazelle, so
// this mechanism may be used to control many aspects of Gazelle's behavior.
package config

import (
	"flag"
	"fmt"
	"log"
	"path/filepath"
	"strings"

	"github.com/bazelbuild/bazel-gazelle/internal/wspace"
	"github.com/bazelbuild/bazel-gazelle/rule"
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

	// IndexLibraries determines whether Gazelle should build an index of
	// libraries in the workspace for dependency resolution
	IndexLibraries bool

	// KindMap maps from a kind name to its replacement. It provides a way for
	// users to customize the kind of rules created by Gazelle, via
	// # gazelle:map_kind.
	KindMap map[string]MappedKind

	// Repos is a list of repository rules declared in the main WORKSPACE file
	// or in macros called by the main WORKSPACE file. This may affect rule
	// generation and dependency resolution.
	Repos []*rule.Rule

	// Exts is a set of configurable extensions. Generally, each language
	// has its own set of extensions, but other modules may provide their own
	// extensions as well. Values in here may be populated by command line
	// arguments, directives in build files, or other mechanisms.
	Exts map[string]interface{}
}

// MappedKind describes a replacement to use for a built-in kind.
type MappedKind struct {
	FromKind, KindName, KindLoad string
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
	cc.KindMap = make(map[string]MappedKind)
	for k, v := range c.KindMap {
		cc.KindMap[k] = v
	}
	return &cc
}

var DefaultValidBuildFileNames = []string{"BUILD.bazel", "BUILD"}

// IsValidBuildFileName returns true if a file with the given base name
// should be treated as a build file.
func (c *Config) IsValidBuildFileName(name string) bool {
	for _, n := range c.ValidBuildFileNames {
		if name == n {
			return true
		}
	}
	return false
}

// DefaultBuildFileName returns the base name used to create new build files.
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
	indexLibraries                                                  bool
}

func (cc *CommonConfigurer) RegisterFlags(fs *flag.FlagSet, cmd string, c *Config) {
	fs.StringVar(&cc.repoRoot, "repo_root", "", "path to a directory which corresponds to go_prefix, otherwise gazelle searches for it.")
	fs.StringVar(&cc.buildFileNames, "build_file_name", strings.Join(DefaultValidBuildFileNames, ","), "comma-separated list of valid build file names.\nThe first element of the list is the name of output build files to generate.")
	fs.BoolVar(&cc.indexLibraries, "index", true, "when true, gazelle will build an index of libraries in the workspace for dependency resolution")
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
	c.IndexLibraries = cc.indexLibraries
	return nil
}

func (cc *CommonConfigurer) KnownDirectives() []string {
	return []string{"build_file_name", "map_kind"}
}

func (cc *CommonConfigurer) Configure(c *Config, rel string, f *rule.File) {
	if f == nil {
		return
	}
	for _, d := range f.Directives {
		switch d.Key {
		case "build_file_name":
			c.ValidBuildFileNames = strings.Split(d.Value, ",")

		case "map_kind":
			vals := strings.Fields(d.Value)
			if len(vals) != 3 {
				log.Printf("expected three arguments (gazelle:map_kind from_kind to_kind load_file), got %v", vals)
				continue
			}
			if c.KindMap == nil {
				c.KindMap = make(map[string]MappedKind)
			}
			c.KindMap[vals[0]] = MappedKind{
				FromKind: vals[0],
				KindName: vals[1],
				KindLoad: vals[2],
			}
		}
	}
}
