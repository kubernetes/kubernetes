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

import (
	"flag"
	"fmt"
	"go/build"
	"log"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/bazelbuild/bazel-gazelle/config"
	gzflag "github.com/bazelbuild/bazel-gazelle/flag"
	"github.com/bazelbuild/bazel-gazelle/language/proto"
	"github.com/bazelbuild/bazel-gazelle/rule"
	bzl "github.com/bazelbuild/buildtools/build"
)

// goConfig contains configuration values related to Go rules.
type goConfig struct {
	// genericTags is a set of tags that Gazelle considers to be true. Set with
	// -build_tags or # gazelle:build_tags. Some tags, like gc, are always on.
	genericTags map[string]bool

	// prefix is a prefix of an import path, used to generate importpath
	// attributes. Set with -go_prefix or # gazelle:prefix.
	prefix string

	// prefixRel is the package name of the directory where the prefix was set
	// ("" for the root directory).
	prefixRel string

	// prefixSet indicates whether the prefix was set explicitly. It is an error
	// to infer an importpath for a rule without setting the prefix.
	prefixSet bool

	// importMapPrefix is a prefix of a package path, used to generate importmap
	// attributes. Set with # gazelle:importmap_prefix.
	importMapPrefix string

	// importMapPrefixRel is the package name of the directory where importMapPrefix
	// was set ("" for the root directory).
	importMapPrefixRel string

	// depMode determines how imports that are not standard, indexed, or local
	// (under the current prefix) should be resolved.
	depMode dependencyMode

	// goProtoCompilers is the protocol buffers compiler(s) to use for go code.
	goProtoCompilers []string

	// goProtoCompilersSet indicates whether goProtoCompiler was set explicitly.
	goProtoCompilersSet bool

	// goGrpcCompilers is the gRPC compiler(s) to use for go code.
	goGrpcCompilers []string

	// goGrpcCompilersSet indicates whether goGrpcCompiler was set explicitly.
	goGrpcCompilersSet bool

	// goRepositoryMode is true if Gazelle was invoked by a go_repository rule.
	// In this mode, we won't go out to the network to resolve external deps.
	goRepositoryMode bool

	// By default, internal packages are only visible to its siblings.
	// goVisibility adds a list of packages the internal packages should be
	// visible to
	goVisibility []string

	// moduleMode is true if the current directory is intended to be built
	// as part of a module. Minimal module compatibility won't be supported
	// if this is true in the root directory. External dependencies may be
	// resolved differently (also depending on goRepositoryMode).
	moduleMode bool

	// submodules is a list of modules which have the current module's path
	// as a prefix of their own path. This affects visibility attributes
	// in internal packages.
	submodules []moduleRepo

	// buildExternalAttr, buildFileNamesAttr, buildFileGenerationAttr,
	// buildTagsAttr, buildFileProtoModeAttr, and buildExtraArgsAttr are
	// attributes for go_repository rules, set on the command line.
	buildExternalAttr, buildFileNamesAttr, buildFileGenerationAttr, buildTagsAttr, buildFileProtoModeAttr, buildExtraArgsAttr string
}

var (
	defaultGoProtoCompilers = []string{"@io_bazel_rules_go//proto:go_proto"}
	defaultGoGrpcCompilers  = []string{"@io_bazel_rules_go//proto:go_grpc"}
)

func newGoConfig() *goConfig {
	gc := &goConfig{
		goProtoCompilers: defaultGoProtoCompilers,
		goGrpcCompilers:  defaultGoGrpcCompilers,
	}
	gc.preprocessTags()
	return gc
}

func getGoConfig(c *config.Config) *goConfig {
	return c.Exts[goName].(*goConfig)
}

func (gc *goConfig) clone() *goConfig {
	gcCopy := *gc
	gcCopy.genericTags = make(map[string]bool)
	for k, v := range gc.genericTags {
		gcCopy.genericTags[k] = v
	}
	gcCopy.goProtoCompilers = gc.goProtoCompilers[:len(gc.goProtoCompilers):len(gc.goProtoCompilers)]
	gcCopy.goGrpcCompilers = gc.goGrpcCompilers[:len(gc.goGrpcCompilers):len(gc.goGrpcCompilers)]
	gcCopy.submodules = gc.submodules[:len(gc.submodules):len(gc.submodules)]
	return &gcCopy
}

// preprocessTags adds some tags which are on by default before they are
// used to match files.
func (gc *goConfig) preprocessTags() {
	if gc.genericTags == nil {
		gc.genericTags = make(map[string]bool)
	}
	gc.genericTags["gc"] = true
}

// setBuildTags sets genericTags by parsing as a comma separated list. An
// error will be returned for tags that wouldn't be recognized by "go build".
// preprocessTags should be called before this.
func (gc *goConfig) setBuildTags(tags string) error {
	if tags == "" {
		return nil
	}
	for _, t := range strings.Split(tags, ",") {
		if strings.HasPrefix(t, "!") {
			return fmt.Errorf("build tags can't be negated: %s", t)
		}
		gc.genericTags[t] = true
	}
	return nil
}

func getProtoMode(c *config.Config) proto.Mode {
	if pc := proto.GetProtoConfig(c); pc != nil {
		return pc.Mode
	} else {
		return proto.DisableGlobalMode
	}
}

// dependencyMode determines how imports of packages outside of the prefix
// are resolved.
type dependencyMode int

const (
	// externalMode indicates imports should be resolved to external dependencies
	// (declared in WORKSPACE).
	externalMode dependencyMode = iota

	// vendorMode indicates imports should be resolved to libraries in the
	// vendor directory.
	vendorMode
)

func (m dependencyMode) String() string {
	if m == externalMode {
		return "external"
	} else {
		return "vendored"
	}
}

type externalFlag struct {
	depMode *dependencyMode
}

func (f *externalFlag) Set(value string) error {
	switch value {
	case "external":
		*f.depMode = externalMode
	case "vendored":
		*f.depMode = vendorMode
	default:
		return fmt.Errorf("unrecognized dependency mode: %q", value)
	}
	return nil
}

func (f *externalFlag) String() string {
	if f == nil || f.depMode == nil {
		return "external"
	}
	return f.depMode.String()
}

type tagsFlag func(string) error

func (f tagsFlag) Set(value string) error {
	return f(value)
}

func (f tagsFlag) String() string {
	return ""
}

type moduleRepo struct {
	repoName, modulePath string
}

var validBuildExternalAttr = []string{"external", "vendored"}
var validBuildFileGenerationAttr = []string{"auto", "on", "off"}
var validBuildFileProtoModeAttr = []string{"default", "legacy", "disable", "disable_global", "package"}

func (*goLang) KnownDirectives() []string {
	return []string{
		"build_tags",
		"go_grpc_compilers",
		"go_proto_compilers",
		"go_visibility",
		"importmap_prefix",
		"prefix",
	}
}

func (*goLang) RegisterFlags(fs *flag.FlagSet, cmd string, c *config.Config) {
	gc := newGoConfig()
	switch cmd {
	case "fix", "update":
		fs.Var(
			tagsFlag(gc.setBuildTags),
			"build_tags",
			"comma-separated list of build tags. If not specified, Gazelle will not\n\tfilter sources with build constraints.")
		fs.Var(
			&gzflag.ExplicitFlag{Value: &gc.prefix, IsSet: &gc.prefixSet},
			"go_prefix",
			"prefix of import paths in the current workspace")
		fs.Var(
			&externalFlag{&gc.depMode},
			"external",
			"external: resolve external packages with go_repository\n\tvendored: resolve external packages as packages in vendor/")
		fs.Var(
			&gzflag.MultiFlag{Values: &gc.goProtoCompilers, IsSet: &gc.goProtoCompilersSet},
			"go_proto_compiler",
			"go_proto_library compiler to use (may be repeated)")
		fs.Var(
			&gzflag.MultiFlag{Values: &gc.goGrpcCompilers, IsSet: &gc.goGrpcCompilersSet},
			"go_grpc_compiler",
			"go_proto_library compiler to use for gRPC (may be repeated)")
		fs.BoolVar(
			&gc.goRepositoryMode,
			"go_repository_mode",
			false,
			"set when gazelle is invoked by go_repository")
		fs.BoolVar(
			&gc.moduleMode,
			"go_repository_module_mode",
			false,
			"set when gazelle is invoked by go_repository in module mode")

	case "update-repos":
		fs.Var(&gzflag.AllowedStringFlag{Value: &gc.buildExternalAttr, Allowed: validBuildExternalAttr},
			"build_external",
			"Sets the build_external attribute for the generated go_repository rule(s).")
		fs.StringVar(&gc.buildExtraArgsAttr,
			"build_extra_args",
			"",
			"Sets the build_extra_args attribute for the generated go_repository rule(s).")
		fs.Var(&gzflag.AllowedStringFlag{Value: &gc.buildFileGenerationAttr, Allowed: validBuildFileGenerationAttr},
			"build_file_generation",
			"Sets the build_file_generation attribute for the generated go_repository rule(s).")
		fs.StringVar(&gc.buildFileNamesAttr,
			"build_file_names",
			"",
			"Sets the build_file_name attribute for the generated go_repository rule(s).")
		fs.Var(&gzflag.AllowedStringFlag{Value: &gc.buildFileProtoModeAttr, Allowed: validBuildFileProtoModeAttr},
			"build_file_proto_mode",
			"Sets the build_file_proto_mode attribute for the generated go_repository rule(s).")
		fs.StringVar(&gc.buildTagsAttr,
			"build_tags",
			"",
			"Sets the build_tags attribute for the generated go_repository rule(s).")
	}
	c.Exts[goName] = gc
}

func (*goLang) CheckFlags(fs *flag.FlagSet, c *config.Config) error {
	// The base of the -go_prefix flag may be used to generate proto_library
	// rule names when there are no .proto sources (empty rules to be deleted)
	// or when the package name can't be determined.
	// TODO(jayconrod): deprecate and remove this behavior.
	gc := getGoConfig(c)
	if pc := proto.GetProtoConfig(c); pc != nil {
		pc.GoPrefix = gc.prefix
	}

	// List modules that may refer to internal packages in this module.
	for _, r := range c.Repos {
		if r.Kind() != "go_repository" {
			continue
		}
		modulePath := r.AttrString("importpath")
		if !strings.HasPrefix(modulePath, gc.prefix+"/") {
			continue
		}
		m := moduleRepo{
			repoName:   r.Name(),
			modulePath: modulePath,
		}
		gc.submodules = append(gc.submodules, m)
	}

	return nil
}

func (*goLang) Configure(c *config.Config, rel string, f *rule.File) {
	var gc *goConfig
	if raw, ok := c.Exts[goName]; !ok {
		gc = newGoConfig()
	} else {
		gc = raw.(*goConfig).clone()
	}
	c.Exts[goName] = gc

	if !gc.moduleMode {
		st, err := os.Stat(filepath.Join(c.RepoRoot, filepath.FromSlash(rel), "go.mod"))
		if err == nil && !st.IsDir() {
			gc.moduleMode = true
		}
	}

	if path.Base(rel) == "vendor" {
		gc.importMapPrefix = inferImportPath(gc, rel)
		gc.importMapPrefixRel = rel
		gc.prefix = ""
		gc.prefixRel = rel
	}

	if f != nil {
		setPrefix := func(prefix string) {
			if err := checkPrefix(prefix); err != nil {
				log.Print(err)
				return
			}
			gc.prefix = prefix
			gc.prefixSet = true
			gc.prefixRel = rel
		}
		for _, d := range f.Directives {
			switch d.Key {
			case "build_tags":
				if err := gc.setBuildTags(d.Value); err != nil {
					log.Print(err)
					continue
				}
				gc.preprocessTags()
				gc.setBuildTags(d.Value)

			case "go_grpc_compilers":
				// Special syntax (empty value) to reset directive.
				if d.Value == "" {
					gc.goGrpcCompilersSet = false
					gc.goGrpcCompilers = defaultGoGrpcCompilers
				} else {
					gc.goGrpcCompilersSet = true
					gc.goGrpcCompilers = splitValue(d.Value)
				}

			case "go_proto_compilers":
				// Special syntax (empty value) to reset directive.
				if d.Value == "" {
					gc.goProtoCompilersSet = false
					gc.goProtoCompilers = defaultGoProtoCompilers
				} else {
					gc.goProtoCompilersSet = true
					gc.goProtoCompilers = splitValue(d.Value)
				}

			case "go_visibility":
				gc.goVisibility = append(gc.goVisibility, strings.TrimSpace(d.Value))

			case "importmap_prefix":
				gc.importMapPrefix = d.Value
				gc.importMapPrefixRel = rel

			case "prefix":
				setPrefix(d.Value)
			}
		}
		if !gc.prefixSet {
			for _, r := range f.Rules {
				switch r.Kind() {
				case "go_prefix":
					args := r.Args()
					if len(args) != 1 {
						continue
					}
					s, ok := args[0].(*bzl.StringExpr)
					if !ok {
						continue
					}
					setPrefix(s.Value)

				case "gazelle":
					if prefix := r.AttrString("prefix"); prefix != "" {
						setPrefix(prefix)
					}
				}
			}
		}
	}
}

// checkPrefix checks that a string may be used as a prefix. We forbid local
// (relative) imports and those beginning with "/". We allow the empty string,
// but generated rules must not have an empty importpath.
func checkPrefix(prefix string) error {
	if strings.HasPrefix(prefix, "/") || build.IsLocalImport(prefix) {
		return fmt.Errorf("invalid prefix: %q", prefix)
	}
	return nil
}

// splitDirective splits a comma-separated directive value into its component
// parts, trimming each of any whitespace characters.
func splitValue(value string) []string {
	parts := strings.Split(value, ",")
	values := make([]string, 0, len(parts))
	for _, part := range parts {
		values = append(values, strings.TrimSpace(part))
	}
	return values
}
