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
	"fmt"
	"go/build"
	"log"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/bazelbuild/bazel-gazelle/internal/config"
	"github.com/bazelbuild/bazel-gazelle/internal/language/proto"
	"github.com/bazelbuild/bazel-gazelle/internal/pathtools"
	"github.com/bazelbuild/bazel-gazelle/internal/rule"
)

func (gl *goLang) GenerateRules(c *config.Config, dir, rel string, f *rule.File, subdirs, regularFiles, genFiles []string, otherEmpty, otherGen []*rule.Rule) (empty, gen []*rule.Rule) {
	// Extract information about proto files. We need this to exclude .pb.go
	// files and generate go_proto_library rules.
	gc := getGoConfig(c)
	pc := proto.GetProtoConfig(c)
	var protoRuleNames []string
	protoPackages := make(map[string]proto.Package)
	protoFileInfo := make(map[string]proto.FileInfo)
	for _, r := range otherGen {
		if r.Kind() != "proto_library" {
			continue
		}
		pkg := r.PrivateAttr(proto.PackageKey).(proto.Package)
		protoPackages[r.Name()] = pkg
		for name, info := range pkg.Files {
			protoFileInfo[name] = info
		}
		protoRuleNames = append(protoRuleNames, r.Name())
	}
	sort.Strings(protoRuleNames)
	var emptyProtoRuleNames []string
	for _, r := range otherEmpty {
		if r.Kind() == "proto_library" {
			emptyProtoRuleNames = append(emptyProtoRuleNames, r.Name())
		}
	}

	// If proto rule generation is enabled, exclude .pb.go files that correspond
	// to any .proto files present.
	if !pc.Mode.ShouldIncludePregeneratedFiles() {
		keep := func(f string) bool {
			if strings.HasSuffix(f, ".pb.go") {
				_, ok := protoFileInfo[strings.TrimSuffix(f, ".pb.go")+".proto"]
				return !ok
			}
			return true
		}
		filterFiles(&regularFiles, keep)
		filterFiles(&genFiles, keep)
	}

	// Split regular files into files which can determine the package name and
	// import path and other files.
	var goFiles, otherFiles []string
	for _, f := range regularFiles {
		if strings.HasSuffix(f, ".go") {
			goFiles = append(goFiles, f)
		} else {
			otherFiles = append(otherFiles, f)
		}
	}

	// Look for a subdirectory named testdata. Only treat it as data if it does
	// not contain a buildable package.
	var hasTestdata bool
	for _, sub := range subdirs {
		if sub == "testdata" {
			hasTestdata = !gl.goPkgRels[path.Join(rel, "testdata")]
			break
		}
	}

	// Build a set of packages from files in this directory.
	goPackageMap, goFilesWithUnknownPackage := buildPackages(c, dir, rel, goFiles, hasTestdata)

	// Select a package to generate rules for. If there is no package, create
	// an empty package so we can generate empty rules.
	var protoName string
	pkg, err := selectPackage(c, dir, goPackageMap)
	if err != nil {
		if _, ok := err.(*build.NoGoError); ok {
			if len(protoPackages) == 1 {
				for name, ppkg := range protoPackages {
					pkg = &goPackage{
						name:       goProtoPackageName(ppkg),
						importPath: goProtoImportPath(gc, ppkg, rel),
						proto:      protoTargetFromProtoPackage(name, ppkg),
					}
					protoName = name
					break
				}
			} else {
				pkg = emptyPackage(c, dir, rel)
			}
		} else {
			log.Print(err)
		}
	}

	// Try to link the selected package with a proto package.
	if pkg != nil {
		if pkg.importPath == "" {
			if err := pkg.inferImportPath(c); err != nil && pkg.firstGoFile() != "" {
				inferImportPathErrorOnce.Do(func() { log.Print(err) })
			}
		}
		for _, name := range protoRuleNames {
			ppkg := protoPackages[name]
			if pkg.importPath == goProtoImportPath(gc, ppkg, rel) {
				protoName = name
				pkg.proto = protoTargetFromProtoPackage(name, ppkg)
				break
			}
		}
	}

	// Generate rules for proto packages. These should come before the other
	// Go rules.
	g := newGenerator(c, f, rel)
	var rules []*rule.Rule
	var protoEmbed string
	for _, name := range protoRuleNames {
		ppkg := protoPackages[name]
		var rs []*rule.Rule
		if name == protoName {
			protoEmbed, rs = g.generateProto(pc.Mode, pkg.proto, pkg.importPath)
		} else {
			target := protoTargetFromProtoPackage(name, ppkg)
			importPath := goProtoImportPath(gc, ppkg, rel)
			_, rs = g.generateProto(pc.Mode, target, importPath)
		}
		rules = append(rules, rs...)
	}
	for _, name := range emptyProtoRuleNames {
		goProtoName := strings.TrimSuffix(name, "_proto") + "_go_proto"
		empty = append(empty, rule.NewRule("go_proto_library", goProtoName))
	}
	if pkg != nil && pc.Mode == proto.PackageMode && pkg.firstGoFile() == "" {
		// In proto package mode, don't generate a go_library embedding a
		// go_proto_library unless there are actually go files.
		protoEmbed = ""
	}

	// Complete the Go package and generate rules for that.
	if pkg != nil {
		// Add files with unknown packages. This happens when there are parse
		// or I/O errors. We should keep the file in the srcs list and let the
		// compiler deal with the error.
		cgo := pkg.haveCgo()
		for _, info := range goFilesWithUnknownPackage {
			if err := pkg.addFile(c, info, cgo); err != nil {
				log.Print(err)
			}
		}

		// Process the other static files.
		for _, file := range otherFiles {
			info := otherFileInfo(filepath.Join(dir, file))
			if err := pkg.addFile(c, info, cgo); err != nil {
				log.Print(err)
			}
		}

		// Process generated files. Note that generated files may have the same names
		// as static files. Bazel will use the generated files, but we will look at
		// the content of static files, assuming they will be the same.
		regularFileSet := make(map[string]bool)
		for _, f := range regularFiles {
			regularFileSet[f] = true
		}
		for _, f := range genFiles {
			if regularFileSet[f] {
				continue
			}
			info := fileNameInfo(filepath.Join(dir, f))
			if err := pkg.addFile(c, info, cgo); err != nil {
				log.Print(err)
			}
		}

		// Generate Go rules.
		if protoName == "" {
			// Empty proto rules for deletion.
			_, rs := g.generateProto(pc.Mode, pkg.proto, pkg.importPath)
			rules = append(rules, rs...)
		}
		lib := g.generateLib(pkg, protoEmbed)
		var libName string
		if !lib.IsEmpty(goKinds[lib.Kind()]) {
			libName = lib.Name()
		}
		rules = append(rules, lib)
		rules = append(rules,
			g.generateBin(pkg, libName),
			g.generateTest(pkg, libName))
	}

	for _, r := range rules {
		if r.IsEmpty(goKinds[r.Kind()]) {
			empty = append(empty, r)
		} else {
			gen = append(gen, r)
		}
	}

	if f != nil || len(gen) > 0 {
		gl.goPkgRels[rel] = true
	} else {
		for _, sub := range subdirs {
			if gl.goPkgRels[path.Join(rel, sub)] {
				gl.goPkgRels[rel] = true
				break
			}
		}
	}

	return empty, gen
}

func filterFiles(files *[]string, pred func(string) bool) {
	w := 0
	for r := 0; r < len(*files); r++ {
		f := (*files)[r]
		if pred(f) {
			(*files)[w] = f
			w++
		}
	}
	*files = (*files)[:w]
}

func buildPackages(c *config.Config, dir, rel string, goFiles []string, hasTestdata bool) (packageMap map[string]*goPackage, goFilesWithUnknownPackage []fileInfo) {
	// Process .go and .proto files first, since these determine the package name.
	packageMap = make(map[string]*goPackage)
	for _, f := range goFiles {
		path := filepath.Join(dir, f)
		info := goFileInfo(path, rel)
		if info.packageName == "" {
			goFilesWithUnknownPackage = append(goFilesWithUnknownPackage, info)
			continue
		}
		if info.packageName == "documentation" {
			// go/build ignores this package
			continue
		}

		if _, ok := packageMap[info.packageName]; !ok {
			packageMap[info.packageName] = &goPackage{
				name:        info.packageName,
				dir:         dir,
				rel:         rel,
				hasTestdata: hasTestdata,
			}
		}
		if err := packageMap[info.packageName].addFile(c, info, false); err != nil {
			log.Print(err)
		}
	}
	return packageMap, goFilesWithUnknownPackage
}

var inferImportPathErrorOnce sync.Once

// selectPackages selects one Go packages out of the buildable packages found
// in a directory. If multiple packages are found, it returns the package
// whose name matches the directory if such a package exists.
func selectPackage(c *config.Config, dir string, packageMap map[string]*goPackage) (*goPackage, error) {
	buildablePackages := make(map[string]*goPackage)
	for name, pkg := range packageMap {
		if pkg.isBuildable(c) {
			buildablePackages[name] = pkg
		}
	}

	if len(buildablePackages) == 0 {
		return nil, &build.NoGoError{Dir: dir}
	}

	if len(buildablePackages) == 1 {
		for _, pkg := range buildablePackages {
			return pkg, nil
		}
	}

	if pkg, ok := buildablePackages[defaultPackageName(c, dir)]; ok {
		return pkg, nil
	}

	err := &build.MultiplePackageError{Dir: dir}
	for name, pkg := range buildablePackages {
		// Add the first file for each package for the error message.
		// Error() method expects these lists to be the same length. File
		// lists must be non-empty. These lists are only created by
		// buildPackage for packages with .go files present.
		err.Packages = append(err.Packages, name)
		err.Files = append(err.Files, pkg.firstGoFile())
	}
	return nil, err
}

func emptyPackage(c *config.Config, dir, rel string) *goPackage {
	pkg := &goPackage{
		name: defaultPackageName(c, dir),
		dir:  dir,
		rel:  rel,
	}
	pkg.inferImportPath(c)
	return pkg
}

func defaultPackageName(c *config.Config, rel string) string {
	gc := getGoConfig(c)
	return pathtools.RelBaseName(rel, gc.prefix, "")
}

// hasDefaultVisibility returns whether oldFile contains a "package" rule with
// a "default_visibility" attribute. Rules generated by Gazelle should not
// have their own visibility attributes if this is the case.
func hasDefaultVisibility(oldFile *rule.File) bool {
	for _, r := range oldFile.Rules {
		if r.Kind() == "package" && r.Attr("default_visibility") != nil {
			return true
		}
	}
	return false
}

// checkInternalVisibility overrides the given visibility if the package is
// internal.
func checkInternalVisibility(rel, visibility string) string {
	if i := strings.LastIndex(rel, "/internal/"); i >= 0 {
		visibility = fmt.Sprintf("//%s:__subpackages__", rel[:i])
	} else if strings.HasPrefix(rel, "internal/") {
		visibility = "//:__subpackages__"
	}
	return visibility
}

type generator struct {
	c                   *config.Config
	rel                 string
	shouldSetVisibility bool
}

func newGenerator(c *config.Config, f *rule.File, rel string) *generator {
	shouldSetVisibility := f == nil || !hasDefaultVisibility(f)
	return &generator{c: c, rel: rel, shouldSetVisibility: shouldSetVisibility}
}

func (g *generator) generateProto(mode proto.Mode, target protoTarget, importPath string) (string, []*rule.Rule) {
	if !mode.ShouldGenerateRules() && mode != proto.LegacyMode {
		// Don't create or delete proto rules in this mode. Any existing rules
		// are likely hand-written.
		return "", nil
	}

	filegroupName := config.DefaultProtosName
	protoName := target.name
	if protoName == "" {
		importPath := inferImportPath(getGoConfig(g.c), g.rel)
		protoName = proto.RuleName(importPath)
	}
	goProtoName := strings.TrimSuffix(protoName, "_proto") + "_go_proto"
	visibility := []string{checkInternalVisibility(g.rel, "//visibility:public")}

	if mode == proto.LegacyMode {
		filegroup := rule.NewRule("filegroup", filegroupName)
		if target.sources.isEmpty() {
			return "", []*rule.Rule{filegroup}
		}
		filegroup.SetAttr("srcs", target.sources.build())
		if g.shouldSetVisibility {
			filegroup.SetAttr("visibility", visibility)
		}
		return "", []*rule.Rule{filegroup}
	}

	if target.sources.isEmpty() {
		return "", []*rule.Rule{
			rule.NewRule("filegroup", filegroupName),
			rule.NewRule("go_proto_library", goProtoName),
		}
	}

	goProtoLibrary := rule.NewRule("go_proto_library", goProtoName)
	goProtoLibrary.SetAttr("proto", ":"+protoName)
	g.setImportAttrs(goProtoLibrary, importPath)
	if target.hasServices {
		goProtoLibrary.SetAttr("compilers", []string{"@io_bazel_rules_go//proto:go_grpc"})
	}
	if g.shouldSetVisibility {
		goProtoLibrary.SetAttr("visibility", visibility)
	}
	goProtoLibrary.SetPrivateAttr(config.GazelleImportsKey, target.imports.build())
	return goProtoName, []*rule.Rule{goProtoLibrary}
}

func (g *generator) generateLib(pkg *goPackage, embed string) *rule.Rule {
	goLibrary := rule.NewRule("go_library", config.DefaultLibName)
	if !pkg.library.sources.hasGo() && embed == "" {
		return goLibrary // empty
	}
	var visibility string
	if pkg.isCommand() {
		// Libraries made for a go_binary should not be exposed to the public.
		visibility = "//visibility:private"
	} else {
		visibility = checkInternalVisibility(pkg.rel, "//visibility:public")
	}
	g.setCommonAttrs(goLibrary, pkg.rel, visibility, pkg.library, embed)
	g.setImportAttrs(goLibrary, pkg.importPath)
	return goLibrary
}

func (g *generator) generateBin(pkg *goPackage, library string) *rule.Rule {
	name := pathtools.RelBaseName(pkg.rel, getGoConfig(g.c).prefix, g.c.RepoRoot)
	goBinary := rule.NewRule("go_binary", name)
	if !pkg.isCommand() || pkg.binary.sources.isEmpty() && library == "" {
		return goBinary // empty
	}
	visibility := checkInternalVisibility(pkg.rel, "//visibility:public")
	g.setCommonAttrs(goBinary, pkg.rel, visibility, pkg.binary, library)
	return goBinary
}

func (g *generator) generateTest(pkg *goPackage, library string) *rule.Rule {
	goTest := rule.NewRule("go_test", config.DefaultTestName)
	if !pkg.test.sources.hasGo() {
		return goTest // empty
	}
	g.setCommonAttrs(goTest, pkg.rel, "", pkg.test, library)
	if pkg.hasTestdata {
		goTest.SetAttr("data", rule.GlobValue{Patterns: []string{"testdata/**"}})
	}
	return goTest
}

func (g *generator) setCommonAttrs(r *rule.Rule, pkgRel, visibility string, target goTarget, embed string) {
	if !target.sources.isEmpty() {
		r.SetAttr("srcs", target.sources.buildFlat())
	}
	if target.cgo {
		r.SetAttr("cgo", true)
	}
	if !target.clinkopts.isEmpty() {
		r.SetAttr("clinkopts", g.options(target.clinkopts.build(), pkgRel))
	}
	if !target.copts.isEmpty() {
		r.SetAttr("copts", g.options(target.copts.build(), pkgRel))
	}
	if g.shouldSetVisibility && visibility != "" {
		r.SetAttr("visibility", []string{visibility})
	}
	if embed != "" {
		r.SetAttr("embed", []string{":" + embed})
	}
	r.SetPrivateAttr(config.GazelleImportsKey, target.imports.build())
}

func (g *generator) setImportAttrs(r *rule.Rule, importPath string) {
	r.SetAttr("importpath", importPath)
	goConf := getGoConfig(g.c)
	if goConf.importMapPrefix != "" {
		fromPrefixRel := pathtools.TrimPrefix(g.rel, goConf.importMapPrefixRel)
		importMap := path.Join(goConf.importMapPrefix, fromPrefixRel)
		if importMap != importPath {
			r.SetAttr("importmap", importMap)
		}
	}
}

var (
	// shortOptPrefixes are strings that come at the beginning of an option
	// argument that includes a path, e.g., -Ifoo/bar.
	shortOptPrefixes = []string{"-I", "-L", "-F"}

	// longOptPrefixes are separate arguments that come before a path argument,
	// e.g., -iquote foo/bar.
	longOptPrefixes = []string{"-I", "-L", "-F", "-iquote", "-isystem"}
)

// options transforms package-relative paths in cgo options into repository-
// root-relative paths that Bazel can understand. For example, if a cgo file
// in //foo declares an include flag in its copts: "-Ibar", this method
// will transform that flag into "-Ifoo/bar".
func (g *generator) options(opts rule.PlatformStrings, pkgRel string) rule.PlatformStrings {
	fixPath := func(opt string) string {
		if strings.HasPrefix(opt, "/") {
			return opt
		}
		return path.Clean(path.Join(pkgRel, opt))
	}

	fixGroups := func(groups []string) ([]string, error) {
		fixedGroups := make([]string, len(groups))
		for i, group := range groups {
			opts := strings.Split(group, optSeparator)
			fixedOpts := make([]string, len(opts))
			isPath := false
			for j, opt := range opts {
				if isPath {
					opt = fixPath(opt)
					isPath = false
					goto next
				}

				for _, short := range shortOptPrefixes {
					if strings.HasPrefix(opt, short) && len(opt) > len(short) {
						opt = short + fixPath(opt[len(short):])
						goto next
					}
				}

				for _, long := range longOptPrefixes {
					if opt == long {
						isPath = true
						goto next
					}
				}

			next:
				fixedOpts[j] = escapeOption(opt)
			}
			fixedGroups[i] = strings.Join(fixedOpts, " ")
		}

		return fixedGroups, nil
	}

	opts, errs := opts.MapSlice(fixGroups)
	if errs != nil {
		log.Panicf("unexpected error when transforming options with pkg %q: %v", pkgRel, errs)
	}
	return opts
}

func escapeOption(opt string) string {
	return strings.NewReplacer(
		`\`, `\\`,
		`'`, `\'`,
		`"`, `\"`,
		` `, `\ `,
		"\t", "\\\t",
		"\n", "\\\n",
		"\r", "\\\r",
	).Replace(opt)
}
