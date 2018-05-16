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

package packages

import (
	"fmt"
	"log"
	"path"
	"sort"
	"strings"

	"github.com/bazelbuild/bazel-gazelle/internal/config"
	"github.com/bazelbuild/bazel-gazelle/internal/pathtools"
)

// Package contains metadata about a Go package extracted from a directory.
// It fills a similar role to go/build.Package, but it separates files by
// target instead of by type, and it supports multiple platforms.
type Package struct {
	// Name is the symbol found in package declarations of the .go files in
	// the package. It does not include the "_test" suffix from external tests.
	Name string

	// Dir is an absolute path to the directory that contains the package.
	Dir string

	// Rel is the relative path to the package directory from the repository
	// root. If the directory is the repository root itself, Rel is empty.
	// Components in Rel are separated with slashes.
	Rel string

	// ImportPath is the string used to import this package in Go.
	ImportPath string

	Library, Binary, Test GoTarget
	Proto                 ProtoTarget

	HasTestdata bool
}

// GoTarget contains metadata about a buildable Go target in a package.
type GoTarget struct {
	Sources, Imports PlatformStrings
	COpts, CLinkOpts PlatformStrings
	Cgo              bool
}

// ProtoTarget contains metadata about proto files in a package.
type ProtoTarget struct {
	Sources, Imports PlatformStrings
	HasServices      bool

	// HasPbGo indicates whether unexcluded .pb.go files are present in the
	// same package. They will not be in this target's sources.
	HasPbGo bool
}

// PlatformStrings contains a set of strings associated with a buildable
// Go target in a package. This is used to store source file names,
// import paths, and flags.
//
// Strings are stored in four sets: generic strings, OS-specific strings,
// arch-specific strings, and OS-and-arch-specific strings. A string may not
// be duplicated within a list or across sets; however, a string may appear
// in more than one list within a set (e.g., in "linux" and "windows" within
// the OS set). Strings within each list should be sorted, though this may
// not be relied upon.
type PlatformStrings struct {
	// Generic is a list of strings not specific to any platform.
	Generic []string

	// OS is a map from OS name (anything in config.KnownOSs) to
	// OS-specific strings.
	OS map[string][]string

	// Arch is a map from architecture name (anything in config.KnownArchs) to
	// architecture-specific strings.
	Arch map[string][]string

	// Platform is a map from platforms to OS and architecture-specific strings.
	Platform map[config.Platform][]string
}

// IsCommand returns true if the package name is "main".
func (p *Package) IsCommand() bool {
	return p.Name == "main"
}

// EmptyPackage returns an empty package. The package name and import path
// are inferred from the directory name and configuration. This is useful
// for deleting rules in directories which no longer have source files.
func EmptyPackage(c *config.Config, dir, rel string) *Package {
	packageName := pathtools.RelBaseName(rel, c.GoPrefix, c.RepoRoot)
	pb := packageBuilder{
		name: packageName,
		dir:  dir,
		rel:  rel,
	}
	pb.inferImportPath(c)
	return pb.build()
}

func (t *GoTarget) HasGo() bool {
	return t.Sources.HasGo()
}

func (t *ProtoTarget) HasProto() bool {
	return !t.Sources.IsEmpty()
}

func (ps *PlatformStrings) HasGo() bool {
	return ps.firstGoFile() != ""
}

func (ps *PlatformStrings) IsEmpty() bool {
	return len(ps.Generic) == 0 && len(ps.OS) == 0 && len(ps.Arch) == 0 && len(ps.Platform) == 0
}

func (ps *PlatformStrings) Flat() []string {
	unique := make(map[string]struct{})
	for _, s := range ps.Generic {
		unique[s] = struct{}{}
	}
	for _, ss := range ps.OS {
		for _, s := range ss {
			unique[s] = struct{}{}
		}
	}
	for _, ss := range ps.Arch {
		for _, s := range ss {
			unique[s] = struct{}{}
		}
	}
	for _, ss := range ps.Platform {
		for _, s := range ss {
			unique[s] = struct{}{}
		}
	}
	flat := make([]string, 0, len(unique))
	for s := range unique {
		flat = append(flat, s)
	}
	sort.Strings(flat)
	return flat
}

func (ps *PlatformStrings) firstGoFile() string {
	for _, f := range ps.Generic {
		if strings.HasSuffix(f, ".go") {
			return f
		}
	}
	for _, fs := range ps.OS {
		for _, f := range fs {
			if strings.HasSuffix(f, ".go") {
				return f
			}
		}
	}
	for _, fs := range ps.Arch {
		for _, f := range fs {
			if strings.HasSuffix(f, ".go") {
				return f
			}
		}
	}
	for _, fs := range ps.Platform {
		for _, f := range fs {
			if strings.HasSuffix(f, ".go") {
				return f
			}
		}
	}
	return ""
}

type packageBuilder struct {
	name, dir, rel             string
	library, binary, test      goTargetBuilder
	proto                      protoTargetBuilder
	hasTestdata                bool
	importPath, importPathFile string
}

type goTargetBuilder struct {
	sources, imports, copts, clinkopts platformStringsBuilder
	cgo                                bool
}

type protoTargetBuilder struct {
	sources, imports     platformStringsBuilder
	hasServices, hasPbGo bool
}

type platformStringsBuilder struct {
	strs map[string]platformStringInfo
}

type platformStringInfo struct {
	set       platformStringSet
	oss       map[string]bool
	archs     map[string]bool
	platforms map[config.Platform]bool
}

type platformStringSet int

const (
	genericSet platformStringSet = iota
	osSet
	archSet
	platformSet
)

// addFile adds the file described by "info" to a target in the package "p" if
// the file is buildable.
//
// "cgo" tells whether any ".go" file in the package contains cgo code. This
// affects whether C files are added to targets.
//
// An error is returned if a file is buildable but invalid (for example, a
// test .go file containing cgo code). Files that are not buildable will not
// be added to any target (for example, .txt files).
func (pb *packageBuilder) addFile(c *config.Config, info fileInfo, cgo bool) error {
	switch {
	case info.category == ignoredExt || info.category == unsupportedExt ||
		!cgo && (info.category == cExt || info.category == csExt) ||
		c.ProtoMode == config.DisableProtoMode && info.category == protoExt:
		return nil
	case info.isTest:
		if info.isCgo {
			return fmt.Errorf("%s: use of cgo in test not supported", info.path)
		}
		pb.test.addFile(c, info)
	case info.category == protoExt:
		pb.proto.addFile(c, info)
	default:
		pb.library.addFile(c, info)
	}
	if strings.HasSuffix(info.name, ".pb.go") {
		pb.proto.hasPbGo = true
	}

	if info.importPath != "" {
		if pb.importPath == "" {
			pb.importPath = info.importPath
			pb.importPathFile = info.path
		} else if pb.importPath != info.importPath {
			return fmt.Errorf("found import comments %q (%s) and %q (%s)", pb.importPath, pb.importPathFile, info.importPath, info.path)
		}
	}

	return nil
}

// isBuildable returns true if anything in the package is buildable.
// This is true if the package has Go code that satisfies build constraints
// on any platform or has proto files not in legacy mode.
func (pb *packageBuilder) isBuildable(c *config.Config) bool {
	return pb.firstGoFile() != "" ||
		len(pb.proto.sources.strs) > 0 && c.ProtoMode == config.DefaultProtoMode
}

// firstGoFile returns the name of a .go file if the package contains at least
// one .go file, or "" otherwise.
func (pb *packageBuilder) firstGoFile() string {
	goSrcs := []platformStringsBuilder{
		pb.library.sources,
		pb.binary.sources,
		pb.test.sources,
	}
	for _, sb := range goSrcs {
		if sb.strs != nil {
			for s, _ := range sb.strs {
				if strings.HasSuffix(s, ".go") {
					return s
				}
			}
		}
	}
	return ""
}

func (pb *packageBuilder) inferImportPath(c *config.Config) error {
	if pb.importPath != "" {
		log.Panic("importPath already set")
	}
	if pb.rel == c.GoPrefixRel {
		if c.GoPrefix == "" {
			return fmt.Errorf("in directory %q, prefix is empty, so importpath would be empty for rules. Set a prefix with a '# gazelle:prefix' comment or with -go_prefix on the command line.", pb.dir)
		}
		pb.importPath = c.GoPrefix
	} else {
		fromPrefixRel := strings.TrimPrefix(pb.rel, c.GoPrefixRel+"/")
		pb.importPath = path.Join(c.GoPrefix, fromPrefixRel)
	}
	return nil
}

func (pb *packageBuilder) build() *Package {
	return &Package{
		Name:        pb.name,
		Dir:         pb.dir,
		Rel:         pb.rel,
		ImportPath:  pb.importPath,
		Library:     pb.library.build(),
		Binary:      pb.binary.build(),
		Test:        pb.test.build(),
		Proto:       pb.proto.build(),
		HasTestdata: pb.hasTestdata,
	}
}

func (tb *goTargetBuilder) addFile(c *config.Config, info fileInfo) {
	tb.cgo = tb.cgo || info.isCgo
	add := getPlatformStringsAddFunction(c, info, nil)
	add(&tb.sources, info.name)
	add(&tb.imports, info.imports...)
	for _, copts := range info.copts {
		optAdd := add
		if len(copts.tags) > 0 {
			optAdd = getPlatformStringsAddFunction(c, info, copts.tags)
		}
		optAdd(&tb.copts, copts.opts)
	}
	for _, clinkopts := range info.clinkopts {
		optAdd := add
		if len(clinkopts.tags) > 0 {
			optAdd = getPlatformStringsAddFunction(c, info, clinkopts.tags)
		}
		optAdd(&tb.clinkopts, clinkopts.opts)
	}
}

func (tb *goTargetBuilder) build() GoTarget {
	return GoTarget{
		Sources:   tb.sources.build(),
		Imports:   tb.imports.build(),
		COpts:     tb.copts.build(),
		CLinkOpts: tb.clinkopts.build(),
		Cgo:       tb.cgo,
	}
}

func (tb *protoTargetBuilder) addFile(c *config.Config, info fileInfo) {
	add := getPlatformStringsAddFunction(c, info, nil)
	add(&tb.sources, info.name)
	add(&tb.imports, info.imports...)
	tb.hasServices = tb.hasServices || info.hasServices
}

func (tb *protoTargetBuilder) build() ProtoTarget {
	return ProtoTarget{
		Sources:     tb.sources.build(),
		Imports:     tb.imports.build(),
		HasServices: tb.hasServices,
		HasPbGo:     tb.hasPbGo,
	}
}

// getPlatformStringsAddFunction returns a function used to add strings to
// a *platformStringsBuilder under the same set of constraints. This is a
// performance optimization to avoid evaluating constraints repeatedly.
func getPlatformStringsAddFunction(c *config.Config, info fileInfo, cgoTags tagLine) func(sb *platformStringsBuilder, ss ...string) {
	isOSSpecific, isArchSpecific := isOSArchSpecific(info, cgoTags)

	switch {
	case !isOSSpecific && !isArchSpecific:
		if checkConstraints(c, "", "", info.goos, info.goarch, info.tags, cgoTags) {
			return func(sb *platformStringsBuilder, ss ...string) {
				for _, s := range ss {
					sb.addGenericString(s)
				}
			}
		}

	case isOSSpecific && !isArchSpecific:
		var osMatch []string
		for _, os := range config.KnownOSs {
			if checkConstraints(c, os, "", info.goos, info.goarch, info.tags, cgoTags) {
				osMatch = append(osMatch, os)
			}
		}
		if len(osMatch) > 0 {
			return func(sb *platformStringsBuilder, ss ...string) {
				for _, s := range ss {
					sb.addOSString(s, osMatch)
				}
			}
		}

	case !isOSSpecific && isArchSpecific:
		var archMatch []string
		for _, arch := range config.KnownArchs {
			if checkConstraints(c, "", arch, info.goos, info.goarch, info.tags, cgoTags) {
				archMatch = append(archMatch, arch)
			}
		}
		if len(archMatch) > 0 {
			return func(sb *platformStringsBuilder, ss ...string) {
				for _, s := range ss {
					sb.addArchString(s, archMatch)
				}
			}
		}

	default:
		var platformMatch []config.Platform
		for _, platform := range config.KnownPlatforms {
			if checkConstraints(c, platform.OS, platform.Arch, info.goos, info.goarch, info.tags, cgoTags) {
				platformMatch = append(platformMatch, platform)
			}
		}
		if len(platformMatch) > 0 {
			return func(sb *platformStringsBuilder, ss ...string) {
				for _, s := range ss {
					sb.addPlatformString(s, platformMatch)
				}
			}
		}
	}

	return func(_ *platformStringsBuilder, _ ...string) {}
}

func (sb *platformStringsBuilder) addGenericString(s string) {
	if sb.strs == nil {
		sb.strs = make(map[string]platformStringInfo)
	}
	sb.strs[s] = platformStringInfo{set: genericSet}
}

func (sb *platformStringsBuilder) addOSString(s string, oss []string) {
	if sb.strs == nil {
		sb.strs = make(map[string]platformStringInfo)
	}
	si, ok := sb.strs[s]
	if !ok {
		si.set = osSet
		si.oss = make(map[string]bool)
	}
	switch si.set {
	case genericSet:
		return
	case osSet:
		for _, os := range oss {
			si.oss[os] = true
		}
	default:
		si.convertToPlatforms()
		for _, os := range oss {
			for _, arch := range config.KnownOSArchs[os] {
				si.platforms[config.Platform{OS: os, Arch: arch}] = true
			}
		}
	}
	sb.strs[s] = si
}

func (sb *platformStringsBuilder) addArchString(s string, archs []string) {
	if sb.strs == nil {
		sb.strs = make(map[string]platformStringInfo)
	}
	si, ok := sb.strs[s]
	if !ok {
		si.set = archSet
		si.archs = make(map[string]bool)
	}
	switch si.set {
	case genericSet:
		return
	case archSet:
		for _, arch := range archs {
			si.archs[arch] = true
		}
	default:
		si.convertToPlatforms()
		for _, arch := range archs {
			for _, os := range config.KnownArchOSs[arch] {
				si.platforms[config.Platform{OS: os, Arch: arch}] = true
			}
		}
	}
	sb.strs[s] = si
}

func (sb *platformStringsBuilder) addPlatformString(s string, platforms []config.Platform) {
	if sb.strs == nil {
		sb.strs = make(map[string]platformStringInfo)
	}
	si, ok := sb.strs[s]
	if !ok {
		si.set = platformSet
		si.platforms = make(map[config.Platform]bool)
	}
	switch si.set {
	case genericSet:
		return
	default:
		si.convertToPlatforms()
		for _, p := range platforms {
			si.platforms[p] = true
		}
	}
	sb.strs[s] = si
}

func (sb *platformStringsBuilder) build() PlatformStrings {
	var ps PlatformStrings
	for s, si := range sb.strs {
		switch si.set {
		case genericSet:
			ps.Generic = append(ps.Generic, s)
		case osSet:
			if ps.OS == nil {
				ps.OS = make(map[string][]string)
			}
			for os, _ := range si.oss {
				ps.OS[os] = append(ps.OS[os], s)
			}
		case archSet:
			if ps.Arch == nil {
				ps.Arch = make(map[string][]string)
			}
			for arch, _ := range si.archs {
				ps.Arch[arch] = append(ps.Arch[arch], s)
			}
		case platformSet:
			if ps.Platform == nil {
				ps.Platform = make(map[config.Platform][]string)
			}
			for p, _ := range si.platforms {
				ps.Platform[p] = append(ps.Platform[p], s)
			}
		}
	}
	sort.Strings(ps.Generic)
	if ps.OS != nil {
		for _, ss := range ps.OS {
			sort.Strings(ss)
		}
	}
	if ps.Arch != nil {
		for _, ss := range ps.Arch {
			sort.Strings(ss)
		}
	}
	if ps.Platform != nil {
		for _, ss := range ps.Platform {
			sort.Strings(ss)
		}
	}
	return ps
}

func (si *platformStringInfo) convertToPlatforms() {
	switch si.set {
	case genericSet:
		log.Panic("cannot convert generic string to platforms")
	case platformSet:
		return
	case osSet:
		si.set = platformSet
		si.platforms = make(map[config.Platform]bool)
		for os, _ := range si.oss {
			for _, arch := range config.KnownOSArchs[os] {
				si.platforms[config.Platform{OS: os, Arch: arch}] = true
			}
		}
		si.oss = nil
	case archSet:
		si.set = platformSet
		si.platforms = make(map[config.Platform]bool)
		for arch, _ := range si.archs {
			for _, os := range config.KnownArchOSs[arch] {
				si.platforms[config.Platform{OS: os, Arch: arch}] = true
			}
		}
		si.archs = nil
	}
}

// MapSlice applies a function that processes slices of strings to the strings
// in "ps" and returns a new PlatformStrings with the results.
func (ps *PlatformStrings) MapSlice(f func([]string) ([]string, error)) (PlatformStrings, []error) {
	var errors []error

	mapSlice := func(ss []string) []string {
		rs, err := f(ss)
		if err != nil {
			errors = append(errors, err)
			return nil
		}
		return rs
	}

	mapStringMap := func(m map[string][]string) map[string][]string {
		if m == nil {
			return nil
		}
		rm := make(map[string][]string)
		for k, ss := range m {
			ss = mapSlice(ss)
			if len(ss) > 0 {
				rm[k] = ss
			}
		}
		if len(rm) == 0 {
			return nil
		}
		return rm
	}

	mapPlatformMap := func(m map[config.Platform][]string) map[config.Platform][]string {
		if m == nil {
			return nil
		}
		rm := make(map[config.Platform][]string)
		for k, ss := range m {
			ss = mapSlice(ss)
			if len(ss) > 0 {
				rm[k] = ss
			}
		}
		if len(rm) == 0 {
			return nil
		}
		return rm
	}

	result := PlatformStrings{
		Generic:  mapSlice(ps.Generic),
		OS:       mapStringMap(ps.OS),
		Arch:     mapStringMap(ps.Arch),
		Platform: mapPlatformMap(ps.Platform),
	}
	return result, errors
}
