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
	"bufio"
	"log"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/bazelbuild/bazel-gazelle/internal/config"
)

// fileInfo holds information used to decide how to build a file. This
// information comes from the file's name, from package and import declarations
// (in .go files), and from +build and cgo comments.
type fileInfo struct {
	path, rel, name, ext string

	// packageName is the Go package name of a .go file, without the
	// "_test" suffix if it was present. It is empty for non-Go files.
	packageName string

	// importPath is the canonical import path for this file's package.
	// This may be read from a package comment (in Go) or a go_package
	// option (in proto). This field is empty for files that don't specify
	// an import path.
	importPath string

	// category is the type of file, based on extension.
	category extCategory

	// isTest is true if the file stem (the part before the extension)
	// ends with "_test.go". This is never true for non-Go files.
	isTest bool

	// isXTest is true for test Go files whose declared package name ends
	// with "_test".
	isXTest bool

	// imports is a list of packages imported by a file. It does not include
	// "C" or anything from the standard library.
	imports []string

	// isCgo is true for .go files that import "C".
	isCgo bool

	// goos and goarch contain the OS and architecture suffixes in the filename,
	// if they were present.
	goos, goarch string

	// tags is a list of build tag lines. Each entry is the trimmed text of
	// a line after a "+build" prefix.
	tags []tagLine

	// copts and clinkopts contain flags that are part of CFLAGS, CPPFLAGS,
	// CXXFLAGS, and LDFLAGS directives in cgo comments.
	copts, clinkopts []taggedOpts

	// hasServices indicates whether a .proto file has service definitions.
	hasServices bool
}

// tagLine represents the space-separated disjunction of build tag groups
// in a line comment.
type tagLine []tagGroup

// check returns true if at least one of the tag groups is satisfied.
func (l tagLine) check(c *config.Config, os, arch string) bool {
	if len(l) == 0 {
		return false
	}
	for _, g := range l {
		if g.check(c, os, arch) {
			return true
		}
	}
	return false
}

// tagGroup represents a comma-separated conjuction of build tags.
type tagGroup []string

// check returns true if all of the tags are true. Tags that start with
// "!" are negated (but "!!") is not allowed. Go release tags (e.g., "go1.8")
// are ignored. If the group contains an os or arch tag, but the os or arch
// parameters are empty, check returns false even if the tag is negated.
func (g tagGroup) check(c *config.Config, os, arch string) bool {
	for _, t := range g {
		if strings.HasPrefix(t, "!!") { // bad syntax, reject always
			return false
		}
		not := strings.HasPrefix(t, "!")
		if not {
			t = t[1:]
		}
		if isIgnoredTag(t) {
			// Release tags are treated as "unknown" and are considered true,
			// whether or not they are negated.
			continue
		}
		var match bool
		if _, ok := config.KnownOSSet[t]; ok {
			if os == "" {
				return false
			}
			match = os == t
		} else if _, ok := config.KnownArchSet[t]; ok {
			if arch == "" {
				return false
			}
			match = arch == t
		} else {
			match = c.GenericTags[t]
		}
		if not {
			match = !match
		}
		if !match {
			return false
		}
	}
	return true
}

// taggedOpts a list of compile or link options which should only be applied
// if the given set of build tags are satisfied. These options have already
// been tokenized using the same algorithm that "go build" uses, then joined
// with OptSeparator.
type taggedOpts struct {
	tags tagLine
	opts string
}

// OptSeparator is a special character inserted between options that appeared
// together in a #cgo directive. This allows options to be split, modified,
// and escaped by other packages.
//
// It's important to keep options grouped together in the same string. For
// example, if we have "-framework IOKit" together in a #cgo directive,
// "-framework" shouldn't be treated as a separate string for the purposes of
// sorting and de-duplicating.
const OptSeparator = "\x1D"

// extCategory indicates how a file should be treated, based on extension.
type extCategory int

const (
	// ignoredExt is applied to files which are not part of a build.
	ignoredExt extCategory = iota

	// unsupportedExt is applied to files that we don't support but would be
	// built with "go build".
	unsupportedExt

	// goExt is applied to .go files.
	goExt

	// cExt is applied to C and C++ files.
	cExt

	// hExt is applied to header files. If cgo code is present, these may be
	// C or C++ headers. If not, they are treated as Go assembly headers.
	hExt

	// sExt is applied to Go assembly files, ending with .s.
	sExt

	// csExt is applied to other assembly files, ending with .S. These are built
	// with the C compiler if cgo code is present.
	csExt

	// protoExt is applied to .proto files.
	protoExt
)

// fileNameInfo returns information that can be inferred from the name of
// a file. It does not read data from the file.
func fileNameInfo(dir, rel, name string) fileInfo {
	ext := path.Ext(name)

	// Categorize the file based on extension. Based on go/build.Context.Import.
	var category extCategory
	switch ext {
	case ".go":
		category = goExt
	case ".c", ".cc", ".cpp", ".cxx":
		category = cExt
	case ".h", ".hh", ".hpp", ".hxx":
		category = hExt
	case ".s":
		category = sExt
	case ".S":
		category = csExt
	case ".proto":
		category = protoExt
	case ".m", ".f", ".F", ".for", ".f90", ".swig", ".swigcxx", ".syso":
		category = unsupportedExt
	default:
		category = ignoredExt
	}

	// Determine test, goos, and goarch. This is intended to match the logic
	// in goodOSArchFile in go/build.
	var isTest bool
	var goos, goarch string
	l := strings.Split(name[:len(name)-len(ext)], "_")
	if len(l) >= 2 && l[len(l)-1] == "test" {
		isTest = category == goExt
		l = l[:len(l)-1]
	}
	switch {
	case len(l) >= 3 && config.KnownOSSet[l[len(l)-2]] && config.KnownArchSet[l[len(l)-1]]:
		goos = l[len(l)-2]
		goarch = l[len(l)-1]
	case len(l) >= 2 && config.KnownOSSet[l[len(l)-1]]:
		goos = l[len(l)-1]
	case len(l) >= 2 && config.KnownArchSet[l[len(l)-1]]:
		goarch = l[len(l)-1]
	}

	return fileInfo{
		path:     filepath.Join(dir, name),
		rel:      rel,
		name:     name,
		ext:      ext,
		category: category,
		isTest:   isTest,
		goos:     goos,
		goarch:   goarch,
	}
}

// otherFileInfo returns information about a non-.go file. It will parse
// part of the file to determine build tags. If the file can't be read, an
// error will be logged, and partial information will be returned.
func otherFileInfo(dir, rel, name string) fileInfo {
	info := fileNameInfo(dir, rel, name)
	if info.category == ignoredExt {
		return info
	}
	if info.category == unsupportedExt {
		log.Printf("%s: warning: file extension not yet supported", info.path)
		return info
	}

	tags, err := readTags(info.path)
	if err != nil {
		log.Printf("%s: error reading file: %v", info.path, err)
		return info
	}
	info.tags = tags
	return info
}

// readTags reads and extracts build tags from the block of comments
// and blank lines at the start of a file which is separated from the
// rest of the file by a blank line. Each string in the returned slice
// is the trimmed text of a line after a "+build" prefix.
// Based on go/build.Context.shouldBuild.
func readTags(path string) ([]tagLine, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)

	// Pass 1: Identify leading run of // comments and blank lines,
	// which must be followed by a blank line.
	var lines []string
	end := 0
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			end = len(lines)
			continue
		}
		if strings.HasPrefix(line, "//") {
			lines = append(lines, line[len("//"):])
			continue
		}
		break
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	lines = lines[:end]

	// Pass 2: Process each line in the run.
	var tagLines []tagLine
	for _, line := range lines {
		fields := strings.Fields(line)
		if len(fields) > 0 && fields[0] == "+build" {
			tagLines = append(tagLines, parseTagsInGroups(fields[1:]))
		}
	}
	return tagLines, nil
}

func parseTagsInGroups(groups []string) tagLine {
	var l tagLine
	for _, g := range groups {
		l = append(l, tagGroup(strings.Split(g, ",")))
	}
	return l
}

func isOSArchSpecific(info fileInfo, cgoTags tagLine) (osSpecific, archSpecific bool) {
	if info.goos != "" {
		osSpecific = true
	}
	if info.goarch != "" {
		archSpecific = true
	}
	lines := info.tags
	if len(cgoTags) > 0 {
		lines = append(lines, cgoTags)
	}
	for _, line := range lines {
		for _, group := range line {
			for _, tag := range group {
				if strings.HasPrefix(tag, "!") {
					tag = tag[1:]
				}
				_, osOk := config.KnownOSSet[tag]
				if osOk {
					osSpecific = true
				}
				_, archOk := config.KnownArchSet[tag]
				if archOk {
					archSpecific = true
				}
			}
		}
	}
	return osSpecific, archSpecific
}

// checkConstraints determines whether build constraints are satisfied on
// a given platform.
//
// The first few arguments describe the platform. genericTags is the set
// of build tags that are true on all platforms. os and arch are the platform
// GOOS and GOARCH strings. If os or arch is empty, checkConstraints will
// return false in the presence of OS and architecture constraints, even
// if they are negated.
//
// The remaining arguments describe the file being tested. All of these may
// be empty or nil. osSuffix and archSuffix are filename suffixes. fileTags
// is a list tags from +build comments found near the top of the file. cgoTags
// is an extra set of tags in a #cgo directive.
func checkConstraints(c *config.Config, os, arch, osSuffix, archSuffix string, fileTags []tagLine, cgoTags tagLine) bool {
	if osSuffix != "" && osSuffix != os || archSuffix != "" && archSuffix != arch {
		return false
	}
	for _, l := range fileTags {
		if !l.check(c, os, arch) {
			return false
		}
	}
	if len(cgoTags) > 0 && !cgoTags.check(c, os, arch) {
		return false
	}
	return true
}

// isIgnoredTag returns whether the tag is "cgo" or is a release tag.
// Release tags match the pattern "go[0-9]\.[0-9]+".
// Gazelle won't consider whether an ignored tag is satisfied when evaluating
// build constraints for a file.
func isIgnoredTag(tag string) bool {
	if tag == "cgo" {
		return true
	}
	if len(tag) < 5 || !strings.HasPrefix(tag, "go") {
		return false
	}
	if tag[2] < '0' || tag[2] > '9' || tag[3] != '.' {
		return false
	}
	for _, c := range tag[4:] {
		if c < '0' || c > '9' {
			return false
		}
	}
	return true
}
