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
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"log"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/bazelbuild/bazel-gazelle/internal/config"
	"github.com/bazelbuild/bazel-gazelle/internal/language/proto"
	"github.com/bazelbuild/bazel-gazelle/internal/rule"
)

// fileInfo holds information used to decide how to build a file. This
// information comes from the file's name, from package and import declarations
// (in .go files), and from +build and cgo comments.
type fileInfo struct {
	path string
	name string

	// ext is the type of file, based on extension.
	ext ext

	// packageName is the Go package name of a .go file, without the
	// "_test" suffix if it was present. It is empty for non-Go files.
	packageName string

	// importPath is the canonical import path for this file's package.
	// This may be read from a package comment (in Go) or a go_package
	// option (in proto). This field is empty for files that don't specify
	// an import path.
	importPath string

	// isTest is true if the file stem (the part before the extension)
	// ends with "_test.go". This is never true for non-Go files.
	isTest bool

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
	goConf := getGoConfig(c)
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
		if _, ok := rule.KnownOSSet[t]; ok {
			if os == "" {
				return false
			}
			match = os == t
		} else if _, ok := rule.KnownArchSet[t]; ok {
			if arch == "" {
				return false
			}
			match = arch == t
		} else {
			match = goConf.genericTags[t]
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

// optSeparator is a special character inserted between options that appeared
// together in a #cgo directive. This allows options to be split, modified,
// and escaped by other packages.
//
// It's important to keep options grouped together in the same string. For
// example, if we have "-framework IOKit" together in a #cgo directive,
// "-framework" shouldn't be treated as a separate string for the purposes of
// sorting and de-duplicating.
const optSeparator = "\x1D"

// ext indicates how a file should be treated, based on extension.
type ext int

const (
	// unknownExt is applied files that aren't buildable with Go.
	unknownExt ext = iota

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
func fileNameInfo(path_ string) fileInfo {
	name := filepath.Base(path_)
	var ext ext
	switch path.Ext(name) {
	case ".go":
		ext = goExt
	case ".c", ".cc", ".cpp", ".cxx", ".m", ".mm":
		ext = cExt
	case ".h", ".hh", ".hpp", ".hxx":
		ext = hExt
	case ".s":
		ext = sExt
	case ".S":
		ext = csExt
	case ".proto":
		ext = protoExt
	default:
		ext = unknownExt
	}

	// Determine test, goos, and goarch. This is intended to match the logic
	// in goodOSArchFile in go/build.
	var isTest bool
	var goos, goarch string
	l := strings.Split(name[:len(name)-len(path.Ext(name))], "_")
	if len(l) >= 2 && l[len(l)-1] == "test" {
		isTest = ext == goExt
		l = l[:len(l)-1]
	}
	switch {
	case len(l) >= 3 && rule.KnownOSSet[l[len(l)-2]] && rule.KnownArchSet[l[len(l)-1]]:
		goos = l[len(l)-2]
		goarch = l[len(l)-1]
	case len(l) >= 2 && rule.KnownOSSet[l[len(l)-1]]:
		goos = l[len(l)-1]
	case len(l) >= 2 && rule.KnownArchSet[l[len(l)-1]]:
		goarch = l[len(l)-1]
	}

	return fileInfo{
		path:   path_,
		name:   name,
		ext:    ext,
		isTest: isTest,
		goos:   goos,
		goarch: goarch,
	}
}

// otherFileInfo returns information about a non-.go file. It will parse
// part of the file to determine build tags. If the file can't be read, an
// error will be logged, and partial information will be returned.
func otherFileInfo(path string) fileInfo {
	info := fileNameInfo(path)
	if info.ext == unknownExt {
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

// goFileInfo returns information about a .go file. It will parse part of the
// file to determine the package name, imports, and build constraints.
// If the file can't be read, an error will be logged, and partial information
// will be returned.
// This function is intended to match go/build.Context.Import.
// TODD(#53): extract canonical import path
func goFileInfo(path, rel string) fileInfo {
	info := fileNameInfo(path)
	fset := token.NewFileSet()
	pf, err := parser.ParseFile(fset, info.path, nil, parser.ImportsOnly|parser.ParseComments)
	if err != nil {
		log.Printf("%s: error reading go file: %v", info.path, err)
		return info
	}

	info.packageName = pf.Name.Name
	if info.isTest && strings.HasSuffix(info.packageName, "_test") {
		info.packageName = info.packageName[:len(info.packageName)-len("_test")]
	}

	for _, decl := range pf.Decls {
		d, ok := decl.(*ast.GenDecl)
		if !ok {
			continue
		}
		for _, dspec := range d.Specs {
			spec, ok := dspec.(*ast.ImportSpec)
			if !ok {
				continue
			}
			quoted := spec.Path.Value
			path, err := strconv.Unquote(quoted)
			if err != nil {
				log.Printf("%s: error reading go file: %v", info.path, err)
				continue
			}

			if path == "C" {
				if info.isTest {
					log.Printf("%s: warning: use of cgo in test not supported", info.path)
				}
				info.isCgo = true
				cg := spec.Doc
				if cg == nil && len(d.Specs) == 1 {
					cg = d.Doc
				}
				if cg != nil {
					if err := saveCgo(&info, rel, cg); err != nil {
						log.Printf("%s: error reading go file: %v", info.path, err)
					}
				}
				continue
			}
			info.imports = append(info.imports, path)
		}
	}

	tags, err := readTags(info.path)
	if err != nil {
		log.Printf("%s: error reading go file: %v", info.path, err)
		return info
	}
	info.tags = tags

	return info
}

// saveCgo extracts CFLAGS, CPPFLAGS, CXXFLAGS, and LDFLAGS directives
// from a comment above a "C" import. This is intended to match logic in
// go/build.Context.saveCgo.
func saveCgo(info *fileInfo, rel string, cg *ast.CommentGroup) error {
	text := cg.Text()
	for _, line := range strings.Split(text, "\n") {
		orig := line

		// Line is
		//	#cgo [GOOS/GOARCH...] LDFLAGS: stuff
		//
		line = strings.TrimSpace(line)
		if len(line) < 5 || line[:4] != "#cgo" || (line[4] != ' ' && line[4] != '\t') {
			continue
		}

		// Split at colon.
		line = strings.TrimSpace(line[4:])
		i := strings.Index(line, ":")
		if i < 0 {
			return fmt.Errorf("%s: invalid #cgo line: %s", info.path, orig)
		}
		line, optstr := strings.TrimSpace(line[:i]), strings.TrimSpace(line[i+1:])

		// Parse tags and verb.
		f := strings.Fields(line)
		if len(f) < 1 {
			return fmt.Errorf("%s: invalid #cgo line: %s", info.path, orig)
		}
		verb := f[len(f)-1]
		tags := parseTagsInGroups(f[:len(f)-1])

		// Parse options.
		opts, err := splitQuoted(optstr)
		if err != nil {
			return fmt.Errorf("%s: invalid #cgo line: %s", info.path, orig)
		}
		var ok bool
		for i, opt := range opts {
			if opt, ok = expandSrcDir(opt, rel); !ok {
				return fmt.Errorf("%s: malformed #cgo argument: %s", info.path, orig)
			}
			opts[i] = opt
		}
		joinedStr := strings.Join(opts, optSeparator)

		// Add tags to appropriate list.
		switch verb {
		case "CFLAGS", "CPPFLAGS", "CXXFLAGS":
			info.copts = append(info.copts, taggedOpts{tags, joinedStr})
		case "LDFLAGS":
			info.clinkopts = append(info.clinkopts, taggedOpts{tags, joinedStr})
		case "pkg-config":
			return fmt.Errorf("%s: pkg-config not supported: %s", info.path, orig)
		default:
			return fmt.Errorf("%s: invalid #cgo verb: %s", info.path, orig)
		}
	}
	return nil
}

// splitQuoted splits the string s around each instance of one or more consecutive
// white space characters while taking into account quotes and escaping, and
// returns an array of substrings of s or an empty list if s contains only white space.
// Single quotes and double quotes are recognized to prevent splitting within the
// quoted region, and are removed from the resulting substrings. If a quote in s
// isn't closed err will be set and r will have the unclosed argument as the
// last element. The backslash is used for escaping.
//
// For example, the following string:
//
//     a b:"c d" 'e''f'  "g\""
//
// Would be parsed as:
//
//     []string{"a", "b:c d", "ef", `g"`}
//
// Copied from go/build.splitQuoted
func splitQuoted(s string) (r []string, err error) {
	var args []string
	arg := make([]rune, len(s))
	escaped := false
	quoted := false
	quote := '\x00'
	i := 0
	for _, rune := range s {
		switch {
		case escaped:
			escaped = false
		case rune == '\\':
			escaped = true
			continue
		case quote != '\x00':
			if rune == quote {
				quote = '\x00'
				continue
			}
		case rune == '"' || rune == '\'':
			quoted = true
			quote = rune
			continue
		case unicode.IsSpace(rune):
			if quoted || i > 0 {
				quoted = false
				args = append(args, string(arg[:i]))
				i = 0
			}
			continue
		}
		arg[i] = rune
		i++
	}
	if quoted || i > 0 {
		args = append(args, string(arg[:i]))
	}
	if quote != 0 {
		err = errors.New("unclosed quote")
	} else if escaped {
		err = errors.New("unfinished escaping")
	}
	return args, err
}

// expandSrcDir expands any occurrence of ${SRCDIR}, making sure
// the result is safe for the shell.
//
// Copied from go/build.expandSrcDir
func expandSrcDir(str string, srcdir string) (string, bool) {
	// "\" delimited paths cause safeCgoName to fail
	// so convert native paths with a different delimiter
	// to "/" before starting (eg: on windows).
	srcdir = filepath.ToSlash(srcdir)

	// Spaces are tolerated in ${SRCDIR}, but not anywhere else.
	chunks := strings.Split(str, "${SRCDIR}")
	if len(chunks) < 2 {
		return str, safeCgoName(str, false)
	}
	ok := true
	for _, chunk := range chunks {
		ok = ok && (chunk == "" || safeCgoName(chunk, false))
	}
	ok = ok && (srcdir == "" || safeCgoName(srcdir, true))
	res := strings.Join(chunks, srcdir)
	return res, ok && res != ""
}

// NOTE: $ is not safe for the shell, but it is allowed here because of linker options like -Wl,$ORIGIN.
// We never pass these arguments to a shell (just to programs we construct argv for), so this should be okay.
// See golang.org/issue/6038.
// The @ is for OS X. See golang.org/issue/13720.
// The % is for Jenkins. See golang.org/issue/16959.
const safeString = "+-.,/0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz:$@%"
const safeSpaces = " "

var safeBytes = []byte(safeSpaces + safeString)

// Copied from go/build.safeCgoName
func safeCgoName(s string, spaces bool) bool {
	if s == "" {
		return false
	}
	safe := safeBytes
	if !spaces {
		safe = safe[len(safeSpaces):]
	}
	for i := 0; i < len(s); i++ {
		if c := s[i]; c < utf8.RuneSelf && bytes.IndexByte(safe, c) < 0 {
			return false
		}
	}
	return true
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
				_, osOk := rule.KnownOSSet[tag]
				if osOk {
					osSpecific = true
				}
				_, archOk := rule.KnownArchSet[tag]
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
	if tag == "cgo" || tag == "race" || tag == "msan" {
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

// protoFileInfo extracts metadata from a proto file. The proto extension
// already "parses" these and stores metadata in proto.FileInfo, so this is
// just processing relevant options.
func protoFileInfo(path_ string, protoInfo proto.FileInfo) fileInfo {
	info := fileNameInfo(path_)

	// Look for "option go_package".  If there's no / in the package option, then
	// it's just a simple package name, not a full import path.
	for _, opt := range protoInfo.Options {
		if opt.Key != "go_package" {
			continue
		}
		if strings.LastIndexByte(opt.Value, '/') == -1 {
			info.packageName = opt.Value
		} else {
			if i := strings.LastIndexByte(opt.Value, ';'); i != -1 {
				info.importPath = opt.Value[:i]
				info.packageName = opt.Value[i+1:]
			} else {
				info.importPath = opt.Value
				info.packageName = path.Base(opt.Value)
			}
		}
	}

	// Set the Go package name from the proto package name if there was no
	// option go_package.
	if info.packageName == "" && protoInfo.PackageName != "" {
		info.packageName = strings.Replace(protoInfo.PackageName, ".", "_", -1)
	}

	info.imports = protoInfo.Imports
	info.hasServices = protoInfo.HasServices
	return info
}
