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
	"bytes"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"log"
	"path/filepath"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/bazelbuild/bazel-gazelle/internal/config"
)

// goFileInfo returns information about a .go file. It will parse part of the
// file to determine the package name, imports, and build constraints.
// If the file can't be read, an error will be logged, and partial information
// will be returned.
// This function is intended to match go/build.Context.Import.
// TODD(#53): extract canonical import path
func goFileInfo(c *config.Config, dir, rel, name string) fileInfo {
	info := fileNameInfo(dir, rel, name)
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
					if err := saveCgo(&info, cg); err != nil {
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
func saveCgo(info *fileInfo, cg *ast.CommentGroup) error {
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
			if opt, ok = expandSrcDir(opt, info.rel); !ok {
				return fmt.Errorf("%s: malformed #cgo argument: %s", info.path, orig)
			}
			opts[i] = opt
		}
		joinedStr := strings.Join(opts, OptSeparator)

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
