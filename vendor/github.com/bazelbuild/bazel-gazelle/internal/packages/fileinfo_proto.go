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
	"io/ioutil"
	"log"
	"path"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode"

	"github.com/bazelbuild/bazel-gazelle/internal/config"
)

var protoRe = buildProtoRegexp()

const (
	importSubexpIndex    = 1
	packageSubexpIndex   = 2
	goPackageSubexpIndex = 3
	serviceSubexpIndex   = 4
)

func protoFileInfo(c *config.Config, dir, rel, name string) fileInfo {
	info := fileNameInfo(dir, rel, name)
	content, err := ioutil.ReadFile(info.path)
	if err != nil {
		log.Printf("%s: error reading proto file: %v", info.path, err)
		return info
	}

	for _, match := range protoRe.FindAllSubmatch(content, -1) {
		switch {
		case match[importSubexpIndex] != nil:
			imp := unquoteProtoString(match[importSubexpIndex])
			info.imports = append(info.imports, imp)

		case match[packageSubexpIndex] != nil:
			pkg := string(match[packageSubexpIndex])
			if info.packageName == "" {
				info.packageName = strings.Replace(pkg, ".", "_", -1)
			}

		case match[goPackageSubexpIndex] != nil:
			gopkg := unquoteProtoString(match[goPackageSubexpIndex])
			// If there's no / in the package option, then it's just a
			// simple package name, not a full import path.
			if strings.LastIndexByte(gopkg, '/') == -1 {
				info.packageName = gopkg
			} else {
				if i := strings.LastIndexByte(gopkg, ';'); i != -1 {
					info.importPath = gopkg[:i]
					info.packageName = gopkg[i+1:]
				} else {
					info.importPath = gopkg
					info.packageName = path.Base(gopkg)
				}
			}

		case match[serviceSubexpIndex] != nil:
			info.hasServices = true

		default:
			// Comment matched. Nothing to extract.
		}
	}
	sort.Strings(info.imports)

	if info.packageName == "" {
		stem := strings.TrimSuffix(name, ".proto")
		fs := strings.FieldsFunc(stem, func(r rune) bool {
			return !(unicode.IsLetter(r) || unicode.IsNumber(r) || r == '_')
		})
		info.packageName = strings.Join(fs, "_")
	}

	return info
}

// Based on https://developers.google.com/protocol-buffers/docs/reference/proto3-spec
func buildProtoRegexp() *regexp.Regexp {
	hexEscape := `\\[xX][0-9a-fA-f]{2}`
	octEscape := `\\[0-7]{3}`
	charEscape := `\\[abfnrtv'"\\]`
	charValue := strings.Join([]string{hexEscape, octEscape, charEscape, "[^\x00\\'\\\"\\\\]"}, "|")
	strLit := `'(?:` + charValue + `|")*'|"(?:` + charValue + `|')*"`
	ident := `[A-Za-z][A-Za-z0-9_]*`
	fullIdent := ident + `(?:\.` + ident + `)*`
	importStmt := `\bimport\s*(?:public|weak)?\s*(?P<import>` + strLit + `)\s*;`
	packageStmt := `\bpackage\s*(?P<package>` + fullIdent + `)\s*;`
	goPackageStmt := `\boption\s*go_package\s*=\s*(?P<go_package>` + strLit + `)\s*;`
	serviceStmt := `(?P<service>service)`
	comment := `//[^\n]*`
	protoReSrc := strings.Join([]string{importStmt, packageStmt, goPackageStmt, serviceStmt, comment}, "|")
	return regexp.MustCompile(protoReSrc)
}

func unquoteProtoString(q []byte) string {
	// Adjust quotes so that Unquote is happy. We need a double quoted string
	// without unescaped double quote characters inside.
	noQuotes := bytes.Split(q[1:len(q)-1], []byte{'"'})
	if len(noQuotes) != 1 {
		for i := 0; i < len(noQuotes)-1; i++ {
			if len(noQuotes[i]) == 0 || noQuotes[i][len(noQuotes[i])-1] != '\\' {
				noQuotes[i] = append(noQuotes[i], '\\')
			}
		}
		q = append([]byte{'"'}, bytes.Join(noQuotes, []byte{'"'})...)
		q = append(q, '"')
	}
	if q[0] == '\'' {
		q[0] = '"'
		q[len(q)-1] = '"'
	}

	s, err := strconv.Unquote(string(q))
	if err != nil {
		log.Panicf("unquoting string literal %s from proto: %v", q, err)
	}
	return s
}
