// Copyright 2020 Google LLC. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gensupport

import (
	"runtime"
	"strings"
	"unicode"
)

// GoVersion returns the Go runtime version. The returned string
// has no whitespace.
func GoVersion() string {
	return goVersion
}

var goVersion = goVer(runtime.Version())

const develPrefix = "devel +"

func goVer(s string) string {
	if strings.HasPrefix(s, develPrefix) {
		s = s[len(develPrefix):]
		if p := strings.IndexFunc(s, unicode.IsSpace); p >= 0 {
			s = s[:p]
		}
		return s
	}

	if strings.HasPrefix(s, "go1") {
		s = s[2:]
		var prerelease string
		if p := strings.IndexFunc(s, notSemverRune); p >= 0 {
			s, prerelease = s[:p], s[p:]
		}
		if strings.HasSuffix(s, ".") {
			s += "0"
		} else if strings.Count(s, ".") < 2 {
			s += ".0"
		}
		if prerelease != "" {
			s += "-" + prerelease
		}
		return s
	}
	return ""
}

func notSemverRune(r rune) bool {
	return !strings.ContainsRune("0123456789.", r)
}
