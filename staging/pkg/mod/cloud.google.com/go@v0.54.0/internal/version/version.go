// Copyright 2016 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:generate ./update_version.sh

// Package version contains version information for Google Cloud Client
// Libraries for Go, as reported in request headers.
package version

import (
	"runtime"
	"strings"
	"unicode"
)

// Repo is the current version of the client libraries in this
// repo. It should be a date in YYYYMMDD format.
const Repo = "20200228"

// Go returns the Go runtime version. The returned string
// has no whitespace.
func Go() string {
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
