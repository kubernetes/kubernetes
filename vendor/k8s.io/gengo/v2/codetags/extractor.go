/*
Copyright 2025 The Kubernetes Authors.

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

package codetags

import (
	"strings"
	"unicode/utf8"
)

// Extract identifies and collects lines containing special metadata tags.
// It processes only lines that begin with the prefix.
//
// The portion of a line immediately following the prefix is treated as
// a potential tag name. To be considered valid, this tag name must
// match the regular expression `[a-zA-Z_][a-zA-Z0-9_.-:]*`.
//
// Extract returns a map where each key is a valid tag name found in
// lines that begin with the prefix.
// The value for each key is a slice of strings. Each string in this slice
// represents the contents of an original line after the prefix has been removed.
//
// Example: When called with prefix "+k8s:", lines:
//
//	Comment line without marker
//	+k8s:noArgs # comment
//	+withValue=value1
//	+withValue=value2
//	+k8s:withArg(arg1)=value1
//	+k8s:withArg(arg2)=value2 # comment
//	+k8s:withNamedArgs(arg1=value1, arg2=value2)=value
//
// Then this function will return:
//
//	map[string][]string{
//		"noArgs":        {"noArgs # comment"},
//		"withArg":       {"withArg(arg1)=value1", "withArg(arg2)=value2 # comment"},
//		"withNamedArgs": {"withNamedArgs(arg1=value1, arg2=value2)=value"},
//	}
func Extract(prefix string, lines []string) map[string][]string {
	out := map[string][]string{}
	for _, line := range lines {
		line = strings.TrimLeft(line, " \t")
		if !strings.HasPrefix(line, prefix) {
			continue
		}
		line = line[len(prefix):]

		// Find the end of the presumed tag name.
		nameEnd := findNameEnd(line)
		name := line[:nameEnd]
		out[name] = append(out[name], line)
	}
	return out
}

// findNameEnd matches a tag in the same way as the parser.
func findNameEnd(s string) int {
	if len(s) == 0 {
		return 0
	}
	if r, _ := utf8.DecodeRuneInString(s); !isIdentBegin(r) {
		return 0
	}
	idx := strings.IndexFunc(s, func(r rune) bool {
		return !(isTagNameInterior(r))
	})
	if idx == -1 {
		return len(s)
	}
	return idx
}
