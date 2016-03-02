/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

// Package types contains go type information, packaged in a way that makes
// auto-generation convenient, whether by template or straight go functions.
package types

import (
	"strings"
)

// ExtractCommentTags parses comments for lines of the form:
//
//   'marker'+"key1=value1,key2=value2".
//
// Values are optional; 'true' is the default. If a key is set multiple times,
// the last one wins.
//
// Example: if you pass "+" for 'marker', and the following two lines are in
// the comments:
//   +foo=value1,bar
//   +foo=value2,baz="frobber"
// Then this function will return:
//   map[string]string{"foo":"value2", "bar": "true", "baz": "frobber"}
//
// TODO: Basically we need to define a standard way of giving instructions to
// autogenerators in the comments of a type. This is a first iteration of that.
// TODO: allow multiple values per key?
func ExtractCommentTags(marker, allLines string) map[string]string {
	lines := strings.Split(allLines, "\n")
	out := map[string]string{}
	for _, line := range lines {
		line = strings.Trim(line, " ")
		if len(line) == 0 {
			continue
		}
		if !strings.HasPrefix(line, marker) {
			continue
		}
		pairs := strings.Split(line[len(marker):], ",")
		for _, p := range pairs {
			kv := strings.Split(p, "=")
			if len(kv) == 2 {
				out[kv[0]] = kv[1]
			} else if len(kv) == 1 {
				out[kv[0]] = "true"
			}
		}
	}
	return out
}
