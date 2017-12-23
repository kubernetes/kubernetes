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

package version

import (
	"fmt"
	"strconv"
	"strings"
)

// Version is a tuple of non-negative integers that represents the version of
// a software package.
type Version []int

func (v Version) String() string {
	cstrs := make([]string, len(v))
	for i, cn := range v {
		cstrs[i] = strconv.Itoa(cn)
	}
	return strings.Join(cstrs, ".")
}

// Compare returns an integer comparing two versions lexicographically.
func (x Version) Compare(y Version) int {
	n := len(x)
	if len(y) < n {
		n = len(y)
	}
	for i := 0; i < n; i++ {
		cmp := x[i] - y[i]
		if cmp != 0 {
			return cmp
		}
	}
	return len(x) - len(y)
}

// ParseVersion parses a version of the form "12.34.56-abcd". Non-negative
// integer components are separated by dots. An arbitrary suffix may appear
// after '-', which is ignored.
func ParseVersion(vs string) (Version, error) {
	i := strings.IndexByte(vs, '-')
	if i >= 0 {
		vs = vs[:i]
	}
	cstrs := strings.Split(vs, ".")
	v := make(Version, len(cstrs))
	for i, cstr := range cstrs {
		cn, err := strconv.Atoi(cstr)
		if err != nil {
			return nil, fmt.Errorf("could not parse version string: %q is not an integer", cstr)
		}
		if cn < 0 {
			return nil, fmt.Errorf("could not parse version string: %q is negative", cstr)
		}
		v[i] = cn
	}
	return v, nil
}
