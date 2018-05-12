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

package pathtools

import (
	"path"
	"path/filepath"
	"strings"
)

// HasPrefix returns whether the slash-separated path p has the given
// prefix. Unlike strings.HasPrefix, this function respects component
// boundaries, so "/home/foo" is not a prefix is "/home/foobar/baz". If the
// prefix is empty, this function always returns true.
func HasPrefix(p, prefix string) bool {
	return prefix == "" || p == prefix || strings.HasPrefix(p, prefix+"/")
}

// TrimPrefix returns p without the provided prefix. If p doesn't start
// with prefix, it returns p unchanged. Unlike strings.HasPrefix, this function
// respects component boundaries (assuming slash-separated paths), so
// TrimPrefix("foo/bar", "foo") returns "baz".
func TrimPrefix(p, prefix string) string {
	if prefix == "" {
		return p
	}
	if prefix == p {
		return ""
	}
	return strings.TrimPrefix(p, prefix+"/")
}

// RelBaseName returns the base name for rel, a slash-separated path relative
// to the repository root. If rel is empty, RelBaseName returns the base name
// of prefix. If prefix is empty, RelBaseName returns the base name of root,
// the absolute file path of the repository root directory. If that's empty
// to, then RelBaseName returns "root".
func RelBaseName(rel, prefix, root string) string {
	base := path.Base(rel)
	if base == "." || base == "/" {
		base = path.Base(prefix)
	}
	if base == "." || base == "/" {
		base = filepath.Base(root)
	}
	if base == "." || base == "/" {
		base = "root"
	}
	return base
}
