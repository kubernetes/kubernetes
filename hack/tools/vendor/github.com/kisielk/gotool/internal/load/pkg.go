// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.9

// Package load loads packages.
package load

import (
	"strings"
)

// isStandardImportPath reports whether $GOROOT/src/path should be considered
// part of the standard distribution. For historical reasons we allow people to add
// their own code to $GOROOT instead of using $GOPATH, but we assume that
// code will start with a domain name (dot in the first element).
func isStandardImportPath(path string) bool {
	i := strings.Index(path, "/")
	if i < 0 {
		i = len(path)
	}
	elem := path[:i]
	return !strings.Contains(elem, ".")
}
