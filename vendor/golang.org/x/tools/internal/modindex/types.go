// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modindex

import (
	"strings"
)

// some special types to avoid confusions

// distinguish various types of directory names. It's easy to get confused.
type Abspath string // absolute paths
type Relpath string // paths with GOMODCACHE prefix removed

func toRelpath(cachedir Abspath, s string) Relpath {
	if strings.HasPrefix(s, string(cachedir)) {
		if s == string(cachedir) {
			return Relpath("")
		}
		return Relpath(s[len(cachedir)+1:])
	}
	return Relpath(s)
}
