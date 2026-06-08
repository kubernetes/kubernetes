// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package keys

import (
	"sort"
	"strings"
)

// Join returns a canonical join of the keys in S:
// a sorted comma-separated string list.
func Join[S ~[]T, T ~string](s S) string {
	strs := make([]string, 0, len(s))
	for _, v := range s {
		strs = append(strs, string(v))
	}
	sort.Strings(strs)
	return strings.Join(strs, ",")
}
