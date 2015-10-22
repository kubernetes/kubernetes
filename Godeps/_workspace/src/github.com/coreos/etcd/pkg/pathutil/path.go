// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pathutil

import "path"

// CanonicalURLPath returns the canonical url path for p, which follows the rules:
// 1. the path always starts with "/"
// 2. replace multiple slashes with a single slash
// 3. replace each '.' '..' path name element with equivalent one
// 4. keep the trailing slash
// The function is borrowed from stdlib http.cleanPath in server.go.
func CanonicalURLPath(p string) string {
	if p == "" {
		return "/"
	}
	if p[0] != '/' {
		p = "/" + p
	}
	np := path.Clean(p)
	// path.Clean removes trailing slash except for root,
	// put the trailing slash back if necessary.
	if p[len(p)-1] == '/' && np != "/" {
		np += "/"
	}
	return np
}
