// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package filesys

import "path/filepath"

// RootedPath returns a rooted path, e.g. "/foo/bar" as
// opposed to "foo/bar".
func RootedPath(elem ...string) string {
	return separator + filepath.Join(elem...)
}
