// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows

package fs

import (
	"os"
	"syscall"

	"github.com/pkg/errors"
)

// renameFallback attempts to determine the appropriate fallback to failed rename
// operation depending on the resulting error.
func renameFallback(err error, src, dst string) error {
	// Rename may fail if src and dst are on different devices; fall back to
	// copy if we detect that case. syscall.EXDEV is the common name for the
	// cross device link error which has varying output text across different
	// operating systems.
	terr, ok := err.(*os.LinkError)
	if !ok {
		return err
	} else if terr.Err != syscall.EXDEV {
		return errors.Wrapf(terr, "link error: cannot rename %s to %s", src, dst)
	}

	return renameByCopy(src, dst)
}
