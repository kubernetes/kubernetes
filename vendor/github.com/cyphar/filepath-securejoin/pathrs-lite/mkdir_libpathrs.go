// SPDX-License-Identifier: MPL-2.0

//go:build libpathrs

// Copyright (C) 2024-2025 Aleksa Sarai <cyphar@cyphar.com>
// Copyright (C) 2024-2025 SUSE LLC
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

package pathrs

import (
	"os"

	"cyphar.com/go-pathrs"
)

// MkdirAllHandle is equivalent to [MkdirAll], except that it is safer to use
// in two respects:
//
//   - The caller provides the root directory as an *[os.File] (preferably O_PATH)
//     handle. This means that the caller can be sure which root directory is
//     being used. Note that this can be emulated by using /proc/self/fd/... as
//     the root path with [os.MkdirAll].
//
//   - Once all of the directories have been created, an *[os.File] O_PATH handle
//     to the directory at unsafePath is returned to the caller. This is done in
//     an effectively-race-free way (an attacker would only be able to swap the
//     final directory component), which is not possible to emulate with
//     [MkdirAll].
//
// In addition, the returned handle is obtained far more efficiently than doing
// a brand new lookup of unsafePath (such as with [SecureJoin] or openat2) after
// doing [MkdirAll]. If you intend to open the directory after creating it, you
// should use MkdirAllHandle.
//
// [SecureJoin]: https://pkg.go.dev/github.com/cyphar/filepath-securejoin#SecureJoin
func MkdirAllHandle(root *os.File, unsafePath string, mode os.FileMode) (*os.File, error) {
	rootRef, err := pathrs.RootFromFile(root)
	if err != nil {
		return nil, err
	}
	defer rootRef.Close() //nolint:errcheck // close failures aren't critical here

	handle, err := rootRef.MkdirAll(unsafePath, mode)
	if err != nil {
		return nil, err
	}
	return handle.IntoFile(), nil
}
