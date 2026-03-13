// SPDX-License-Identifier: MPL-2.0

//go:build linux

// Copyright (C) 2024-2025 Aleksa Sarai <cyphar@cyphar.com>
// Copyright (C) 2024-2025 SUSE LLC
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

package pathrs

import (
	"os"

	"golang.org/x/sys/unix"
)

// OpenInRoot safely opens the provided unsafePath within the root.
// Effectively, OpenInRoot(root, unsafePath) is equivalent to
//
//	path, _ := securejoin.SecureJoin(root, unsafePath)
//	handle, err := os.OpenFile(path, unix.O_PATH|unix.O_CLOEXEC)
//
// But is much safer. The above implementation is unsafe because if an attacker
// can modify the filesystem tree between [SecureJoin] and [os.OpenFile], it is
// possible for the returned file to be outside of the root.
//
// Note that the returned handle is an O_PATH handle, meaning that only a very
// limited set of operations will work on the handle. This is done to avoid
// accidentally opening an untrusted file that could cause issues (such as a
// disconnected TTY that could cause a DoS, or some other issue). In order to
// use the returned handle, you can "upgrade" it to a proper handle using
// [Reopen].
//
// [SecureJoin]: https://pkg.go.dev/github.com/cyphar/filepath-securejoin#SecureJoin
func OpenInRoot(root, unsafePath string) (*os.File, error) {
	rootDir, err := os.OpenFile(root, unix.O_PATH|unix.O_DIRECTORY|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, err
	}
	defer rootDir.Close() //nolint:errcheck // close failures aren't critical here
	return OpenatInRoot(rootDir, unsafePath)
}
