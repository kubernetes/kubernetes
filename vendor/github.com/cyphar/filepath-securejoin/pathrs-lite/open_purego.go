// SPDX-License-Identifier: MPL-2.0

//go:build linux && !libpathrs

// Copyright (C) 2024-2025 Aleksa Sarai <cyphar@cyphar.com>
// Copyright (C) 2024-2025 SUSE LLC
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

package pathrs

import (
	"os"

	"github.com/cyphar/filepath-securejoin/pathrs-lite/internal/gopathrs"
	"github.com/cyphar/filepath-securejoin/pathrs-lite/internal/procfs"
)

// OpenatInRoot is equivalent to [OpenInRoot], except that the root is provided
// using an *[os.File] handle, to ensure that the correct root directory is used.
func OpenatInRoot(root *os.File, unsafePath string) (*os.File, error) {
	return gopathrs.OpenatInRoot(root, unsafePath)
}

// Reopen takes an *[os.File] handle and re-opens it through /proc/self/fd.
// Reopen(file, flags) is effectively equivalent to
//
//	fdPath := fmt.Sprintf("/proc/self/fd/%d", file.Fd())
//	os.OpenFile(fdPath, flags|unix.O_CLOEXEC)
//
// But with some extra hardenings to ensure that we are not tricked by a
// maliciously-configured /proc mount. While this attack scenario is not
// common, in container runtimes it is possible for higher-level runtimes to be
// tricked into configuring an unsafe /proc that can be used to attack file
// operations. See [CVE-2019-19921] for more details.
//
// [CVE-2019-19921]: https://github.com/advisories/GHSA-fh74-hm69-rqjw
func Reopen(handle *os.File, flags int) (*os.File, error) {
	return procfs.ReopenFd(handle, flags)
}
