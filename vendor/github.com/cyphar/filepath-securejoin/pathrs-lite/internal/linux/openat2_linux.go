// SPDX-License-Identifier: MPL-2.0

//go:build linux

// Copyright (C) 2024-2025 Aleksa Sarai <cyphar@cyphar.com>
// Copyright (C) 2024-2025 SUSE LLC
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

package linux

import (
	"golang.org/x/sys/unix"

	"github.com/cyphar/filepath-securejoin/pathrs-lite/internal/gocompat"
)

// HasOpenat2 returns whether openat2(2) is supported on the running kernel.
var HasOpenat2 = gocompat.SyncOnceValue(func() bool {
	fd, err := unix.Openat2(unix.AT_FDCWD, ".", &unix.OpenHow{
		Flags:   unix.O_PATH | unix.O_CLOEXEC,
		Resolve: unix.RESOLVE_NO_SYMLINKS | unix.RESOLVE_IN_ROOT,
	})
	if err != nil {
		return false
	}
	_ = unix.Close(fd)
	return true
})
