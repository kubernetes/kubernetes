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

// sawOpenat2Error stores whether we have seen an error from HasOpenat2. This
// is a one-way toggle, so as soon as we see an error we "lock" into that mode.
// We cannot use sync.OnceValue to store the success/fail state once because it
// is possible for the program we are running in to apply a seccomp-bpf filter
// and thus disable openat2 during execution.
var sawOpenat2Error gocompat.Bool

// HasOpenat2 returns whether openat2(2) is supported on the running kernel.
var HasOpenat2 = func() bool {
	if sawOpenat2Error.Load() {
		return false
	}

	fd, err := unix.Openat2(unix.AT_FDCWD, ".", &unix.OpenHow{
		Flags:   unix.O_PATH | unix.O_CLOEXEC,
		Resolve: unix.RESOLVE_NO_SYMLINKS | unix.RESOLVE_IN_ROOT,
	})
	if err != nil {
		sawOpenat2Error.Store(true) // doesn't matter if we race here
		return false
	}
	_ = unix.Close(fd)
	return true
}
