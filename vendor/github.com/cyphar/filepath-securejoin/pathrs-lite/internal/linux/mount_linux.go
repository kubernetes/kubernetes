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
	"github.com/cyphar/filepath-securejoin/pathrs-lite/internal/kernelversion"
)

// HasNewMountAPI returns whether the new fsopen(2) mount API is supported on
// the running kernel.
var HasNewMountAPI = gocompat.SyncOnceValue(func() bool {
	// All of the pieces of the new mount API we use (fsopen, fsconfig,
	// fsmount, open_tree) were added together in Linux 5.2[1,2], so we can
	// just check for one of the syscalls and the others should also be
	// available.
	//
	// Just try to use open_tree(2) to open a file without OPEN_TREE_CLONE.
	// This is equivalent to openat(2), but tells us if open_tree is
	// available (and thus all of the other basic new mount API syscalls).
	// open_tree(2) is most light-weight syscall to test here.
	//
	// [1]: merge commit 400913252d09
	// [2]: <https://lore.kernel.org/lkml/153754740781.17872.7869536526927736855.stgit@warthog.procyon.org.uk/>
	fd, err := unix.OpenTree(-int(unix.EBADF), "/", unix.OPEN_TREE_CLOEXEC)
	if err != nil {
		return false
	}
	_ = unix.Close(fd)

	// RHEL 8 has a backport of fsopen(2) that appears to have some very
	// difficult to debug performance pathology. As such, it seems prudent to
	// simply reject pre-5.2 kernels.
	isNotBackport, _ := kernelversion.GreaterEqualThan(kernelversion.KernelVersion{5, 2})
	return isNotBackport
})
