// SPDX-License-Identifier: BSD-3-Clause

// Copyright (C) 2022 The Go Authors. All rights reserved.
// Copyright (C) 2025 SUSE LLC. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE.BSD file.

// The parsing logic is very loosely based on the Go stdlib's
// src/internal/syscall/unix/kernel_version_linux.go but with an API that looks
// a bit like runc's libcontainer/system/kernelversion.
//
// TODO(cyphar): This API has been copied around to a lot of different projects
// (Docker, containerd, runc, and now filepath-securejoin) -- maybe we should
// put it in a separate project?

// Package kernelversion provides a simple mechanism for checking whether the
// running kernel is at least as new as some baseline kernel version. This is
// often useful when checking for features that would be too complicated to
// test support for (or in cases where we know that some kernel features in
// backport-heavy kernels are broken and need to be avoided).
package kernelversion

import (
	"bytes"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"golang.org/x/sys/unix"

	"github.com/cyphar/filepath-securejoin/pathrs-lite/internal/gocompat"
)

// KernelVersion is a numeric representation of the key numerical elements of a
// kernel version (for instance, "4.1.2-default-1" would be represented as
// KernelVersion{4, 1, 2}).
type KernelVersion []uint64

func (kver KernelVersion) String() string {
	var str strings.Builder
	for idx, elem := range kver {
		if idx != 0 {
			_, _ = str.WriteRune('.')
		}
		_, _ = str.WriteString(strconv.FormatUint(elem, 10))
	}
	return str.String()
}

var errInvalidKernelVersion = errors.New("invalid kernel version")

// parseKernelVersion parses a string and creates a KernelVersion based on it.
func parseKernelVersion(kverStr string) (KernelVersion, error) {
	kver := make(KernelVersion, 1, 3)
	for idx, ch := range kverStr {
		if '0' <= ch && ch <= '9' {
			v := &kver[len(kver)-1]
			*v = (*v * 10) + uint64(ch-'0')
		} else {
			if idx == 0 || kverStr[idx-1] < '0' || '9' < kverStr[idx-1] {
				// "." must be preceded by a digit while in version section
				return nil, fmt.Errorf("%w %q: kernel version has dot(s) followed by non-digit in version section", errInvalidKernelVersion, kverStr)
			}
			if ch != '.' {
				break
			}
			kver = append(kver, 0)
		}
	}
	if len(kver) < 2 {
		return nil, fmt.Errorf("%w %q: kernel versions must contain at least two components", errInvalidKernelVersion, kverStr)
	}
	return kver, nil
}

// getKernelVersion gets the current kernel version.
var getKernelVersion = gocompat.SyncOnceValues(func() (KernelVersion, error) {
	var uts unix.Utsname
	if err := unix.Uname(&uts); err != nil {
		return nil, err
	}
	// Remove the \x00 from the release.
	release := uts.Release[:]
	return parseKernelVersion(string(release[:bytes.IndexByte(release, 0)]))
})

// GreaterEqualThan returns true if the the host kernel version is greater than
// or equal to the provided [KernelVersion]. When doing this comparison, any
// non-numerical suffixes of the host kernel version are ignored.
//
// If the number of components provided is not equal to the number of numerical
// components of the host kernel version, any missing components are treated as
// 0. This means that GreaterEqualThan(KernelVersion{4}) will be treated the
// same as GreaterEqualThan(KernelVersion{4, 0, 0, ..., 0, 0}), and that if the
// host kernel version is "4" then GreaterEqualThan(KernelVersion{4, 1}) will
// return false (because the host version will be treated as "4.0").
func GreaterEqualThan(wantKver KernelVersion) (bool, error) {
	hostKver, err := getKernelVersion()
	if err != nil {
		return false, err
	}

	// Pad out the kernel version lengths to match one another.
	cmpLen := gocompat.Max2(len(hostKver), len(wantKver))
	hostKver = append(hostKver, make(KernelVersion, cmpLen-len(hostKver))...)
	wantKver = append(wantKver, make(KernelVersion, cmpLen-len(wantKver))...)

	for i := 0; i < cmpLen; i++ {
		switch gocompat.CmpCompare(hostKver[i], wantKver[i]) {
		case -1:
			// host < want
			return false, nil
		case +1:
			// host > want
			return true, nil
		case 0:
			continue
		}
	}
	// equal version values
	return true, nil
}
