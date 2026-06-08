// SPDX-License-Identifier: BSD-3-Clause

// Copyright (C) 2014-2015 Docker Inc & Go Authors. All rights reserved.
// Copyright (C) 2017-2024 SUSE LLC. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package securejoin implements a set of helpers to make it easier to write Go
// code that is safe against symlink-related escape attacks. The primary idea
// is to let you resolve a path within a rootfs directory as if the rootfs was
// a chroot.
//
// securejoin has two APIs, a "legacy" API and a "modern" API.
//
// The legacy API is [SecureJoin] and [SecureJoinVFS]. These methods are
// **not** safe against race conditions where an attacker changes the
// filesystem after (or during) the [SecureJoin] operation.
//
// The new API is available in the [pathrs-lite] subpackage, and provide
// protections against racing attackers as well as several other key
// protections against attacks often seen by container runtimes. As the name
// suggests, [pathrs-lite] is a stripped down (pure Go) reimplementation of
// [libpathrs]. The main APIs provided are [OpenInRoot], [MkdirAll], and
// [procfs.Handle] -- other APIs are not planned to be ported. The long-term
// goal is for users to migrate to [libpathrs] which is more fully-featured.
//
// securejoin has been used by several container runtimes (Docker, runc,
// Kubernetes, etc) for quite a few years as a de-facto standard for operating
// on container filesystem paths "safely". However, most users still use the
// legacy API which is unsafe against various attacks (there is a fairly long
// history of CVEs in dependent as a result). Users should switch to the modern
// API as soon as possible (or even better, switch to libpathrs).
//
// This project was initially intended to be included in the Go standard
// library, but it was rejected (see https://go.dev/issue/20126). Much later,
// [os.Root] was added to the Go stdlib that shares some of the goals of
// filepath-securejoin. However, its design is intended to work like
// openat2(RESOLVE_BENEATH) which does not fit the usecase of container
// runtimes and most system tools.
//
// [pathrs-lite]: https://pkg.go.dev/github.com/cyphar/filepath-securejoin/pathrs-lite
// [libpathrs]: https://github.com/openSUSE/libpathrs
// [OpenInRoot]: https://pkg.go.dev/github.com/cyphar/filepath-securejoin/pathrs-lite#OpenInRoot
// [MkdirAll]: https://pkg.go.dev/github.com/cyphar/filepath-securejoin/pathrs-lite#MkdirAll
// [procfs.Handle]: https://pkg.go.dev/github.com/cyphar/filepath-securejoin/pathrs-lite/procfs#Handle
// [os.Root]: https:///pkg.go.dev/os#Root
package securejoin
