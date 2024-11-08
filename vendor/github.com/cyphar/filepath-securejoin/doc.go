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
// The new API is made up of [OpenInRoot] and [MkdirAll] (and derived
// functions). These are safe against racing attackers and have several other
// protections that are not provided by the legacy API. There are many more
// operations that most programs expect to be able to do safely, but we do not
// provide explicit support for them because we want to encourage users to
// switch to [libpathrs](https://github.com/openSUSE/libpathrs) which is a
// cross-language next-generation library that is entirely designed around
// operating on paths safely.
//
// securejoin has been used by several container runtimes (Docker, runc,
// Kubernetes, etc) for quite a few years as a de-facto standard for operating
// on container filesystem paths "safely". However, most users still use the
// legacy API which is unsafe against various attacks (there is a fairly long
// history of CVEs in dependent as a result). Users should switch to the modern
// API as soon as possible (or even better, switch to libpathrs).
//
// This project was initially intended to be included in the Go standard
// library, but [it was rejected](https://go.dev/issue/20126). There is now a
// [new Go proposal](https://go.dev/issue/67002) for a safe path resolution API
// that shares some of the goals of filepath-securejoin. However, that design
// is intended to work like `openat2(RESOLVE_BENEATH)` which does not fit the
// usecase of container runtimes and most system tools.
package securejoin
