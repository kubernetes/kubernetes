// SPDX-License-Identifier: BSD-3-Clause
//go:build linux && go1.20

// Copyright (C) 2025 SUSE LLC. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gocompat includes compatibility shims (backported from future Go
// stdlib versions) to permit filepath-securejoin to be used with older Go
// versions (often filepath-securejoin is added in security patches for old
// releases, so avoiding the need to bump Go compiler requirements is a huge
// plus to downstreams).
package gocompat
