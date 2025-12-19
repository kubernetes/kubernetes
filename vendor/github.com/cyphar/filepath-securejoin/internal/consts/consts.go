// SPDX-License-Identifier: BSD-3-Clause

// Copyright (C) 2014-2015 Docker Inc & Go Authors. All rights reserved.
// Copyright (C) 2017-2025 SUSE LLC. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package consts contains the definitions of internal constants used
// throughout filepath-securejoin.
package consts

// MaxSymlinkLimit is the maximum number of symlinks that can be encountered
// during a single lookup before returning -ELOOP. At time of writing, Linux
// has an internal limit of 40.
const MaxSymlinkLimit = 255
