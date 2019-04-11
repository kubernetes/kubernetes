// Copyright 2011 Google Inc.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The uuid package generates and inspects UUIDs.
//
// UUIDs are based on RFC 4122 and DCE 1.1: Authentication and Security
// Services.
//
// This package is a partial wrapper around the github.com/google/uuid package.
// This package represents a UUID as []byte while github.com/google/uuid
// represents a UUID as [16]byte.
package uuid
