// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package editionssupport defines constants for editions that are supported.
package editionssupport

import descriptorpb "google.golang.org/protobuf/types/descriptorpb"

const (
	Minimum = descriptorpb.Edition_EDITION_PROTO2
	Maximum = descriptorpb.Edition_EDITION_2023
)
