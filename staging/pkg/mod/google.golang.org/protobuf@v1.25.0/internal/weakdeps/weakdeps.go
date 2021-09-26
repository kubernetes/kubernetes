// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build weak_dependency

package weakdeps

import (
	// Ensure that any program using "github.com/golang/protobuf"
	// uses a version that wraps this module so that there is a
	// unified view on what protobuf types are globally registered.
	_ "github.com/golang/protobuf/proto"

	// Ensure that any program using "google.golang.org/genproto"
	// uses a version that forwards their generated well-known types
	// to reference the ones declared in this module.
	_ "google.golang.org/genproto/protobuf/api"
	_ "google.golang.org/genproto/protobuf/field_mask"
	_ "google.golang.org/genproto/protobuf/ptype"
	_ "google.golang.org/genproto/protobuf/source_context"
)
