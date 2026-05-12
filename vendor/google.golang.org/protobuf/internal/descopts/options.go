// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package descopts contains the nil pointers to concrete descriptor options.
//
// This package exists as a form of reverse dependency injection so that certain
// packages (e.g., internal/filedesc and internal/filetype can avoid a direct
// dependency on the descriptor proto package).
package descopts

import "google.golang.org/protobuf/reflect/protoreflect"

// These variables are set by the init function in descriptor.pb.go via logic
// in internal/filetype. In other words, so long as the descriptor proto package
// is linked in, these variables will be populated.
//
// Each variable is populated with a nil pointer to the options struct.
var (
	File           protoreflect.ProtoMessage
	Enum           protoreflect.ProtoMessage
	EnumValue      protoreflect.ProtoMessage
	Message        protoreflect.ProtoMessage
	Field          protoreflect.ProtoMessage
	Oneof          protoreflect.ProtoMessage
	ExtensionRange protoreflect.ProtoMessage
	Service        protoreflect.ProtoMessage
	Method         protoreflect.ProtoMessage
)
