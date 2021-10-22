// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package ptype aliases all exported identifiers in
// package "google.golang.org/protobuf/types/known/typepb".
package ptype

import "google.golang.org/protobuf/types/known/typepb"

type Syntax = typepb.Syntax

const (
	Syntax_SYNTAX_PROTO2 = typepb.Syntax_SYNTAX_PROTO2
	Syntax_SYNTAX_PROTO3 = typepb.Syntax_SYNTAX_PROTO3
)

var (
	Syntax_name  = typepb.Syntax_name
	Syntax_value = typepb.Syntax_value
)

type Field_Kind = typepb.Field_Kind

const (
	Field_TYPE_UNKNOWN  = typepb.Field_TYPE_UNKNOWN
	Field_TYPE_DOUBLE   = typepb.Field_TYPE_DOUBLE
	Field_TYPE_FLOAT    = typepb.Field_TYPE_FLOAT
	Field_TYPE_INT64    = typepb.Field_TYPE_INT64
	Field_TYPE_UINT64   = typepb.Field_TYPE_UINT64
	Field_TYPE_INT32    = typepb.Field_TYPE_INT32
	Field_TYPE_FIXED64  = typepb.Field_TYPE_FIXED64
	Field_TYPE_FIXED32  = typepb.Field_TYPE_FIXED32
	Field_TYPE_BOOL     = typepb.Field_TYPE_BOOL
	Field_TYPE_STRING   = typepb.Field_TYPE_STRING
	Field_TYPE_GROUP    = typepb.Field_TYPE_GROUP
	Field_TYPE_MESSAGE  = typepb.Field_TYPE_MESSAGE
	Field_TYPE_BYTES    = typepb.Field_TYPE_BYTES
	Field_TYPE_UINT32   = typepb.Field_TYPE_UINT32
	Field_TYPE_ENUM     = typepb.Field_TYPE_ENUM
	Field_TYPE_SFIXED32 = typepb.Field_TYPE_SFIXED32
	Field_TYPE_SFIXED64 = typepb.Field_TYPE_SFIXED64
	Field_TYPE_SINT32   = typepb.Field_TYPE_SINT32
	Field_TYPE_SINT64   = typepb.Field_TYPE_SINT64
)

var (
	Field_Kind_name  = typepb.Field_Kind_name
	Field_Kind_value = typepb.Field_Kind_value
)

type Field_Cardinality = typepb.Field_Cardinality

const (
	Field_CARDINALITY_UNKNOWN  = typepb.Field_CARDINALITY_UNKNOWN
	Field_CARDINALITY_OPTIONAL = typepb.Field_CARDINALITY_OPTIONAL
	Field_CARDINALITY_REQUIRED = typepb.Field_CARDINALITY_REQUIRED
	Field_CARDINALITY_REPEATED = typepb.Field_CARDINALITY_REPEATED
)

var (
	Field_Cardinality_name  = typepb.Field_Cardinality_name
	Field_Cardinality_value = typepb.Field_Cardinality_value
)

type Type = typepb.Type
type Field = typepb.Field
type Enum = typepb.Enum
type EnumValue = typepb.EnumValue
type Option = typepb.Option

var File_google_protobuf_type_proto = typepb.File_google_protobuf_type_proto
