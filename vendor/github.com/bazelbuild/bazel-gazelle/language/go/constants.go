/* Copyright 2018 The Bazel Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package golang

const (
	// defaultLibName is the name of the default go_library rule in a Go
	// package directory. This name was originally chosen so that rules_go
	// could translate between Bazel labels and Go import paths using go_prefix.
	// It is no longer needed since go_prefix was deleted.
	defaultLibName = "go_default_library"

	// defaultTestName is a name of an internal test corresponding to
	// defaultLibName. It does not need to be consistent to something but it
	// just needs to be unique in the Bazel package
	defaultTestName = "go_default_test"

	// legacyProtoFilegroupName is the anme of a filegroup created in legacy
	// mode for libraries that contained .pb.go files and .proto files.
	legacyProtoFilegroupName = "go_default_library_protos"

	// grpcCompilerLabel is the label for the gRPC compiler plugin, used in the
	// "compilers" attribute of go_proto_library rules.
	grpcCompilerLabel = "@io_bazel_rules_go//proto:go_grpc"

	// wellKnownTypesGoPrefix is the import path for the Go repository containing
	// pre-generated code for the Well Known Types.
	wellKnownTypesGoPrefix = "github.com/golang/protobuf"

	// wellKnownTypesPkg is the package name for the predefined WKTs in rules_go.
	wellKnownTypesPkg = "proto/wkt"
)
