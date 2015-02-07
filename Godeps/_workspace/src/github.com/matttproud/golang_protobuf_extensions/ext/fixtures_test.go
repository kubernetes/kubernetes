// Copyright 2010 The Go Authors.  All rights reserved.
// http://code.google.com/p/goprotobuf/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package ext

import (
	. "code.google.com/p/goprotobuf/proto"
	. "code.google.com/p/goprotobuf/proto/testdata"
)

// FROM https://code.google.com/p/goprotobuf/source/browse/proto/all_test.go.

func initGoTestField() *GoTestField {
	f := new(GoTestField)
	f.Label = String("label")
	f.Type = String("type")
	return f
}

// These are all structurally equivalent but the tag numbers differ.
// (It's remarkable that required, optional, and repeated all have
// 8 letters.)
func initGoTest_RequiredGroup() *GoTest_RequiredGroup {
	return &GoTest_RequiredGroup{
		RequiredField: String("required"),
	}
}

func initGoTest_OptionalGroup() *GoTest_OptionalGroup {
	return &GoTest_OptionalGroup{
		RequiredField: String("optional"),
	}
}

func initGoTest_RepeatedGroup() *GoTest_RepeatedGroup {
	return &GoTest_RepeatedGroup{
		RequiredField: String("repeated"),
	}
}

func initGoTest(setdefaults bool) *GoTest {
	pb := new(GoTest)
	if setdefaults {
		pb.F_BoolDefaulted = Bool(Default_GoTest_F_BoolDefaulted)
		pb.F_Int32Defaulted = Int32(Default_GoTest_F_Int32Defaulted)
		pb.F_Int64Defaulted = Int64(Default_GoTest_F_Int64Defaulted)
		pb.F_Fixed32Defaulted = Uint32(Default_GoTest_F_Fixed32Defaulted)
		pb.F_Fixed64Defaulted = Uint64(Default_GoTest_F_Fixed64Defaulted)
		pb.F_Uint32Defaulted = Uint32(Default_GoTest_F_Uint32Defaulted)
		pb.F_Uint64Defaulted = Uint64(Default_GoTest_F_Uint64Defaulted)
		pb.F_FloatDefaulted = Float32(Default_GoTest_F_FloatDefaulted)
		pb.F_DoubleDefaulted = Float64(Default_GoTest_F_DoubleDefaulted)
		pb.F_StringDefaulted = String(Default_GoTest_F_StringDefaulted)
		pb.F_BytesDefaulted = Default_GoTest_F_BytesDefaulted
		pb.F_Sint32Defaulted = Int32(Default_GoTest_F_Sint32Defaulted)
		pb.F_Sint64Defaulted = Int64(Default_GoTest_F_Sint64Defaulted)
	}

	pb.Kind = GoTest_TIME.Enum()
	pb.RequiredField = initGoTestField()
	pb.F_BoolRequired = Bool(true)
	pb.F_Int32Required = Int32(3)
	pb.F_Int64Required = Int64(6)
	pb.F_Fixed32Required = Uint32(32)
	pb.F_Fixed64Required = Uint64(64)
	pb.F_Uint32Required = Uint32(3232)
	pb.F_Uint64Required = Uint64(6464)
	pb.F_FloatRequired = Float32(3232)
	pb.F_DoubleRequired = Float64(6464)
	pb.F_StringRequired = String("string")
	pb.F_BytesRequired = []byte("bytes")
	pb.F_Sint32Required = Int32(-32)
	pb.F_Sint64Required = Int64(-64)
	pb.Requiredgroup = initGoTest_RequiredGroup()

	return pb
}
