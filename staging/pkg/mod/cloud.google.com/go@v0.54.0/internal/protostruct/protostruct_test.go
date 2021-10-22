// Copyright 2017 Google LLC
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

// Package protostruct supports operations on the protocol buffer Struct message.
package protostruct

import (
	"testing"

	"cloud.google.com/go/internal/testutil"
	pb "github.com/golang/protobuf/ptypes/struct"
)

func TestDecodeToMap(t *testing.T) {
	if got := DecodeToMap(nil); !testutil.Equal(got, map[string]interface{}(nil)) {
		t.Errorf("DecodeToMap(nil) = %v, want nil", got)
	}
	nullv := &pb.Value{Kind: &pb.Value_NullValue{}}
	stringv := &pb.Value{Kind: &pb.Value_StringValue{"x"}}
	boolv := &pb.Value{Kind: &pb.Value_BoolValue{true}}
	numberv := &pb.Value{Kind: &pb.Value_NumberValue{2.7}}
	in := &pb.Struct{Fields: map[string]*pb.Value{
		"n": nullv,
		"s": stringv,
		"b": boolv,
		"f": numberv,
		"l": {Kind: &pb.Value_ListValue{&pb.ListValue{
			Values: []*pb.Value{nullv, stringv, boolv, numberv},
		}}},
		"S": {Kind: &pb.Value_StructValue{&pb.Struct{Fields: map[string]*pb.Value{
			"n1": nullv,
			"b1": boolv,
		}}}},
	}}
	want := map[string]interface{}{
		"n": nil,
		"s": "x",
		"b": true,
		"f": 2.7,
		"l": []interface{}{nil, "x", true, 2.7},
		"S": map[string]interface{}{"n1": nil, "b1": true},
	}
	got := DecodeToMap(in)
	if diff := testutil.Diff(got, want); diff != "" {
		t.Error(diff)
	}
}
