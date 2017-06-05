// Copyright 2016 Google Inc. All Rights Reserved.
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

package datastore

import (
	"testing"

	pb "google.golang.org/genproto/googleapis/datastore/v1"
)

func TestInterfaceToProtoNilKey(t *testing.T) {
	var iv *Key
	pv, err := interfaceToProto(iv, false)
	if err != nil {
		t.Fatalf("nil key: interfaceToProto: %v", err)
	}

	_, ok := pv.ValueType.(*pb.Value_NullValue)
	if !ok {
		t.Errorf("nil key: type:\ngot: %T\nwant: %T", pv.ValueType, &pb.Value_NullValue{})
	}
}
