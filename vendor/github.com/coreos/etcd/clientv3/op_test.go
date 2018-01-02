// Copyright 2016 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package clientv3

import (
	"reflect"
	"testing"

	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
)

// TestOpWithSort tests if WithSort(ASCEND, KEY) and WithLimit are specified,
// RangeRequest ignores the SortOption to avoid unnecessarily fetching
// the entire key-space.
func TestOpWithSort(t *testing.T) {
	opReq := OpGet("foo", WithSort(SortByKey, SortAscend), WithLimit(10)).toRequestOp().Request
	q, ok := opReq.(*pb.RequestOp_RequestRange)
	if !ok {
		t.Fatalf("expected range request, got %v", reflect.TypeOf(opReq))
	}
	req := q.RequestRange
	wreq := &pb.RangeRequest{Key: []byte("foo"), SortOrder: pb.RangeRequest_NONE, Limit: 10}
	if !reflect.DeepEqual(req, wreq) {
		t.Fatalf("expected %+v, got %+v", wreq, req)
	}
}
