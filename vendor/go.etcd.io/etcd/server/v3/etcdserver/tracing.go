// Copyright 2025 The etcd Authors
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

package etcdserver

import pb "go.etcd.io/etcd/api/v3/etcdserverpb"

// firstCompareKey returns first non-empty key in the list of comparison operations.
func firstCompareKey(c []*pb.Compare) string {
	for _, op := range c {
		key := string(op.GetKey())
		if key != "" {
			return key
		}
	}
	return ""
}

// firstOpKey returns first non-empty key in the list of request operations.
func firstOpKey(ops []*pb.RequestOp) string {
	for _, operation := range ops {
		var key string
		switch op := operation.GetRequest().(type) {
		case *pb.RequestOp_RequestPut:
			key = string(op.RequestPut.GetKey())
		case *pb.RequestOp_RequestRange:
			key = string(op.RequestRange.GetKey())
		case *pb.RequestOp_RequestDeleteRange:
			key = string(op.RequestDeleteRange.GetKey())
		}
		if key != "" {
			return key
		}
	}
	return ""
}

// firstOpType returns type of the first operation in the list.
func firstOpType(ops []*pb.RequestOp) string {
	for _, operation := range ops {
		switch operation.GetRequest().(type) {
		case *pb.RequestOp_RequestPut:
			return "put"
		case *pb.RequestOp_RequestRange:
			return "range"
		case *pb.RequestOp_RequestDeleteRange:
			return "delete_range"
		case *pb.RequestOp_RequestTxn:
			return "txn"
		}
	}
	return ""
}

// firstOpLease returns lease ID of the first PUT operation in the list.
func firstOpLease(ops []*pb.RequestOp) int64 {
	for _, operation := range ops {
		if op, ok := operation.GetRequest().(*pb.RequestOp_RequestPut); ok {
			return op.RequestPut.GetLease()
		}
	}
	return -1
}
