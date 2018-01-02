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

	"github.com/coreos/etcd/etcdserver/etcdserverpb"
)

func TestCompactOp(t *testing.T) {
	req1 := OpCompact(100, WithCompactPhysical()).toRequest()
	req2 := &etcdserverpb.CompactionRequest{Revision: 100, Physical: true}
	if !reflect.DeepEqual(req1, req2) {
		t.Fatalf("expected %+v, got %+v", req2, req1)
	}
}
