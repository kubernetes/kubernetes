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
	"testing"

	"github.com/coreos/etcd/mvcc/mvccpb"
)

func TestEvent(t *testing.T) {
	tests := []struct {
		ev       *Event
		isCreate bool
		isModify bool
	}{{
		ev: &Event{
			Type: EventTypePut,
			Kv: &mvccpb.KeyValue{
				CreateRevision: 3,
				ModRevision:    3,
			},
		},
		isCreate: true,
	}, {
		ev: &Event{
			Type: EventTypePut,
			Kv: &mvccpb.KeyValue{
				CreateRevision: 3,
				ModRevision:    4,
			},
		},
		isModify: false,
	}}
	for i, tt := range tests {
		if tt.isCreate && !tt.ev.IsCreate() {
			t.Errorf("#%d: event should be Create event", i)
		}
		if tt.isModify && !tt.ev.IsModify() {
			t.Errorf("#%d: event should be Modify event", i)
		}
	}
}
