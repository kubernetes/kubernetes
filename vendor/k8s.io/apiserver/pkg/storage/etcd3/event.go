/*
Copyright 2016 The Kubernetes Authors.

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

package etcd3

import (
	"fmt"
	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/mvcc/mvccpb"
)

type event struct {
	key       string
	value     []byte
	prevValue []byte
	rev       int64
	isDeleted bool
	isCreated bool
}

// parseKV converts a KeyValue retrieved from an initial sync() listing to a synthetic isCreated event.
func parseKV(kv *mvccpb.KeyValue) *event {
	return &event{
		key:       string(kv.Key),
		value:     kv.Value,
		prevValue: nil,
		rev:       kv.ModRevision,
		isDeleted: false,
		isCreated: true,
	}
}

func parseEvent(e *clientv3.Event) (*event, error) {
	if !e.IsCreate() && e.PrevKv == nil {
		// If the previous value is nil, error. One example of how this is possible is if the previous value has been compacted already.
		return nil, fmt.Errorf("etcd event received with PrevKv=nil (key=%q, modRevision=%d, type=%s)", string(e.Kv.Key), e.Kv.ModRevision, e.Type.String())

	}
	ret := &event{
		key:       string(e.Kv.Key),
		value:     e.Kv.Value,
		rev:       e.Kv.ModRevision,
		isDeleted: e.Type == clientv3.EventTypeDelete,
		isCreated: e.IsCreate(),
	}
	if e.PrevKv != nil {
		ret.prevValue = e.PrevKv.Value
	}
	return ret, nil
}
