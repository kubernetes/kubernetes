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
	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/mvcc/mvccpb"
)

type event struct {
	key       string
	value     []byte
	rev       int64
	isDeleted bool
	isCreated bool
}

func parseKV(kv *mvccpb.KeyValue) *event {
	return &event{
		key:       string(kv.Key),
		value:     kv.Value,
		rev:       kv.ModRevision,
		isDeleted: false,
		isCreated: kv.ModRevision == kv.CreateRevision,
	}
}

func parseEvent(e *clientv3.Event) *event {
	return &event{
		key:       string(e.Kv.Key),
		value:     e.Kv.Value,
		rev:       e.Kv.ModRevision,
		isDeleted: e.Type == clientv3.EventTypeDelete,
		isCreated: e.IsCreate(),
	}
}
