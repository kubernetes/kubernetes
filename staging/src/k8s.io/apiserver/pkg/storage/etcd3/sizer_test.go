/*
Copyright 2025 The Kubernetes Authors.

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
	"testing"

	"github.com/stretchr/testify/assert"
	"go.etcd.io/etcd/api/v3/mvccpb"
)

func TestSizeCache(t *testing.T) {
	store := newSizeCache()
	assert.Equal(t, int64(0), store.AverageObjectSize([]*mvccpb.KeyValue{}))

	store.AddOrUpdate("foo1", 2, 10)
	assert.Equal(t, int64(10), store.AverageObjectSize([]*mvccpb.KeyValue{{Key: []byte("foo1")}}))

	store.AddOrUpdate("foo2", 3, 20)
	assert.Equal(t, int64(15), store.AverageObjectSize([]*mvccpb.KeyValue{{Key: []byte("foo1")}, {Key: []byte("foo2")}}))

	store.AddOrUpdate("foo1", 4, 100)
	assert.Equal(t, int64(60), store.AverageObjectSize([]*mvccpb.KeyValue{{Key: []byte("foo1")}, {Key: []byte("foo2")}}))

	store.AddOrUpdate("foo2", 5, 200)
	assert.Equal(t, int64(150), store.AverageObjectSize([]*mvccpb.KeyValue{{Key: []byte("foo1")}, {Key: []byte("foo2")}}))

	store.Delete("foo1", 5)
	assert.Equal(t, int64(200), store.AverageObjectSize([]*mvccpb.KeyValue{{Key: []byte("foo2")}}))

	// Replay from revision 2
	store.AddOrUpdate("foo1", 2, 10)
	assert.Equal(t, int64(200), store.AverageObjectSize([]*mvccpb.KeyValue{{Key: []byte("foo2")}}))

	store.AddOrUpdate("foo2", 3, 20)
	assert.Equal(t, int64(200), store.AverageObjectSize([]*mvccpb.KeyValue{{Key: []byte("foo2")}}))

	store.AddOrUpdate("foo1", 4, 100)
	assert.Equal(t, int64(200), store.AverageObjectSize([]*mvccpb.KeyValue{{Key: []byte("foo2")}}))

	store.AddOrUpdate("foo2", 5, 200)
	assert.Equal(t, int64(200), store.AverageObjectSize([]*mvccpb.KeyValue{{Key: []byte("foo2")}}))

	store.Delete("foo1", 5)
	assert.Equal(t, int64(200), store.AverageObjectSize([]*mvccpb.KeyValue{{Key: []byte("foo2")}}))

	store.Delete("foo2", 6)
	assert.Equal(t, int64(0), store.AverageObjectSize([]*mvccpb.KeyValue{}))
}
