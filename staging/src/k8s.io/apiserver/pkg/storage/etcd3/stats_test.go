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
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.etcd.io/etcd/api/v3/mvccpb"
)

func TestStatsCache(t *testing.T) {
	ctx := t.Context()
	store := newStatsCache("/prefix", func(ctx context.Context) ([]string, error) { return []string{}, nil })
	defer store.Close()

	stats, err := store.Stats(ctx)
	require.NoError(t, err)
	assert.Equal(t, int64(0), stats.EstimatedAverageObjectSizeBytes)

	store.UpdateKey(&mvccpb.KeyValue{Key: []byte("/prefix/foo1"), Value: []byte("0123456789"), ModRevision: 2})
	store.getKeys = func(ctx context.Context) ([]string, error) { return []string{"/prefix/foo1"}, nil }
	stats, err = store.Stats(ctx)
	require.NoError(t, err)
	assert.Equal(t, int64(10), stats.EstimatedAverageObjectSizeBytes)

	store.UpdateKey(&mvccpb.KeyValue{Key: []byte("/prefix/foo2"), Value: []byte("01234567890123456789"), ModRevision: 3})
	store.getKeys = func(ctx context.Context) ([]string, error) { return []string{"/prefix/foo1", "/prefix/foo2"}, nil }
	stats, err = store.Stats(ctx)
	require.NoError(t, err)
	assert.Equal(t, int64(15), stats.EstimatedAverageObjectSizeBytes)

	store.UpdateKey(&mvccpb.KeyValue{Key: []byte("/prefix/foo1"), Value: []byte("012345678901234567890123456789"), ModRevision: 4})
	stats, err = store.Stats(ctx)
	require.NoError(t, err)
	assert.Equal(t, int64(25), stats.EstimatedAverageObjectSizeBytes)

	store.UpdateKey(&mvccpb.KeyValue{Key: []byte("/prefix/foo2"), Value: []byte("0123456789"), ModRevision: 5})
	stats, err = store.Stats(ctx)
	require.NoError(t, err)
	assert.Equal(t, int64(20), stats.EstimatedAverageObjectSizeBytes)

	store.DeleteKey(&mvccpb.KeyValue{Key: []byte("/prefix/foo1"), ModRevision: 6})
	store.getKeys = func(ctx context.Context) ([]string, error) { return []string{"/prefix/foo2"}, nil }
	stats, err = store.Stats(ctx)
	require.NoError(t, err)
	assert.Equal(t, int64(10), stats.EstimatedAverageObjectSizeBytes)

	// Snapshot revision from revision 3
	store.Update([]*mvccpb.KeyValue{
		{Key: []byte("/prefix/foo1"), Value: []byte("0123456789"), ModRevision: 2},
		{Key: []byte("/prefix/foo2"), Value: []byte("01234567890123456789"), ModRevision: 3},
	})
	stats, err = store.Stats(ctx)
	require.NoError(t, err)
	assert.Equal(t, int64(10), stats.EstimatedAverageObjectSizeBytes)

	// Replay from revision 2
	store.UpdateKey(&mvccpb.KeyValue{Key: []byte("/prefix/foo1"), Value: []byte("0123456789"), ModRevision: 2})
	stats, err = store.Stats(ctx)
	require.NoError(t, err)
	assert.Equal(t, int64(10), stats.EstimatedAverageObjectSizeBytes)

	store.UpdateKey(&mvccpb.KeyValue{Key: []byte("/prefix/foo2"), Value: []byte("01234567890123456789"), ModRevision: 3})
	stats, err = store.Stats(ctx)
	require.NoError(t, err)
	assert.Equal(t, int64(10), stats.EstimatedAverageObjectSizeBytes)

	store.UpdateKey(&mvccpb.KeyValue{Key: []byte("/prefix/foo1"), Value: []byte("012345678901234567890123456789"), ModRevision: 4})
	stats, err = store.Stats(ctx)
	require.NoError(t, err)
	assert.Equal(t, int64(10), stats.EstimatedAverageObjectSizeBytes)

	store.UpdateKey(&mvccpb.KeyValue{Key: []byte("/prefix/foo2"), Value: []byte("0123456789"), ModRevision: 5})
	stats, err = store.Stats(ctx)
	require.NoError(t, err)
	assert.Equal(t, int64(10), stats.EstimatedAverageObjectSizeBytes)

	store.DeleteKey(&mvccpb.KeyValue{Key: []byte("/prefix/foo1"), ModRevision: 6})
	stats, err = store.Stats(ctx)
	require.NoError(t, err)
	assert.Equal(t, int64(10), stats.EstimatedAverageObjectSizeBytes)

	store.DeleteKey(&mvccpb.KeyValue{Key: []byte("/prefix/foo1"), ModRevision: 7})
	store.getKeys = func(ctx context.Context) ([]string, error) { return []string{}, nil }
	stats, err = store.Stats(ctx)
	require.NoError(t, err)
	assert.Equal(t, int64(0), stats.EstimatedAverageObjectSizeBytes)

	// Old snapshot might restore old revision if keys were recreated
	store.getKeys = func(ctx context.Context) ([]string, error) { return []string{"foo1", "foo2"}, nil }
	store.Update([]*mvccpb.KeyValue{
		{Key: []byte("/prefix/foo1"), Value: []byte("0123456789"), ModRevision: 2},
		{Key: []byte("/prefix/foo2"), Value: []byte("01234567890123456789"), ModRevision: 3},
	})
	stats, err = store.Stats(ctx)
	require.NoError(t, err)
	assert.Equal(t, int64(15), stats.EstimatedAverageObjectSizeBytes)

	// Cleanup if keys were deleted
	store.getKeys = func(ctx context.Context) ([]string, error) { return []string{}, nil }
	stats, err = store.Stats(ctx)
	require.NoError(t, err)
	assert.Equal(t, int64(0), stats.EstimatedAverageObjectSizeBytes)
}
