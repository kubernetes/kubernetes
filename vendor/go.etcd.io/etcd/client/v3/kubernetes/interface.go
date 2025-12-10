// Copyright 2024 The etcd Authors
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

package kubernetes

import (
	"context"

	"go.etcd.io/etcd/api/v3/mvccpb"
	clientv3 "go.etcd.io/etcd/client/v3"
)

// Interface defines the minimal client-side interface that Kubernetes requires
// to interact with etcd. Methods below are standard etcd operations with
// semantics adjusted to better suit Kubernetes' needs.
type Interface interface {
	// Get retrieves a single key-value pair from etcd.
	//
	// If opts.Revision is set to a non-zero value, the key-value pair is retrieved at the specified revision.
	// If the required revision has been compacted, the request will fail with ErrCompacted.
	Get(ctx context.Context, key string, opts GetOptions) (GetResponse, error)

	// List retrieves key-value pairs with the specified prefix, ordered lexicographically by key.
	//
	// If opts.Revision is non-zero, the key-value pairs are retrieved at the specified revision.
	// If the required revision has been compacted, the request will fail with ErrCompacted.
	// If opts.Limit is greater than zero, the number of returned key-value pairs is bounded by the limit.
	// If opts.Continue is not empty, the listing will start from the key
	// specified by it. When paginating, the Continue value should be set
	// to the last observed key with "\x00" appended to it.
	List(ctx context.Context, prefix string, opts ListOptions) (ListResponse, error)

	// Count returns the number of keys with the specified prefix.
	//
	// Currently, there are no options for the Count operation. However, a placeholder options struct (CountOptions)
	// is provided for future extensibility in case options become necessary.
	Count(ctx context.Context, prefix string, opts CountOptions) (int64, error)

	// OptimisticPut creates or updates a key-value pair if the key has not been modified or created
	// since the revision specified in expectedRevision.
	//
	// An OptimisticPut fails if the key has been modified since expectedRevision.
	OptimisticPut(ctx context.Context, key string, value []byte, expectedRevision int64, opts PutOptions) (PutResponse, error)

	// OptimisticDelete deletes the key-value pair if it hasn't been modified since the revision
	// specified in expectedRevision.
	//
	// An OptimisticDelete fails if the key has been modified since expectedRevision.
	OptimisticDelete(ctx context.Context, key string, expectedRevision int64, opts DeleteOptions) (DeleteResponse, error)
}

type GetOptions struct {
	// Revision is the point-in-time of the etcd key-value store to use for the Get operation.
	// If Revision is 0, it gets the latest value.
	Revision int64
}

type ListOptions struct {
	// Revision is the point-in-time of the etcd key-value store to use for the List operation.
	// If Revision is 0, it gets the latest values.
	Revision int64

	// Limit is the maximum number of keys to return for a List operation.
	// 0 means no limitation.
	Limit int64

	// Continue is a key from which to resume the List operation.
	// It should be set to the last key from a previous ListResponse
	// with "\x00" appended to it when paginating.
	Continue string
}

// CountOptions is a placeholder for potential future options for the Count operation.
type CountOptions struct{}

type PutOptions struct {
	// GetOnFailure specifies whether to return the modified key-value pair if the Put operation fails due to a revision mismatch.
	GetOnFailure bool

	// LeaseID is the ID of a lease to associate with the key allowing for automatic deletion after lease expires after it's TTL (time to live).
	// Deprecated: Should be replaced with TTL when Interface starts using one lease per object.
	LeaseID clientv3.LeaseID
}

type DeleteOptions struct {
	// GetOnFailure specifies whether to return the modified key-value pair if the Delete operation fails due to a revision mismatch.
	GetOnFailure bool
}

type GetResponse struct {
	// KV is the key-value pair retrieved from etcd.
	KV *mvccpb.KeyValue

	// Revision is the revision of the key-value store at the time of the Get operation.
	Revision int64
}

type ListResponse struct {
	// Kvs is the list of key-value pairs retrieved from etcd, ordered lexicographically by key.
	Kvs []*mvccpb.KeyValue

	// Count is the total number of keys with the specified prefix, even if not all were returned due to a limit.
	Count int64

	// Revision is the revision of the key-value store at the time of the List operation.
	Revision int64
}

type PutResponse struct {
	// KV is the created or updated key-value pair. If the Put operation failed and GetOnFailure was true, this
	// will be the modified key-value pair that caused the failure.
	KV *mvccpb.KeyValue

	// Succeeded indicates whether the Put operation was successful.
	Succeeded bool

	// Revision is the revision of the key-value store after the Put operation.
	Revision int64
}

type DeleteResponse struct {
	// KV is the deleted key-value pair. If the Delete operation failed and GetOnFailure was true, this
	// will be the modified key-value pair that caused the failure.
	KV *mvccpb.KeyValue

	// Succeeded indicates whether the Delete operation was successful.
	Succeeded bool

	// Revision is the revision of the key-value store after the Delete operation.
	Revision int64
}
