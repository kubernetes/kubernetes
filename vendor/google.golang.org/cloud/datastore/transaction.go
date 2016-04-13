// Copyright 2014 Google Inc. All Rights Reserved.
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
	"errors"
	"net/http"

	"github.com/golang/protobuf/proto"
	"golang.org/x/net/context"

	pb "google.golang.org/cloud/internal/datastore"
	"google.golang.org/cloud/internal/transport"
)

// ErrConcurrentTransaction is returned when a transaction is rolled back due
// to a conflict with a concurrent transaction.
var ErrConcurrentTransaction = errors.New("datastore: concurrent transaction")

var errExpiredTransaction = errors.New("datastore: transaction expired")

// A TransactionOption configures the Transaction returned by NewTransaction.
type TransactionOption interface {
	apply(*pb.BeginTransactionRequest)
}

type isolation struct {
	level pb.BeginTransactionRequest_IsolationLevel
}

func (i isolation) apply(req *pb.BeginTransactionRequest) {
	req.IsolationLevel = i.level.Enum()
}

var (
	// Snapshot causes the transaction to enforce a snapshot isolation level.
	Snapshot TransactionOption = isolation{pb.BeginTransactionRequest_SNAPSHOT}
	// Serializable causes the transaction to enforce a serializable isolation level.
	Serializable TransactionOption = isolation{pb.BeginTransactionRequest_SERIALIZABLE}
)

// Transaction represents a set of datastore operations to be committed atomically.
//
// Operations are enqueued by calling the Put and Delete methods on Transaction
// (or their Multi-equivalents).  These operations are only committed when the
// Commit method is invoked. To ensure consistency, reads must be performed by
// using Transaction's Get method or by using the Transaction method when
// building a query.
//
// A Transaction must be committed or rolled back exactly once.
type Transaction struct {
	id       []byte
	client   *Client
	ctx      context.Context
	mutation *pb.Mutation  // The mutations to apply.
	pending  []*PendingKey // Incomplete keys pending transaction completion.
}

// NewTransaction starts a new transaction.
func (c *Client) NewTransaction(ctx context.Context, opts ...TransactionOption) (*Transaction, error) {
	req, resp := &pb.BeginTransactionRequest{}, &pb.BeginTransactionResponse{}
	for _, o := range opts {
		o.apply(req)
	}
	if err := c.call(ctx, "beginTransaction", req, resp); err != nil {
		return nil, err
	}

	return &Transaction{
		id:       resp.Transaction,
		ctx:      ctx,
		client:   c,
		mutation: &pb.Mutation{},
	}, nil
}

// RunInTransaction runs f in a transaction. f is invoked with a Transaction
// that f should use for all the transaction's datastore operations.
//
// f must not call Commit or Rollback on the provided Transaction.
//
// If f returns nil, RunInTransaction commits the transaction,
// returning the Commit and a nil error if it succeeds. If the commit fails due
// to a conflicting transaction, RunInTransaction retries f with a new
// Transaction. It gives up and returns ErrConcurrentTransaction after three
// failed attempts.
//
// If f returns non-nil, then the transaction will be rolled back and
// RunInTransaction will return the same error. The function f is not retried.
//
// Note that when f returns, the transaction is not committed. Calling code
// must not assume that any of f's changes have been committed until
// RunInTransaction returns nil.
//
// Since f may be called multiple times, f should usually be idempotent.
// Note that Transaction.Get is not idempotent when unmarshaling slice fields.
func (c *Client) RunInTransaction(ctx context.Context, f func(tx *Transaction) error, opts ...TransactionOption) (*Commit, error) {
	// TODO(djd): Allow configuring of attempts.
	const attempts = 3
	for n := 0; n < attempts; n++ {
		tx, err := c.NewTransaction(ctx, opts...)
		if err != nil {
			return nil, err
		}
		if err := f(tx); err != nil {
			tx.Rollback()
			return nil, err
		}
		if cmt, err := tx.Commit(); err != ErrConcurrentTransaction {
			return cmt, err
		}
	}
	return nil, ErrConcurrentTransaction
}

// Commit applies the enqueued operations atomically.
func (t *Transaction) Commit() (*Commit, error) {
	if t.id == nil {
		return nil, errExpiredTransaction
	}
	req := &pb.CommitRequest{
		Transaction: t.id,
		Mutation:    t.mutation,
		Mode:        pb.CommitRequest_TRANSACTIONAL.Enum(),
	}
	t.id = nil
	resp := &pb.CommitResponse{}
	if err := t.client.call(t.ctx, "commit", req, resp); err != nil {
		if e, ok := err.(*transport.ErrHTTP); ok && e.StatusCode == http.StatusConflict {
			// TODO(jbd): Make sure that we explicitly handle the case where response
			// has an HTTP 409 and the error message indicates that it's an concurrent
			// transaction error.
			return nil, ErrConcurrentTransaction
		}
		return nil, err
	}

	// Copy any newly minted keys into the returned keys.
	if len(t.pending) != len(resp.MutationResult.InsertAutoIdKey) {
		return nil, errors.New("datastore: internal error: server returned the wrong number of keys")
	}
	commit := &Commit{}
	for i, p := range t.pending {
		key, err := protoToKey(resp.MutationResult.InsertAutoIdKey[i])
		if err != nil {
			return nil, errors.New("datastore: internal error: server returned an invalid key")
		}
		p.key = key
		p.commit = commit
	}

	return commit, nil
}

// Rollback abandons a pending transaction.
func (t *Transaction) Rollback() error {
	if t.id == nil {
		return errExpiredTransaction
	}
	id := t.id
	t.id = nil
	return t.client.call(t.ctx, "rollback", &pb.RollbackRequest{Transaction: id}, &pb.RollbackResponse{})
}

// Get is the transaction-specific version of the package function Get.
// All reads performed during the transaction will come from a single consistent
// snapshot. Furthermore, if the transaction is set to a serializable isolation
// level, another transaction cannot concurrently modify the data that is read
// or modified by this transaction.
func (t *Transaction) Get(key *Key, dst interface{}) error {
	err := t.client.get(t.ctx, []*Key{key}, []interface{}{dst}, &pb.ReadOptions{Transaction: t.id})
	if me, ok := err.(MultiError); ok {
		return me[0]
	}
	return err
}

// GetMulti is a batch version of Get.
func (t *Transaction) GetMulti(keys []*Key, dst interface{}) error {
	if t.id == nil {
		return errExpiredTransaction
	}
	return t.client.get(t.ctx, keys, dst, &pb.ReadOptions{Transaction: t.id})
}

// Put is the transaction-specific version of the package function Put.
//
// Put returns a PendingKey which can be resolved into a Key using the
// return value from a successful Commit. If key is an incomplete key, the
// returned pending key will resolve to a unique key generated by the
// datastore.
func (t *Transaction) Put(key *Key, src interface{}) (*PendingKey, error) {
	h, err := t.PutMulti([]*Key{key}, []interface{}{src})
	if err != nil {
		if me, ok := err.(MultiError); ok {
			return nil, me[0]
		}
		return nil, err
	}
	return h[0], nil
}

// PutMulti is a batch version of Put. One PendingKey is returned for each
// element of src in the same order.
func (t *Transaction) PutMulti(keys []*Key, src interface{}) ([]*PendingKey, error) {
	if t.id == nil {
		return nil, errExpiredTransaction
	}
	mutation, err := putMutation(keys, src)
	if err != nil {
		return nil, err
	}
	proto.Merge(t.mutation, mutation)

	// Prepare the returned handles, pre-populating where possible.
	ret := make([]*PendingKey, len(keys))
	for i, key := range keys {
		h := &PendingKey{}
		if key.Incomplete() {
			// This key will be in the final commit result.
			t.pending = append(t.pending, h)
		} else {
			h.key = key
		}

		ret[i] = h
	}

	return ret, nil
}

// Delete is the transaction-specific version of the package function Delete.
// Delete enqueues the deletion of the entity for the given key, to be
// committed atomically upon calling Commit.
func (t *Transaction) Delete(key *Key) error {
	err := t.DeleteMulti([]*Key{key})
	if me, ok := err.(MultiError); ok {
		return me[0]
	}
	return err
}

// DeleteMulti is a batch version of Delete.
func (t *Transaction) DeleteMulti(keys []*Key) error {
	if t.id == nil {
		return errExpiredTransaction
	}
	mutation, err := deleteMutation(keys)
	if err != nil {
		return err
	}
	proto.Merge(t.mutation, mutation)
	return nil
}

// Commit represents the result of a committed transaction.
type Commit struct{}

// Key resolves a pending key handle into a final key.
func (c *Commit) Key(p *PendingKey) *Key {
	if c != p.commit {
		panic("PendingKey was not created by corresponding transaction")
	}
	return p.key
}

// PendingKey represents the key for newly-inserted entity. It can be
// resolved into a Key by calling the Key method of Commit.
type PendingKey struct {
	key    *Key
	commit *Commit
}
