// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package datastore

import (
	"errors"

	"github.com/golang/protobuf/proto"

	"google.golang.org/appengine"
	"google.golang.org/appengine/internal"
	basepb "google.golang.org/appengine/internal/base"
	pb "google.golang.org/appengine/internal/datastore"
)

func init() {
	internal.RegisterTransactionSetter(func(x *pb.Query, t *pb.Transaction) {
		x.Transaction = t
	})
	internal.RegisterTransactionSetter(func(x *pb.GetRequest, t *pb.Transaction) {
		x.Transaction = t
	})
	internal.RegisterTransactionSetter(func(x *pb.PutRequest, t *pb.Transaction) {
		x.Transaction = t
	})
	internal.RegisterTransactionSetter(func(x *pb.DeleteRequest, t *pb.Transaction) {
		x.Transaction = t
	})
}

// ErrConcurrentTransaction is returned when a transaction is rolled back due
// to a conflict with a concurrent transaction.
var ErrConcurrentTransaction = errors.New("datastore: concurrent transaction")

type transaction struct {
	appengine.Context
	transaction pb.Transaction
	finished    bool
}

func (t *transaction) Call(service, method string, in, out proto.Message, opts *internal.CallOptions) error {
	if t.finished {
		return errors.New("datastore: transaction context has expired")
	}
	internal.ApplyTransaction(in, &t.transaction)
	return t.Context.Call(service, method, in, out, opts)
}

func runOnce(c appengine.Context, f func(appengine.Context) error, opts *TransactionOptions) error {
	// Begin the transaction.
	t := &transaction{Context: c}
	req := &pb.BeginTransactionRequest{
		App: proto.String(c.FullyQualifiedAppID()),
	}
	if opts != nil && opts.XG {
		req.AllowMultipleEg = proto.Bool(true)
	}
	if err := t.Context.Call("datastore_v3", "BeginTransaction", req, &t.transaction, nil); err != nil {
		return err
	}

	// Call f, rolling back the transaction if f returns a non-nil error, or panics.
	// The panic is not recovered.
	defer func() {
		if t.finished {
			return
		}
		t.finished = true
		// Ignore the error return value, since we are already returning a non-nil
		// error (or we're panicking).
		c.Call("datastore_v3", "Rollback", &t.transaction, &basepb.VoidProto{}, nil)
	}()
	if err := f(t); err != nil {
		return err
	}
	t.finished = true

	// Commit the transaction.
	res := &pb.CommitResponse{}
	err := c.Call("datastore_v3", "Commit", &t.transaction, res, nil)
	if ae, ok := err.(*internal.APIError); ok {
		if appengine.IsDevAppServer() {
			// The Python Dev AppServer raises an ApplicationError with error code 2 (which is
			// Error.CONCURRENT_TRANSACTION) and message "Concurrency exception.".
			if ae.Code == int32(pb.Error_BAD_REQUEST) && ae.Detail == "ApplicationError: 2 Concurrency exception." {
				return ErrConcurrentTransaction
			}
		}
		if ae.Code == int32(pb.Error_CONCURRENT_TRANSACTION) {
			return ErrConcurrentTransaction
		}
	}
	return err
}

// RunInTransaction runs f in a transaction. It calls f with a transaction
// context tc that f should use for all App Engine operations.
//
// If f returns nil, RunInTransaction attempts to commit the transaction,
// returning nil if it succeeds. If the commit fails due to a conflicting
// transaction, RunInTransaction retries f, each time with a new transaction
// context. It gives up and returns ErrConcurrentTransaction after three
// failed attempts.
//
// If f returns non-nil, then any datastore changes will not be applied and
// RunInTransaction returns that same error. The function f is not retried.
//
// Note that when f returns, the transaction is not yet committed. Calling code
// must be careful not to assume that any of f's changes have been committed
// until RunInTransaction returns nil.
//
// Nested transactions are not supported; c may not be a transaction context.
func RunInTransaction(c appengine.Context, f func(tc appengine.Context) error, opts *TransactionOptions) error {
	if _, ok := c.(*transaction); ok {
		return errors.New("datastore: nested transactions are not supported")
	}
	for i := 0; i < 3; i++ {
		if err := runOnce(c, f, opts); err != ErrConcurrentTransaction {
			return err
		}
	}
	return ErrConcurrentTransaction
}

// TransactionOptions are the options for running a transaction.
type TransactionOptions struct {
	// XG is whether the transaction can cross multiple entity groups. In
	// comparison, a single group transaction is one where all datastore keys
	// used have the same root key. Note that cross group transactions do not
	// have the same behavior as single group transactions. In particular, it
	// is much more likely to see partially applied transactions in different
	// entity groups, in global queries.
	// It is valid to set XG to true even if the transaction is within a
	// single entity group.
	XG bool
}
