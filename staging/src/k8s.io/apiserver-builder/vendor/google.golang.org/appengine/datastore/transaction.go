// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package datastore

import (
	"errors"

	"golang.org/x/net/context"

	"google.golang.org/appengine/internal"
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

// RunInTransaction runs f in a transaction. It calls f with a transaction
// context tc that f should use for all App Engine operations.
//
// If f returns nil, RunInTransaction attempts to commit the transaction,
// returning nil if it succeeds. If the commit fails due to a conflicting
// transaction, RunInTransaction retries f, each time with a new transaction
// context. It gives up and returns ErrConcurrentTransaction after three
// failed attempts. The number of attempts can be configured by specifying
// TransactionOptions.Attempts.
//
// If f returns non-nil, then any datastore changes will not be applied and
// RunInTransaction returns that same error. The function f is not retried.
//
// Note that when f returns, the transaction is not yet committed. Calling code
// must be careful not to assume that any of f's changes have been committed
// until RunInTransaction returns nil.
//
// Since f may be called multiple times, f should usually be idempotent.
// datastore.Get is not idempotent when unmarshaling slice fields.
//
// Nested transactions are not supported; c may not be a transaction context.
func RunInTransaction(c context.Context, f func(tc context.Context) error, opts *TransactionOptions) error {
	xg := false
	if opts != nil {
		xg = opts.XG
	}
	attempts := 3
	if opts != nil && opts.Attempts > 0 {
		attempts = opts.Attempts
	}
	for i := 0; i < attempts; i++ {
		if err := internal.RunTransactionOnce(c, f, xg); err != internal.ErrConcurrentTransaction {
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
	// Attempts controls the number of retries to perform when commits fail
	// due to a conflicting transaction. If omitted, it defaults to 3.
	Attempts int
}
