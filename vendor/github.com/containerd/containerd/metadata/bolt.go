package metadata

import (
	"context"

	"github.com/boltdb/bolt"
	"github.com/pkg/errors"
)

type transactionKey struct{}

// WithTransactionContext returns a new context holding the provided
// bolt transaction. Functions which require a bolt transaction will
// first check to see if a transaction is already created on the
// context before creating their own.
func WithTransactionContext(ctx context.Context, tx *bolt.Tx) context.Context {
	return context.WithValue(ctx, transactionKey{}, tx)
}

type transactor interface {
	View(fn func(*bolt.Tx) error) error
	Update(fn func(*bolt.Tx) error) error
}

// view gets a bolt db transaction either from the context
// or starts a new one with the provided bolt database.
func view(ctx context.Context, db transactor, fn func(*bolt.Tx) error) error {
	tx, ok := ctx.Value(transactionKey{}).(*bolt.Tx)
	if !ok {
		return db.View(fn)
	}
	return fn(tx)
}

// update gets a writable bolt db transaction either from the context
// or starts a new one with the provided bolt database.
func update(ctx context.Context, db transactor, fn func(*bolt.Tx) error) error {
	tx, ok := ctx.Value(transactionKey{}).(*bolt.Tx)
	if !ok {
		return db.Update(fn)
	} else if !tx.Writable() {
		return errors.Wrap(bolt.ErrTxNotWritable, "unable to use transaction from context")
	}
	return fn(tx)
}
