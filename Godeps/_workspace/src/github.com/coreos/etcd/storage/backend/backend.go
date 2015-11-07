package backend

import (
	"io"
	"log"
	"time"

	"github.com/boltdb/bolt"
)

type Backend interface {
	BatchTx() BatchTx
	Snapshot(w io.Writer) (n int64, err error)
	ForceCommit()
	Close() error
}

type backend struct {
	db *bolt.DB

	batchInterval time.Duration
	batchLimit    int
	batchTx       *batchTx

	stopc chan struct{}
	donec chan struct{}
}

func New(path string, d time.Duration, limit int) Backend {
	return newBackend(path, d, limit)
}

func newBackend(path string, d time.Duration, limit int) *backend {
	db, err := bolt.Open(path, 0600, nil)
	if err != nil {
		log.Panicf("backend: cannot open database at %s (%v)", path, err)
	}

	b := &backend{
		db: db,

		batchInterval: d,
		batchLimit:    limit,

		stopc: make(chan struct{}),
		donec: make(chan struct{}),
	}
	b.batchTx = newBatchTx(b)
	go b.run()
	return b
}

// BatchTx returns the current batch tx in coalescer. The tx can be used for read and
// write operations. The write result can be retrieved within the same tx immediately.
// The write result is isolated with other txs until the current one get committed.
func (b *backend) BatchTx() BatchTx {
	return b.batchTx
}

// force commit the current batching tx.
func (b *backend) ForceCommit() {
	b.batchTx.Commit()
}

func (b *backend) Snapshot(w io.Writer) (n int64, err error) {
	b.db.View(func(tx *bolt.Tx) error {
		n, err = tx.WriteTo(w)
		return nil
	})
	return n, err
}

func (b *backend) run() {
	defer close(b.donec)

	for {
		select {
		case <-time.After(b.batchInterval):
		case <-b.stopc:
			b.batchTx.CommitAndStop()
			return
		}
		b.batchTx.Commit()
	}
}

func (b *backend) Close() error {
	close(b.stopc)
	<-b.donec
	return b.db.Close()
}
