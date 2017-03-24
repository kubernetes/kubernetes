package content

import (
	"context"
	"io"
	"sync"
	"time"

	"github.com/opencontainers/go-digest"
	"github.com/pkg/errors"
)

var (
	// ErrNotFound is returned when an item is not found.
	//
	// Use IsNotFound(err) to detect this condition.
	ErrNotFound = errors.New("content: not found")

	// ErrExists is returned when something exists when it may not be expected.
	//
	// Use IsExists(err) to detect this condition.
	ErrExists = errors.New("content: exists")

	bufPool = sync.Pool{
		New: func() interface{} {
			return make([]byte, 1<<20)
		},
	}
)

type Info struct {
	Digest      digest.Digest
	Size        int64
	CommittedAt time.Time
}

type Provider interface {
	Reader(ctx context.Context, dgst digest.Digest) (io.ReadCloser, error)
}

type Status struct {
	Ref       string
	Offset    int64
	Total     int64
	StartedAt time.Time
	UpdatedAt time.Time
}

type Writer interface {
	io.WriteCloser
	Status() (Status, error)
	Digest() digest.Digest
	Commit(size int64, expected digest.Digest) error
	Truncate(size int64) error
}

type Ingester interface {
	Writer(ctx context.Context, ref string, size int64, expected digest.Digest) (Writer, error)
}

func IsNotFound(err error) bool {
	return errors.Cause(err) == ErrNotFound
}

func IsExists(err error) bool {
	return errors.Cause(err) == ErrExists
}
