package tarsum

import (
	"io"
)

type writeCloseFlusher interface {
	io.WriteCloser
	Flush() error
}

type nopCloseFlusher struct {
	io.Writer
}

func (n *nopCloseFlusher) Close() error {
	return nil
}

func (n *nopCloseFlusher) Flush() error {
	return nil
}
