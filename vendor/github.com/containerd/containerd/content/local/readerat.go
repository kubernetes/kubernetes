package local

import (
	"os"
)

// readerat implements io.ReaderAt in a completely stateless manner by opening
// the referenced file for each call to ReadAt.
type sizeReaderAt struct {
	size int64
	fp   *os.File
}

func (ra sizeReaderAt) ReadAt(p []byte, offset int64) (int, error) {
	return ra.fp.ReadAt(p, offset)
}

func (ra sizeReaderAt) Size() int64 {
	return ra.size
}

func (ra sizeReaderAt) Close() error {
	return ra.fp.Close()
}
