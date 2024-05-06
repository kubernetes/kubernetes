package internal

import (
	"bufio"
	"compress/gzip"
	"errors"
	"io"
	"os"
)

// NewBufferedSectionReader wraps an io.ReaderAt in an appropriately-sized
// buffered reader. It is a convenience function for reading subsections of
// ELF sections while minimizing the amount of read() syscalls made.
//
// Syscall overhead is non-negligible in continuous integration context
// where ELFs might be accessed over virtual filesystems with poor random
// access performance. Buffering reads makes sense because (sub)sections
// end up being read completely anyway.
//
// Use instead of the r.Seek() + io.LimitReader() pattern.
func NewBufferedSectionReader(ra io.ReaderAt, off, n int64) *bufio.Reader {
	// Clamp the size of the buffer to one page to avoid slurping large parts
	// of a file into memory. bufio.NewReader uses a hardcoded default buffer
	// of 4096. Allow arches with larger pages to allocate more, but don't
	// allocate a fixed 4k buffer if we only need to read a small segment.
	buf := n
	if ps := int64(os.Getpagesize()); n > ps {
		buf = ps
	}

	return bufio.NewReaderSize(io.NewSectionReader(ra, off, n), int(buf))
}

// DiscardZeroes makes sure that all written bytes are zero
// before discarding them.
type DiscardZeroes struct{}

func (DiscardZeroes) Write(p []byte) (int, error) {
	for _, b := range p {
		if b != 0 {
			return 0, errors.New("encountered non-zero byte")
		}
	}
	return len(p), nil
}

// ReadAllCompressed decompresses a gzipped file into memory.
func ReadAllCompressed(file string) ([]byte, error) {
	fh, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer fh.Close()

	gz, err := gzip.NewReader(fh)
	if err != nil {
		return nil, err
	}
	defer gz.Close()

	return io.ReadAll(gz)
}
