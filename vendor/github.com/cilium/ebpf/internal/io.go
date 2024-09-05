package internal

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
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

// ReadUint64FromFile reads a uint64 from a file.
//
// format specifies the contents of the file in fmt.Scanf syntax.
func ReadUint64FromFile(format string, path ...string) (uint64, error) {
	filename := filepath.Join(path...)
	data, err := os.ReadFile(filename)
	if err != nil {
		return 0, fmt.Errorf("reading file %q: %w", filename, err)
	}

	var value uint64
	n, err := fmt.Fscanf(bytes.NewReader(data), format, &value)
	if err != nil {
		return 0, fmt.Errorf("parsing file %q: %w", filename, err)
	}
	if n != 1 {
		return 0, fmt.Errorf("parsing file %q: expected 1 item, got %d", filename, n)
	}

	return value, nil
}

type uint64FromFileKey struct {
	format, path string
}

var uint64FromFileCache = struct {
	sync.RWMutex
	values map[uint64FromFileKey]uint64
}{
	values: map[uint64FromFileKey]uint64{},
}

// ReadUint64FromFileOnce is like readUint64FromFile but memoizes the result.
func ReadUint64FromFileOnce(format string, path ...string) (uint64, error) {
	filename := filepath.Join(path...)
	key := uint64FromFileKey{format, filename}

	uint64FromFileCache.RLock()
	if value, ok := uint64FromFileCache.values[key]; ok {
		uint64FromFileCache.RUnlock()
		return value, nil
	}
	uint64FromFileCache.RUnlock()

	value, err := ReadUint64FromFile(format, filename)
	if err != nil {
		return 0, err
	}

	uint64FromFileCache.Lock()
	defer uint64FromFileCache.Unlock()

	if value, ok := uint64FromFileCache.values[key]; ok {
		// Someone else got here before us, use what is cached.
		return value, nil
	}

	uint64FromFileCache.values[key] = value
	return value, nil
}
