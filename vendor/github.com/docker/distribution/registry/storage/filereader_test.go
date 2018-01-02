package storage

import (
	"bytes"
	"io"
	mrand "math/rand"
	"os"
	"testing"

	"github.com/docker/distribution/context"
	"github.com/docker/distribution/registry/storage/driver/inmemory"
	"github.com/opencontainers/go-digest"
)

func TestSimpleRead(t *testing.T) {
	ctx := context.Background()
	content := make([]byte, 1<<20)
	n, err := mrand.Read(content)
	if err != nil {
		t.Fatalf("unexpected error building random data: %v", err)
	}

	if n != len(content) {
		t.Fatalf("random read didn't fill buffer")
	}

	dgst, err := digest.FromReader(bytes.NewReader(content))
	if err != nil {
		t.Fatalf("unexpected error digesting random content: %v", err)
	}

	driver := inmemory.New()
	path := "/random"

	if err := driver.PutContent(ctx, path, content); err != nil {
		t.Fatalf("error putting patterned content: %v", err)
	}

	fr, err := newFileReader(ctx, driver, path, int64(len(content)))
	if err != nil {
		t.Fatalf("error allocating file reader: %v", err)
	}

	verifier := dgst.Verifier()
	io.Copy(verifier, fr)

	if !verifier.Verified() {
		t.Fatalf("unable to verify read data")
	}
}

func TestFileReaderSeek(t *testing.T) {
	driver := inmemory.New()
	pattern := "01234567890ab" // prime length block
	repititions := 1024
	path := "/patterned"
	content := bytes.Repeat([]byte(pattern), repititions)
	ctx := context.Background()

	if err := driver.PutContent(ctx, path, content); err != nil {
		t.Fatalf("error putting patterned content: %v", err)
	}

	fr, err := newFileReader(ctx, driver, path, int64(len(content)))

	if err != nil {
		t.Fatalf("unexpected error creating file reader: %v", err)
	}

	// Seek all over the place, in blocks of pattern size and make sure we get
	// the right data.
	for _, repitition := range mrand.Perm(repititions - 1) {
		targetOffset := int64(len(pattern) * repitition)
		// Seek to a multiple of pattern size and read pattern size bytes
		offset, err := fr.Seek(targetOffset, os.SEEK_SET)
		if err != nil {
			t.Fatalf("unexpected error seeking: %v", err)
		}

		if offset != targetOffset {
			t.Fatalf("did not seek to correct offset: %d != %d", offset, targetOffset)
		}

		p := make([]byte, len(pattern))

		n, err := fr.Read(p)
		if err != nil {
			t.Fatalf("error reading pattern: %v", err)
		}

		if n != len(pattern) {
			t.Fatalf("incorrect read length: %d != %d", n, len(pattern))
		}

		if string(p) != pattern {
			t.Fatalf("incorrect read content: %q != %q", p, pattern)
		}

		// Check offset
		current, err := fr.Seek(0, os.SEEK_CUR)
		if err != nil {
			t.Fatalf("error checking current offset: %v", err)
		}

		if current != targetOffset+int64(len(pattern)) {
			t.Fatalf("unexpected offset after read: %v", err)
		}
	}

	start, err := fr.Seek(0, os.SEEK_SET)
	if err != nil {
		t.Fatalf("error seeking to start: %v", err)
	}

	if start != 0 {
		t.Fatalf("expected to seek to start: %v != 0", start)
	}

	end, err := fr.Seek(0, os.SEEK_END)
	if err != nil {
		t.Fatalf("error checking current offset: %v", err)
	}

	if end != int64(len(content)) {
		t.Fatalf("expected to seek to end: %v != %v", end, len(content))
	}

	// 4. Seek before start, ensure error.

	// seek before start
	before, err := fr.Seek(-1, os.SEEK_SET)
	if err == nil {
		t.Fatalf("error expected, returned offset=%v", before)
	}

	// 5. Seek after end,
	after, err := fr.Seek(1, os.SEEK_END)
	if err != nil {
		t.Fatalf("unexpected error expected, returned offset=%v", after)
	}

	p := make([]byte, 16)
	n, err := fr.Read(p)

	if n != 0 {
		t.Fatalf("bytes reads %d != %d", n, 0)
	}

	if err != io.EOF {
		t.Fatalf("expected io.EOF, got %v", err)
	}
}

// TestFileReaderNonExistentFile ensures the reader behaves as expected with a
// missing or zero-length remote file. While the file may not exist, the
// reader should not error out on creation and should return 0-bytes from the
// read method, with an io.EOF error.
func TestFileReaderNonExistentFile(t *testing.T) {
	driver := inmemory.New()
	fr, err := newFileReader(context.Background(), driver, "/doesnotexist", 10)
	if err != nil {
		t.Fatalf("unexpected error initializing reader: %v", err)
	}

	var buf [1024]byte

	n, err := fr.Read(buf[:])
	if n != 0 {
		t.Fatalf("non-zero byte read reported: %d != 0", n)
	}

	if err != io.EOF {
		t.Fatalf("read on missing file should return io.EOF, got %v", err)
	}
}

// TestLayerReadErrors covers the various error return type for different
// conditions that can arise when reading a layer.
func TestFileReaderErrors(t *testing.T) {
	// TODO(stevvooe): We need to cover error return types, driven by the
	// errors returned via the HTTP API. For now, here is an incomplete list:
	//
	// 	1. Layer Not Found: returned when layer is not found or access is
	//        denied.
	//	2. Layer Unavailable: returned when link references are unresolved,
	//     but layer is known to the registry.
	//  3. Layer Invalid: This may more split into more errors, but should be
	//     returned when name or tarsum does not reference a valid error. We
	//     may also need something to communication layer verification errors
	//     for the inline tarsum check.
	//	4. Timeout: timeouts to backend. Need to better understand these
	//     failure cases and how the storage driver propagates these errors
	//     up the stack.
}
