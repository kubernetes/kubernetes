package getter

import (
	"compress/bzip2"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// Bzip2Decompressor is an implementation of Decompressor that can
// decompress bz2 files.
type Bzip2Decompressor struct{}

func (d *Bzip2Decompressor) Decompress(dst, src string, dir bool) error {
	// Directory isn't supported at all
	if dir {
		return fmt.Errorf("bzip2-compressed files can only unarchive to a single file")
	}

	// If we're going into a directory we should make that first
	if err := os.MkdirAll(filepath.Dir(dst), 0755); err != nil {
		return err
	}

	// File first
	f, err := os.Open(src)
	if err != nil {
		return err
	}
	defer f.Close()

	// Bzip2 compression is second
	bzipR := bzip2.NewReader(f)

	// Copy it out
	dstF, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer dstF.Close()

	_, err = io.Copy(dstF, bzipR)
	return err
}
