package getter

import (
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// GzipDecompressor is an implementation of Decompressor that can
// decompress gzip files.
type GzipDecompressor struct{}

func (d *GzipDecompressor) Decompress(dst, src string, dir bool) error {
	// Directory isn't supported at all
	if dir {
		return fmt.Errorf("gzip-compressed files can only unarchive to a single file")
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

	// gzip compression is second
	gzipR, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gzipR.Close()

	// Copy it out
	dstF, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer dstF.Close()

	_, err = io.Copy(dstF, gzipR)
	return err
}
