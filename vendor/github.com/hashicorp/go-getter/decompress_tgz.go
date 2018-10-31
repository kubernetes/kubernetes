package getter

import (
	"compress/gzip"
	"fmt"
	"os"
	"path/filepath"
)

// TarGzipDecompressor is an implementation of Decompressor that can
// decompress tar.gzip files.
type TarGzipDecompressor struct{}

func (d *TarGzipDecompressor) Decompress(dst, src string, dir bool) error {
	// If we're going into a directory we should make that first
	mkdir := dst
	if !dir {
		mkdir = filepath.Dir(dst)
	}
	if err := os.MkdirAll(mkdir, 0755); err != nil {
		return err
	}

	// File first
	f, err := os.Open(src)
	if err != nil {
		return err
	}
	defer f.Close()

	// Gzip compression is second
	gzipR, err := gzip.NewReader(f)
	if err != nil {
		return fmt.Errorf("Error opening a gzip reader for %s: %s", src, err)
	}
	defer gzipR.Close()

	return untar(gzipR, dst, src, dir)
}
