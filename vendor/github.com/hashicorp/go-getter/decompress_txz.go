package getter

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/ulikunitz/xz"
)

// TarXzDecompressor is an implementation of Decompressor that can
// decompress tar.xz files.
type TarXzDecompressor struct{}

func (d *TarXzDecompressor) Decompress(dst, src string, dir bool) error {
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

	// xz compression is second
	txzR, err := xz.NewReader(f)
	if err != nil {
		return fmt.Errorf("Error opening an xz reader for %s: %s", src, err)
	}

	return untar(txzR, dst, src, dir)
}
