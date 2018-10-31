package getter

import (
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/ulikunitz/xz"
)

// XzDecompressor is an implementation of Decompressor that can
// decompress xz files.
type XzDecompressor struct{}

func (d *XzDecompressor) Decompress(dst, src string, dir bool) error {
	// Directory isn't supported at all
	if dir {
		return fmt.Errorf("xz-compressed files can only unarchive to a single file")
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

	// xz compression is second
	xzR, err := xz.NewReader(f)
	if err != nil {
		return err
	}

	// Copy it out
	dstF, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer dstF.Close()

	_, err = io.Copy(dstF, xzR)
	return err
}
