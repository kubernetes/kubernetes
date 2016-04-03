package chrootarchive

import (
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/system"
)

var chrootArchiver = &archive.Archiver{Untar: Untar}

// Untar reads a stream of bytes from `archive`, parses it as a tar archive,
// and unpacks it into the directory at `dest`.
// The archive may be compressed with one of the following algorithms:
//  identity (uncompressed), gzip, bzip2, xz.
func Untar(tarArchive io.Reader, dest string, options *archive.TarOptions) error {

	if tarArchive == nil {
		return fmt.Errorf("Empty archive")
	}
	if options == nil {
		options = &archive.TarOptions{}
	}
	if options.ExcludePatterns == nil {
		options.ExcludePatterns = []string{}
	}

	dest = filepath.Clean(dest)
	if _, err := os.Stat(dest); os.IsNotExist(err) {
		if err := system.MkdirAll(dest, 0777); err != nil {
			return err
		}
	}

	decompressedArchive, err := archive.DecompressStream(tarArchive)
	if err != nil {
		return err
	}
	defer decompressedArchive.Close()

	return invokeUnpack(decompressedArchive, dest, options)
}

// TarUntar is a convenience function which calls Tar and Untar, with the output of one piped into the other.
// If either Tar or Untar fails, TarUntar aborts and returns the error.
func TarUntar(src, dst string) error {
	return chrootArchiver.TarUntar(src, dst)
}

// CopyWithTar creates a tar archive of filesystem path `src`, and
// unpacks it at filesystem path `dst`.
// The archive is streamed directly with fixed buffering and no
// intermediary disk IO.
func CopyWithTar(src, dst string) error {
	return chrootArchiver.CopyWithTar(src, dst)
}

// CopyFileWithTar emulates the behavior of the 'cp' command-line
// for a single file. It copies a regular file from path `src` to
// path `dst`, and preserves all its metadata.
//
// If `dst` ends with a trailing slash '/' ('\' on Windows), the final
// destination path will be `dst/base(src)` or `dst\base(src)`
func CopyFileWithTar(src, dst string) (err error) {
	return chrootArchiver.CopyFileWithTar(src, dst)
}

// UntarPath is a convenience function which looks for an archive
// at filesystem path `src`, and unpacks it at `dst`.
func UntarPath(src, dst string) error {
	return chrootArchiver.UntarPath(src, dst)
}
