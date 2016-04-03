// +build windows

package archive

import (
	"archive/tar"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/fsouza/go-dockerclient/external/github.com/docker/docker/pkg/longpath"
)

// fixVolumePathPrefix does platform specific processing to ensure that if
// the path being passed in is not in a volume path format, convert it to one.
func fixVolumePathPrefix(srcPath string) string {
	return longpath.AddPrefix(srcPath)
}

// getWalkRoot calculates the root path when performing a TarWithOptions.
// We use a separate function as this is platform specific.
func getWalkRoot(srcPath string, include string) string {
	return filepath.Join(srcPath, include)
}

// CanonicalTarNameForPath returns platform-specific filepath
// to canonical posix-style path for tar archival. p is relative
// path.
func CanonicalTarNameForPath(p string) (string, error) {
	// windows: convert windows style relative path with backslashes
	// into forward slashes. Since windows does not allow '/' or '\'
	// in file names, it is mostly safe to replace however we must
	// check just in case
	if strings.Contains(p, "/") {
		return "", fmt.Errorf("Windows path contains forward slash: %s", p)
	}
	return strings.Replace(p, string(os.PathSeparator), "/", -1), nil

}

// chmodTarEntry is used to adjust the file permissions used in tar header based
// on the platform the archival is done.
func chmodTarEntry(perm os.FileMode) os.FileMode {
	perm &= 0755
	// Add the x bit: make everything +x from windows
	perm |= 0111

	return perm
}

func setHeaderForSpecialDevice(hdr *tar.Header, ta *tarAppender, name string, stat interface{}) (inode uint64, err error) {
	// do nothing. no notion of Rdev, Inode, Nlink in stat on Windows
	return
}

// handleTarTypeBlockCharFifo is an OS-specific helper function used by
// createTarFile to handle the following types of header: Block; Char; Fifo
func handleTarTypeBlockCharFifo(hdr *tar.Header, path string) error {
	return nil
}

func handleLChmod(hdr *tar.Header, path string, hdrInfo os.FileInfo) error {
	return nil
}

func getFileUIDGID(stat interface{}) (int, int, error) {
	// no notion of file ownership mapping yet on Windows
	return 0, 0, nil
}
