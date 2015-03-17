// +build windows

package archive

import (
	"github.com/docker/docker/vendor/src/code.google.com/p/go/src/pkg/archive/tar"
)

func setHeaderForSpecialDevice(hdr *tar.Header, ta *tarAppender, name string, stat interface{}) (nlink uint32, inode uint64, err error) {
	// do nothing. no notion of Rdev, Inode, Nlink in stat on Windows
	return
}
