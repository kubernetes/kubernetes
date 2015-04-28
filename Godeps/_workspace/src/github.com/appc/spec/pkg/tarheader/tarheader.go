package tarheader

import (
	"archive/tar"
	"os"
)

var populateHeaderStat []func(h *tar.Header, fi os.FileInfo, seen map[uint64]string)

func Populate(h *tar.Header, fi os.FileInfo, seen map[uint64]string) {
	for _, pop := range populateHeaderStat {
		pop(h, fi, seen)
	}
}
