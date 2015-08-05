package tarheader

/*
#define _BSD_SOURCE
#define _DEFAULT_SOURCE
#include <sys/types.h>

unsigned int
my_major(dev_t dev)
{
  return major(dev);
}

unsigned int
my_minor(dev_t dev)
{
  return minor(dev);
}

*/
import "C"
import (
	"archive/tar"
	"os"
	"syscall"
)

func init() {
	populateHeaderStat = append(populateHeaderStat, populateHeaderUnix)
}

func populateHeaderUnix(h *tar.Header, fi os.FileInfo, seen map[uint64]string) {
	st, ok := fi.Sys().(*syscall.Stat_t)
	if !ok {
		return
	}
	h.Uid = int(st.Uid)
	h.Gid = int(st.Gid)
	if st.Mode&syscall.S_IFMT == syscall.S_IFBLK || st.Mode&syscall.S_IFMT == syscall.S_IFCHR {
		h.Devminor = int64(C.my_minor(C.dev_t(st.Rdev)))
		h.Devmajor = int64(C.my_major(C.dev_t(st.Rdev)))
	}
	// If we have already seen this inode, generate a hardlink
	p, ok := seen[uint64(st.Ino)]
	if ok {
		h.Linkname = p
		h.Typeflag = tar.TypeLink
	} else {
		seen[uint64(st.Ino)] = h.Name
	}
}
