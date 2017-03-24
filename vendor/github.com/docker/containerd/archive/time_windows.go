package archive

import (
	"syscall"
	"time"
)

// chtimes will set the create time on a file using the given modtime.
// This requires calling SetFileTime and explicitly including the create time.
func chtimes(path string, atime, mtime time.Time) error {
	ctimespec := syscall.NsecToTimespec(mtime.UnixNano())
	pathp, e := syscall.UTF16PtrFromString(path)
	if e != nil {
		return e
	}
	h, e := syscall.CreateFile(pathp,
		syscall.FILE_WRITE_ATTRIBUTES, syscall.FILE_SHARE_WRITE, nil,
		syscall.OPEN_EXISTING, syscall.FILE_FLAG_BACKUP_SEMANTICS, 0)
	if e != nil {
		return e
	}
	defer syscall.Close(h)
	c := syscall.NsecToFiletime(syscall.TimespecToNsec(ctimespec))
	return syscall.SetFileTime(h, &c, nil, nil)
}
