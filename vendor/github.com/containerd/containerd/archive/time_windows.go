package archive

import (
	"time"

	"golang.org/x/sys/windows"
)

// chtimes will set the create time on a file using the given modtime.
// This requires calling SetFileTime and explicitly including the create time.
func chtimes(path string, atime, mtime time.Time) error {
	ctimespec := windows.NsecToTimespec(mtime.UnixNano())
	pathp, e := windows.UTF16PtrFromString(path)
	if e != nil {
		return e
	}
	h, e := windows.CreateFile(pathp,
		windows.FILE_WRITE_ATTRIBUTES, windows.FILE_SHARE_WRITE, nil,
		windows.OPEN_EXISTING, windows.FILE_FLAG_BACKUP_SEMANTICS, 0)
	if e != nil {
		return e
	}
	defer windows.Close(h)
	c := windows.NsecToFiletime(windows.TimespecToNsec(ctimespec))
	return windows.SetFileTime(h, &c, nil, nil)
}
