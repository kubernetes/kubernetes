// +build windows

package system

import (
	"time"

	"golang.org/x/sys/windows"
)

//setCTime will set the create time on a file. On Windows, this requires
//calling SetFileTime and explicitly including the create time.
func setCTime(path string, ctime time.Time) error {
	ctimespec := windows.NsecToTimespec(ctime.UnixNano())
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
