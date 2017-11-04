// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package user

import (
	"io"
	"os"

	"golang.org/x/sys/unix"
)

// Unix-specific path to the passwd and group formatted files.
const (
	unixPasswdPath = "/etc/passwd"
	unixGroupPath  = "/etc/group"
)

func GetPasswdPath() (string, error) {
	return unixPasswdPath, nil
}

func GetPasswd() (io.ReadCloser, error) {
	return os.Open(unixPasswdPath)
}

func GetGroupPath() (string, error) {
	return unixGroupPath, nil
}

func GetGroup() (io.ReadCloser, error) {
	return os.Open(unixGroupPath)
}

// CurrentUser looks up the current user by their user id in /etc/passwd. If the
// user cannot be found (or there is no /etc/passwd file on the filesystem),
// then CurrentUser returns an error.
func CurrentUser() (User, error) {
	return LookupUid(unix.Getuid())
}

// CurrentGroup looks up the current user's group by their primary group id's
// entry in /etc/passwd. If the group cannot be found (or there is no
// /etc/group file on the filesystem), then CurrentGroup returns an error.
func CurrentGroup() (Group, error) {
	return LookupGid(unix.Getgid())
}
