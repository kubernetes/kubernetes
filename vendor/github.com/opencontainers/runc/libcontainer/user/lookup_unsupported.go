// +build !darwin,!dragonfly,!freebsd,!linux,!netbsd,!openbsd,!solaris

package user

import (
	"io"
	"syscall"
)

func GetPasswdPath() (string, error) {
	return "", ErrUnsupported
}

func GetPasswd() (io.ReadCloser, error) {
	return nil, ErrUnsupported
}

func GetGroupPath() (string, error) {
	return "", ErrUnsupported
}

func GetGroup() (io.ReadCloser, error) {
	return nil, ErrUnsupported
}

// CurrentUser looks up the current user by their user id in /etc/passwd. If the
// user cannot be found (or there is no /etc/passwd file on the filesystem),
// then CurrentUser returns an error.
func CurrentUser() (User, error) {
	return LookupUid(syscall.Getuid())
}

// CurrentGroup looks up the current user's group by their primary group id's
// entry in /etc/passwd. If the group cannot be found (or there is no
// /etc/group file on the filesystem), then CurrentGroup returns an error.
func CurrentGroup() (Group, error) {
	return LookupGid(syscall.Getgid())
}
