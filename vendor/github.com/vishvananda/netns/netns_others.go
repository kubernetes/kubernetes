//go:build !linux
// +build !linux

package netns

import "errors"

var ErrNotImplemented = errors.New("not implemented")

// Setns sets namespace using golang.org/x/sys/unix.Setns on Linux. It
// is not implemented on other platforms.
//
// Deprecated: Use golang.org/x/sys/unix.Setns instead.
func Setns(ns NsHandle, nstype int) error {
	return ErrNotImplemented
}

func Set(ns NsHandle) error {
	return ErrNotImplemented
}

func New() (NsHandle, error) {
	return -1, ErrNotImplemented
}

func NewNamed(name string) (NsHandle, error) {
	return -1, ErrNotImplemented
}

func DeleteNamed(name string) error {
	return ErrNotImplemented
}

func Get() (NsHandle, error) {
	return -1, ErrNotImplemented
}

func GetFromPath(path string) (NsHandle, error) {
	return -1, ErrNotImplemented
}

func GetFromName(name string) (NsHandle, error) {
	return -1, ErrNotImplemented
}

func GetFromPid(pid int) (NsHandle, error) {
	return -1, ErrNotImplemented
}

func GetFromThread(pid int, tid int) (NsHandle, error) {
	return -1, ErrNotImplemented
}

func GetFromDocker(id string) (NsHandle, error) {
	return -1, ErrNotImplemented
}
