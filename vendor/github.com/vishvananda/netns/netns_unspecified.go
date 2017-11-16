// +build !linux

package netns

import (
	"errors"
)

var (
	ErrNotImplemented = errors.New("not implemented")
)

func Set(ns NsHandle) (err error) {
	return ErrNotImplemented
}

func New() (ns NsHandle, err error) {
	return -1, ErrNotImplemented
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

func GetFromThread(pid, tid int) (NsHandle, error) {
	return -1, ErrNotImplemented
}

func GetFromDocker(id string) (NsHandle, error) {
	return -1, ErrNotImplemented
}
