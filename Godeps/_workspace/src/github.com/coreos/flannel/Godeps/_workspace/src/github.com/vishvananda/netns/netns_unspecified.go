// +build !linux

package netns

import (
	"errors"
)

var (
	ErrNotImplemented = errors.New("not implemented")
)

func Set(ns Namespace) (err error) {
	return ErrNotImplemented
}

func New() (ns Namespace, err error) {
	return -1, ErrNotImplemented
}

func Get() (Namespace, error) {
	return -1, ErrNotImplemented
}

func GetFromName(name string) (Namespace, error) {
	return -1, ErrNotImplemented
}

func GetFromPid(pid int) (Namespace, error) {
	return -1, ErrNotImplemented
}

func GetFromDocker(id string) (Namespace, error) {
	return -1, ErrNotImplemented
}
