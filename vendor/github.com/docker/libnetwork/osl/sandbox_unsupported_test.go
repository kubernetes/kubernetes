// +build !linux,!solaris

package osl

import (
	"errors"
	"testing"
)

var ErrNotImplemented = errors.New("not implemented")

func newKey(t *testing.T) (string, error) {
	return nil, ErrNotImplemented
}

func verifySandbox(t *testing.T, s Sandbox) {
	return
}
