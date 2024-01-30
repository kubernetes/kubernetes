/*
Copyright 2023 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package knftables

import (
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"syscall"
)

type nftablesError struct {
	wrapped error
	msg     string
	errno   syscall.Errno
}

// wrapError wraps an error resulting from running nft
func wrapError(err error) error {
	nerr := &nftablesError{wrapped: err, msg: err.Error()}
	if ee, ok := err.(*exec.ExitError); ok {
		if len(ee.Stderr) > 0 {
			nerr.msg = string(ee.Stderr)
			eol := strings.Index(nerr.msg, "\n")
			// The nft binary does not call setlocale() and so will return
			// English error strings regardless of the locale.
			enoent := strings.Index(nerr.msg, "No such file or directory")
			eexist := strings.Index(nerr.msg, "File exists")
			if enoent != -1 && (enoent < eol || eol == -1) {
				nerr.errno = syscall.ENOENT
			} else if eexist != -1 && (eexist < eol || eol == -1) {
				nerr.errno = syscall.EEXIST
			}
		}
	}
	return nerr
}

// notFoundError returns an nftablesError with the given message for which IsNotFound will
// return true.
func notFoundError(format string, args ...interface{}) error {
	return &nftablesError{msg: fmt.Sprintf(format, args...), errno: syscall.ENOENT}
}

// existsError returns an nftablesError with the given message for which IsAlreadyExists
// will return true.
func existsError(format string, args ...interface{}) error {
	return &nftablesError{msg: fmt.Sprintf(format, args...), errno: syscall.EEXIST}
}

func (nerr *nftablesError) Error() string {
	return nerr.msg
}

func (nerr *nftablesError) Unwrap() error {
	return nerr.wrapped
}

// IsNotFound tests if err corresponds to an nftables "not found" error of any sort.
// (e.g., in response to a "delete rule" command, this might indicate that the rule
// doesn't exist, or the chain doesn't exist, or the table doesn't exist.)
func IsNotFound(err error) bool {
	var nerr *nftablesError
	if errors.As(err, &nerr) {
		return nerr.errno == syscall.ENOENT
	}
	return false
}

// IsAlreadyExists tests if err corresponds to an nftables "already exists" error (e.g.
// when doing a "create" rather than an "add").
func IsAlreadyExists(err error) bool {
	var nerr *nftablesError
	if errors.As(err, &nerr) {
		return nerr.errno == syscall.EEXIST
	}
	return false
}
