// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Package errors provides libraries for working with the go-errors/errors library.
package errors

import (
	"fmt"

	goerrors "github.com/go-errors/errors"
)

// Wrap returns err wrapped in a go-error.  If err is nil, returns nil.
func Wrap(err interface{}) error {
	if err == nil {
		return nil
	}
	return goerrors.Wrap(err, 1)
}

// WrapPrefixf returns err wrapped in a go-error with a message prefix.  If err is nil, returns nil.
func WrapPrefixf(err interface{}, msg string, args ...interface{}) error {
	if err == nil {
		return nil
	}
	return goerrors.WrapPrefix(err, fmt.Sprintf(msg, args...), 1)
}

// Errorf returns a new go-error.
func Errorf(msg string, args ...interface{}) error {
	return goerrors.Wrap(fmt.Errorf(msg, args...), 1)
}

// GetStack returns a stack trace for the error if it has one
func GetStack(err error) string {
	if e, ok := err.(*goerrors.Error); ok {
		return string(e.Stack())
	}
	return ""
}
