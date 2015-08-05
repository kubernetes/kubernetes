// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package errors contains common error types for the OpenPGP packages.
package errors

import (
	"strconv"
)

// A StructuralError is returned when OpenPGP data is found to be syntactically
// invalid.
type StructuralError string

func (s StructuralError) Error() string {
	return "openpgp: invalid data: " + string(s)
}

// UnsupportedError indicates that, although the OpenPGP data is valid, it
// makes use of currently unimplemented features.
type UnsupportedError string

func (s UnsupportedError) Error() string {
	return "openpgp: unsupported feature: " + string(s)
}

// InvalidArgumentError indicates that the caller is in error and passed an
// incorrect value.
type InvalidArgumentError string

func (i InvalidArgumentError) Error() string {
	return "openpgp: invalid argument: " + string(i)
}

// SignatureError indicates that a syntactically valid signature failed to
// validate.
type SignatureError string

func (b SignatureError) Error() string {
	return "openpgp: invalid signature: " + string(b)
}

type keyIncorrectError int

func (ki keyIncorrectError) Error() string {
	return "openpgp: incorrect key"
}

var ErrKeyIncorrect error = keyIncorrectError(0)

type unknownIssuerError int

func (unknownIssuerError) Error() string {
	return "openpgp: signature made by unknown entity"
}

var ErrUnknownIssuer error = unknownIssuerError(0)

type keyRevokedError int

func (keyRevokedError) Error() string {
	return "openpgp: signature made by revoked key"
}

var ErrKeyRevoked error = keyRevokedError(0)

type UnknownPacketTypeError uint8

func (upte UnknownPacketTypeError) Error() string {
	return "openpgp: unknown packet type: " + strconv.Itoa(int(upte))
}
