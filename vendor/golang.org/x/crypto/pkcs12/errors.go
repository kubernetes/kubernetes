// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkcs12

import "errors"

var (
	// ErrDecryption represents a failure to decrypt the input.
	ErrDecryption = errors.New("pkcs12: decryption error, incorrect padding")

	// ErrIncorrectPassword is returned when an incorrect password is detected.
	// Usually, P12/PFX data is signed to be able to verify the password.
	ErrIncorrectPassword = errors.New("pkcs12: decryption password incorrect")
)

// NotImplementedError indicates that the input is not currently supported.
type NotImplementedError string

func (e NotImplementedError) Error() string {
	return "pkcs12: " + string(e)
}
