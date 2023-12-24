// Copyright 2018 Thales e-Security, Inc
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

package crypto11

import (
	"fmt"

	"github.com/miekg/pkcs11"
)

// cipher.Block ---------------------------------------------------------

// BlockSize returns the cipher's block size in bytes.
func (key *SecretKey) BlockSize() int {
	return key.Cipher.BlockSize
}

// Decrypt decrypts the first block in src into dst.
// Dst and src must overlap entirely or not at all.
//
// Using this method for bulk operation is very inefficient, as it makes a round trip to the HSM
// (which may be network-connected) for each block.
// For more efficient operation, see NewCBCDecrypterCloser, NewCBCDecrypter or NewCBC.
func (key *SecretKey) Decrypt(dst, src []byte) {
	var result []byte
	if err := key.context.withSession(func(session *pkcs11Session) (err error) {
		mech := []*pkcs11.Mechanism{pkcs11.NewMechanism(key.Cipher.ECBMech, nil)}
		if err = session.ctx.DecryptInit(session.handle, mech, key.handle); err != nil {
			return
		}
		if result, err = session.ctx.Decrypt(session.handle, src[:key.Cipher.BlockSize]); err != nil {
			return
		}
		if len(result) != key.Cipher.BlockSize {
			err = fmt.Errorf("C_Decrypt: returned %v bytes, wanted %v", len(result), key.Cipher.BlockSize)
			return
		}
		return
	}); err != nil {
		panic(err)
	} else {
		copy(dst[:key.Cipher.BlockSize], result)
	}
}

// Encrypt encrypts the first block in src into dst.
// Dst and src must overlap entirely or not at all.
//
// Using this method for bulk operation is very inefficient, as it makes a round trip to the HSM
// (which may be network-connected) for each block.
// For more efficient operation, see NewCBCEncrypterCloser, NewCBCEncrypter or NewCBC.
func (key *SecretKey) Encrypt(dst, src []byte) {
	var result []byte
	if err := key.context.withSession(func(session *pkcs11Session) (err error) {
		mech := []*pkcs11.Mechanism{pkcs11.NewMechanism(key.Cipher.ECBMech, nil)}
		if err = session.ctx.EncryptInit(session.handle, mech, key.handle); err != nil {
			return
		}
		if result, err = session.ctx.Encrypt(session.handle, src[:key.Cipher.BlockSize]); err != nil {
			return
		}
		if len(result) != key.Cipher.BlockSize {
			err = fmt.Errorf("C_Encrypt: unexpectedly returned %v bytes, wanted %v", len(result), key.Cipher.BlockSize)
			return
		}
		return
	}); err != nil {
		panic(err)
	} else {
		copy(dst[:key.Cipher.BlockSize], result)
	}
}
