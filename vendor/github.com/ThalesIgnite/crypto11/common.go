// Copyright 2017 Thales e-Security, Inc
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
	"C"
	"encoding/asn1"
	"math/big"
	"unsafe"

	"github.com/miekg/pkcs11"
	"github.com/pkg/errors"
)

func ulongToBytes(n uint) []byte {
	return C.GoBytes(unsafe.Pointer(&n), C.sizeof_ulong) // ugh!
}

func bytesToUlong(bs []byte) (n uint) {
	sliceSize := len(bs)
	if sliceSize == 0 {
		return 0
	}

	value := *(*uint)(unsafe.Pointer(&bs[0]))
	if sliceSize > C.sizeof_ulong {
		return value
	}

	// truncate the value to the # of bits present in the byte slice since
	// the unsafe pointer will always grab/convert ULONG # of bytes
	var mask uint
	for i := 0; i < sliceSize; i++ {
		mask |= 0xff << uint(i * 8)
	}
	return value & mask
}

func concat(slices ...[]byte) []byte {
	n := 0
	for _, slice := range slices {
		n += len(slice)
	}
	r := make([]byte, n)
	n = 0
	for _, slice := range slices {
		n += copy(r[n:], slice)
	}
	return r
}

// Representation of a *DSA signature
type dsaSignature struct {
	R, S *big.Int
}

// Populate a dsaSignature from a raw byte sequence
func (sig *dsaSignature) unmarshalBytes(sigBytes []byte) error {
	if len(sigBytes) == 0 || len(sigBytes)%2 != 0 {
		return errors.New("DSA signature length is invalid from token")
	}
	n := len(sigBytes) / 2
	sig.R, sig.S = new(big.Int), new(big.Int)
	sig.R.SetBytes(sigBytes[:n])
	sig.S.SetBytes(sigBytes[n:])
	return nil
}

// Populate a dsaSignature from DER encoding
func (sig *dsaSignature) unmarshalDER(sigDER []byte) error {
	if rest, err := asn1.Unmarshal(sigDER, sig); err != nil {
		return errors.WithMessage(err, "DSA signature contains invalid ASN.1 data")
	} else if len(rest) > 0 {
		return errors.New("unexpected data found after DSA signature")
	}
	return nil
}

// Return the DER encoding of a dsaSignature
func (sig *dsaSignature) marshalDER() ([]byte, error) {
	return asn1.Marshal(*sig)
}

// Compute *DSA signature and marshal the result in DER form
func (c *Context) dsaGeneric(key pkcs11.ObjectHandle, mechanism uint, digest []byte) ([]byte, error) {
	var err error
	var sigBytes []byte
	var sig dsaSignature
	mech := []*pkcs11.Mechanism{pkcs11.NewMechanism(mechanism, nil)}
	err = c.withSession(func(session *pkcs11Session) error {
		if err = c.ctx.SignInit(session.handle, mech, key); err != nil {
			return err
		}
		sigBytes, err = c.ctx.Sign(session.handle, digest)
		return err
	})
	if err != nil {
		return nil, err
	}
	err = sig.unmarshalBytes(sigBytes)
	if err != nil {
		return nil, err
	}

	return sig.marshalDER()
}
