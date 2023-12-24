// Copyright 2016, 2017 Thales e-Security, Inc
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
	"io"
)

// NewRandomReader returns a reader for the random number generator on the token.
func (c *Context) NewRandomReader() (io.Reader, error) {
	if c.closed.Get() {
		return nil, errClosed
	}

	return pkcs11RandReader{c}, nil
}

// pkcs11RandReader is a random number reader that uses PKCS#11.
type pkcs11RandReader struct {
	context *Context
}

// This implements the Reader interface for pkcs11RandReader.
func (r pkcs11RandReader) Read(data []byte) (n int, err error) {
	var result []byte

	if err = r.context.withSession(func(session *pkcs11Session) error {
		result, err = r.context.ctx.GenerateRandom(session.handle, len(data))
		return err
	}); err != nil {
		return 0, err
	}
	copy(data, result)
	return len(result), err
}
