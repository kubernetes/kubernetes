// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gldriver

import "image"

type bufferImpl struct {
	// buf should always be equal to (i.e. the same ptr, len, cap as) rgba.Pix.
	// It is a separate, redundant field in order to detect modifications to
	// the rgba field that are invalid as per the screen.Buffer documentation.
	buf  []byte
	rgba image.RGBA
	size image.Point
}

func (b *bufferImpl) Release()                {}
func (b *bufferImpl) Size() image.Point       { return b.size }
func (b *bufferImpl) Bounds() image.Rectangle { return image.Rectangle{Max: b.size} }
func (b *bufferImpl) RGBA() *image.RGBA       { return &b.rgba }

func (b *bufferImpl) preUpload() {
	// Check that the program hasn't tried to modify the rgba field via the
	// pointer returned by the bufferImpl.RGBA method. This check doesn't catch
	// 100% of all cases; it simply tries to detect some invalid uses of a
	// screen.Buffer such as:
	//	*buffer.RGBA() = anotherImageRGBA
	if len(b.buf) != 0 && len(b.rgba.Pix) != 0 && &b.buf[0] != &b.rgba.Pix[0] {
		panic("gldriver: invalid Buffer.RGBA modification")
	}
}
