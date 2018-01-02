// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package windriver

import (
	"image"
	"image/draw"
	"sync"
	"syscall"

	"golang.org/x/exp/shiny/driver/internal/swizzle"
)

type bufferImpl struct {
	hbitmap syscall.Handle
	buf     []byte
	rgba    image.RGBA
	size    image.Point

	mu        sync.Mutex
	nUpload   uint32
	released  bool
	cleanedUp bool
}

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
		panic("windriver: invalid Buffer.RGBA modification")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if b.released {
		panic("windriver: Buffer.Upload called after Buffer.Release")
	}
	if b.nUpload == 0 {
		swizzle.BGRA(b.buf)
	}
	b.nUpload++
}

func (b *bufferImpl) postUpload() {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.nUpload--
	if b.nUpload != 0 {
		return
	}

	if b.released {
		go b.cleanUp()
	} else {
		swizzle.BGRA(b.buf)
	}
}

func (b *bufferImpl) Release() {
	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.released && b.nUpload == 0 {
		go b.cleanUp()
	}
	b.released = true
}

func (b *bufferImpl) cleanUp() {
	b.mu.Lock()
	if b.cleanedUp {
		b.mu.Unlock()
		panic("windriver: Buffer clean-up occurred twice")
	}
	b.cleanedUp = true
	b.mu.Unlock()

	b.rgba.Pix = nil
	_DeleteObject(b.hbitmap)
}

func (b *bufferImpl) blitToDC(dc syscall.Handle, dp image.Point, sr image.Rectangle) error {
	b.preUpload()
	defer b.postUpload()

	dr := sr.Add(dp.Sub(sr.Min))
	return copyBitmapToDC(dc, dr, b.hbitmap, sr, draw.Src)
}
