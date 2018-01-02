// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package windriver

import (
	"fmt"
	"image"
	"sync"
	"syscall"
	"unsafe"

	"golang.org/x/exp/shiny/driver/internal/win32"
	"golang.org/x/exp/shiny/screen"
)

var theScreen = &screenImpl{
	windows: make(map[syscall.Handle]*windowImpl),
}

type screenImpl struct {
	mu      sync.Mutex
	windows map[syscall.Handle]*windowImpl
}

func (*screenImpl) NewBuffer(size image.Point) (screen.Buffer, error) {
	// Buffer length must fit in BITMAPINFO.Header.SizeImage (uint32), as
	// well as in Go slice length (int). It's easiest to be consistent
	// between 32-bit and 64-bit, so we just use int32.
	const (
		maxInt32  = 0x7fffffff
		maxBufLen = maxInt32
	)
	if size.X < 0 || size.X > maxInt32 || size.Y < 0 || size.Y > maxInt32 || int64(size.X)*int64(size.Y)*4 > maxBufLen {
		return nil, fmt.Errorf("windriver: invalid buffer size %v", size)
	}

	hbitmap, bitvalues, err := mkbitmap(size)
	if err != nil {
		return nil, err
	}
	bufLen := 4 * size.X * size.Y
	array := (*[maxBufLen]byte)(unsafe.Pointer(bitvalues))
	buf := (*array)[:bufLen:bufLen]
	return &bufferImpl{
		hbitmap: hbitmap,
		buf:     buf,
		rgba: image.RGBA{
			Pix:    buf,
			Stride: 4 * size.X,
			Rect:   image.Rectangle{Max: size},
		},
		size: size,
	}, nil
}

func (*screenImpl) NewTexture(size image.Point) (screen.Texture, error) {
	return newTexture(size)
}

func (s *screenImpl) NewWindow(opts *screen.NewWindowOptions) (screen.Window, error) {
	w := &windowImpl{}

	var err error
	w.hwnd, err = win32.NewWindow(opts)
	if err != nil {
		return nil, err
	}

	s.mu.Lock()
	s.windows[w.hwnd] = w
	s.mu.Unlock()

	win32.Show(w.hwnd)
	return w, nil
}
