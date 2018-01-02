// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package windriver

import (
	"errors"
	"image"
	"image/color"
	"image/draw"
	"sync"
	"syscall"
	"unsafe"

	"golang.org/x/exp/shiny/driver/internal/win32"
	"golang.org/x/exp/shiny/screen"
)

type textureImpl struct {
	size   image.Point
	dc     syscall.Handle
	bitmap syscall.Handle

	mu       sync.Mutex
	released bool
}

type handleCreateTextureParams struct {
	size   image.Point
	dc     syscall.Handle
	bitmap syscall.Handle
	err    error
}

var msgCreateTexture = win32.AddScreenMsg(handleCreateTexture)

func newTexture(size image.Point) (screen.Texture, error) {
	p := handleCreateTextureParams{size: size}
	win32.SendScreenMessage(msgCreateTexture, 0, uintptr(unsafe.Pointer(&p)))
	if p.err != nil {
		return nil, p.err
	}
	return &textureImpl{
		size:   size,
		dc:     p.dc,
		bitmap: p.bitmap,
	}, nil
}

func handleCreateTexture(hwnd syscall.Handle, uMsg uint32, wParam, lParam uintptr) {
	// This code needs to run on Windows message pump thread.
	// Firstly, it calls GetDC(nil) and, according to Windows documentation
	// (https://msdn.microsoft.com/en-us/library/windows/desktop/dd144871(v=vs.85).aspx),
	// has to be released on the same thread.
	// Secondly, according to Windows documentation
	// (https://msdn.microsoft.com/en-us/library/windows/desktop/dd183489(v=vs.85).aspx),
	// ... thread that calls CreateCompatibleDC owns the HDC that is created.
	// When this thread is destroyed, the HDC is no longer valid. ...
	// So making Windows message pump thread own returned HDC makes DC
	// live as long as we want to.
	p := (*handleCreateTextureParams)(unsafe.Pointer(lParam))

	screenDC, err := win32.GetDC(0)
	if err != nil {
		p.err = err
		return
	}
	defer win32.ReleaseDC(0, screenDC)

	dc, err := _CreateCompatibleDC(screenDC)
	if err != nil {
		p.err = err
		return
	}
	bitmap, err := _CreateCompatibleBitmap(screenDC, int32(p.size.X), int32(p.size.Y))
	if err != nil {
		_DeleteDC(dc)
		p.err = err
		return
	}
	p.dc = dc
	p.bitmap = bitmap
}

func (t *textureImpl) Bounds() image.Rectangle {
	return image.Rectangle{Max: t.size}
}

func (t *textureImpl) Fill(r image.Rectangle, c color.Color, op draw.Op) {
	err := t.update(func(dc syscall.Handle) error {
		return fill(dc, r, c, op)
	})
	if err != nil {
		panic(err) // TODO handle error
	}
}

func (t *textureImpl) Release() {
	if err := t.release(); err != nil {
		panic(err) // TODO handle error
	}
}

func (t *textureImpl) release() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.released {
		return nil
	}
	t.released = true

	err := _DeleteObject(t.bitmap)
	if err != nil {
		return err
	}
	return _DeleteDC(t.dc)
}

func (t *textureImpl) Size() image.Point {
	return t.size
}

func (t *textureImpl) Upload(dp image.Point, src screen.Buffer, sr image.Rectangle) {
	err := t.update(func(dc syscall.Handle) error {
		return src.(*bufferImpl).blitToDC(dc, dp, sr)
	})
	if err != nil {
		panic(err) // TODO handle error
	}
}

// update prepares texture t for update and executes f over texture device
// context dc in a safe manner.
func (t *textureImpl) update(f func(dc syscall.Handle) error) (retErr error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.released {
		return errors.New("windriver: Texture.Upload called after Texture.Release")
	}

	// Select t.bitmap into t.dc, so our drawing gets recorded
	// into t.bitmap and not into 1x1 default bitmap created
	// during CreateCompatibleDC call.
	prev, err := _SelectObject(t.dc, t.bitmap)
	if err != nil {
		return err
	}
	defer func() {
		_, err2 := _SelectObject(t.dc, prev)
		if retErr == nil {
			retErr = err2
		}
	}()

	return f(t.dc)
}
