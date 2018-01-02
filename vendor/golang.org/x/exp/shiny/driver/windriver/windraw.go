// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package windriver

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"syscall"
	"unsafe"
)

func mkbitmap(size image.Point) (syscall.Handle, *byte, error) {
	bi := _BITMAPINFO{
		Header: _BITMAPINFOHEADER{
			Size:        uint32(unsafe.Sizeof(_BITMAPINFOHEADER{})),
			Width:       int32(size.X),
			Height:      -int32(size.Y), // negative height to force top-down drawing
			Planes:      1,
			BitCount:    32,
			Compression: _BI_RGB,
			SizeImage:   uint32(size.X * size.Y * 4),
		},
	}

	var ppvBits *byte
	bitmap, err := _CreateDIBSection(0, &bi, _DIB_RGB_COLORS, &ppvBits, 0, 0)
	if err != nil {
		return 0, nil, err
	}
	return bitmap, ppvBits, nil
}

var blendOverFunc = _BLENDFUNCTION{
	BlendOp:             _AC_SRC_OVER,
	BlendFlags:          0,
	SourceConstantAlpha: 255,           // only use per-pixel alphas
	AlphaFormat:         _AC_SRC_ALPHA, // premultiplied
}

func copyBitmapToDC(dc syscall.Handle, dr image.Rectangle, src syscall.Handle, sr image.Rectangle, op draw.Op) (retErr error) {
	memdc, err := _CreateCompatibleDC(dc)
	if err != nil {
		return err
	}
	defer _DeleteDC(memdc)

	prev, err := _SelectObject(memdc, src)
	if err != nil {
		return err
	}
	defer func() {
		_, err2 := _SelectObject(memdc, prev)
		if retErr == nil {
			retErr = err2
		}
	}()

	switch op {
	case draw.Src:
		return _StretchBlt(dc, int32(dr.Min.X), int32(dr.Min.Y), int32(dr.Dx()), int32(dr.Dy()),
			memdc, int32(sr.Min.X), int32(sr.Min.Y), int32(sr.Dx()), int32(sr.Dy()), _SRCCOPY)
	case draw.Over:
		return _AlphaBlend(dc, int32(dr.Min.X), int32(dr.Min.Y), int32(dr.Dx()), int32(dr.Dy()),
			memdc, int32(sr.Min.X), int32(sr.Min.Y), int32(sr.Dx()), int32(sr.Dy()), blendOverFunc.ToUintptr())
	default:
		return fmt.Errorf("windriver: invalid draw operation %v", op)
	}
}

func fill(dc syscall.Handle, dr image.Rectangle, c color.Color, op draw.Op) error {
	r, g, b, a := c.RGBA()
	r >>= 8
	g >>= 8
	b >>= 8
	a >>= 8

	if op == draw.Src {
		color := _RGB(byte(r), byte(g), byte(b))
		brush, err := _CreateSolidBrush(color)
		if err != nil {
			return err
		}
		defer _DeleteObject(brush)

		rect := _RECT{
			Left:   int32(dr.Min.X),
			Top:    int32(dr.Min.Y),
			Right:  int32(dr.Max.X),
			Bottom: int32(dr.Max.Y),
		}
		return _FillRect(dc, &rect, brush)
	}

	// AlphaBlend will stretch the input image (using StretchBlt's
	// COLORONCOLOR mode) to fill the output rectangle. Testing
	// this shows that the result appears to be the same as if we had
	// used a MxN bitmap instead.
	sr := image.Rect(0, 0, 1, 1)
	bitmap, bitvalues, err := mkbitmap(sr.Size())
	if err != nil {
		return err
	}
	defer _DeleteObject(bitmap) // TODO handle error?

	color := _COLORREF((a << 24) | (r << 16) | (g << 8) | b)
	*(*_COLORREF)(unsafe.Pointer(bitvalues)) = color

	return copyBitmapToDC(dc, dr, bitmap, sr, draw.Over)
}
