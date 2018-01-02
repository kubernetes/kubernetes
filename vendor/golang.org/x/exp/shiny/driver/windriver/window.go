// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package windriver

// TODO: implement a back buffer.

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"math"
	"syscall"
	"unsafe"

	"golang.org/x/exp/shiny/driver/internal/drawer"
	"golang.org/x/exp/shiny/driver/internal/event"
	"golang.org/x/exp/shiny/driver/internal/win32"
	"golang.org/x/exp/shiny/screen"
	"golang.org/x/image/math/f64"
	"golang.org/x/mobile/event/key"
	"golang.org/x/mobile/event/lifecycle"
	"golang.org/x/mobile/event/mouse"
	"golang.org/x/mobile/event/paint"
	"golang.org/x/mobile/event/size"
)

type windowImpl struct {
	hwnd syscall.Handle

	event.Deque

	sz             size.Event
	lifecycleStage lifecycle.Stage
}

func (w *windowImpl) Release() {
	win32.Release(w.hwnd)
}

func (w *windowImpl) Upload(dp image.Point, src screen.Buffer, sr image.Rectangle) {
	src.(*bufferImpl).preUpload()
	defer src.(*bufferImpl).postUpload()

	w.execCmd(&cmd{
		id:     cmdUpload,
		dp:     dp,
		buffer: src.(*bufferImpl),
		sr:     sr,
	})
}

func (w *windowImpl) Fill(dr image.Rectangle, src color.Color, op draw.Op) {
	w.execCmd(&cmd{
		id:    cmdFill,
		dr:    dr,
		color: src,
		op:    op,
	})
}

func (w *windowImpl) Draw(src2dst f64.Aff3, src screen.Texture, sr image.Rectangle, op draw.Op, opts *screen.DrawOptions) {
	if op != draw.Src && op != draw.Over {
		// TODO:
		return
	}
	w.execCmd(&cmd{
		id:      cmdDraw,
		src2dst: src2dst,
		texture: src.(*textureImpl).bitmap,
		sr:      sr,
		op:      op,
	})
}

func drawWindow(dc syscall.Handle, src2dst f64.Aff3, src syscall.Handle, sr image.Rectangle, op draw.Op) (retErr error) {
	var dr image.Rectangle
	if src2dst[1] != 0 || src2dst[3] != 0 {
		// general drawing
		dr = sr.Sub(sr.Min)

		prevmode, err := _SetGraphicsMode(dc, _GM_ADVANCED)
		if err != nil {
			return err
		}
		defer func() {
			_, err := _SetGraphicsMode(dc, prevmode)
			if retErr == nil {
				retErr = err
			}
		}()

		x := _XFORM{
			eM11: +float32(src2dst[0]),
			eM12: -float32(src2dst[1]),
			eM21: -float32(src2dst[3]),
			eM22: +float32(src2dst[4]),
			eDx:  +float32(src2dst[2]),
			eDy:  +float32(src2dst[5]),
		}
		err = _SetWorldTransform(dc, &x)
		if err != nil {
			return err
		}
		defer func() {
			err := _ModifyWorldTransform(dc, nil, _MWT_IDENTITY)
			if retErr == nil {
				retErr = err
			}
		}()
	} else if src2dst[0] == 1 && src2dst[4] == 1 {
		// copy bitmap
		dp := image.Point{int(src2dst[2]), int(src2dst[5])}
		dr = sr.Add(dp.Sub(sr.Min))
	} else {
		// scale bitmap
		dstXMin := float64(sr.Min.X)*src2dst[0] + src2dst[2]
		dstXMax := float64(sr.Max.X)*src2dst[0] + src2dst[2]
		if dstXMin > dstXMax {
			// TODO: check if this (and below) works when src2dst[0] < 0.
			dstXMin, dstXMax = dstXMax, dstXMin
		}
		dstYMin := float64(sr.Min.Y)*src2dst[4] + src2dst[5]
		dstYMax := float64(sr.Max.Y)*src2dst[4] + src2dst[5]
		if dstYMin > dstYMax {
			// TODO: check if this (and below) works when src2dst[4] < 0.
			dstYMin, dstYMax = dstYMax, dstYMin
		}
		dr = image.Rectangle{
			image.Point{int(math.Floor(dstXMin)), int(math.Floor(dstYMin))},
			image.Point{int(math.Ceil(dstXMax)), int(math.Ceil(dstYMax))},
		}
	}
	return copyBitmapToDC(dc, dr, src, sr, op)
}

func (w *windowImpl) Copy(dp image.Point, src screen.Texture, sr image.Rectangle, op draw.Op, opts *screen.DrawOptions) {
	drawer.Copy(w, dp, src, sr, op, opts)
}

func (w *windowImpl) Scale(dr image.Rectangle, src screen.Texture, sr image.Rectangle, op draw.Op, opts *screen.DrawOptions) {
	drawer.Scale(w, dr, src, sr, op, opts)
}

func (w *windowImpl) Publish() screen.PublishResult {
	// TODO
	return screen.PublishResult{}
}

func init() {
	send := func(hwnd syscall.Handle, e interface{}) {
		theScreen.mu.Lock()
		w := theScreen.windows[hwnd]
		theScreen.mu.Unlock()

		w.Send(e)
	}
	win32.MouseEvent = func(hwnd syscall.Handle, e mouse.Event) { send(hwnd, e) }
	win32.PaintEvent = func(hwnd syscall.Handle, e paint.Event) { send(hwnd, e) }
	win32.KeyEvent = func(hwnd syscall.Handle, e key.Event) { send(hwnd, e) }
	win32.LifecycleEvent = lifecycleEvent
	win32.SizeEvent = sizeEvent
}

func lifecycleEvent(hwnd syscall.Handle, to lifecycle.Stage) {
	theScreen.mu.Lock()
	w := theScreen.windows[hwnd]
	theScreen.mu.Unlock()

	if w.lifecycleStage == to {
		return
	}
	w.Send(lifecycle.Event{
		From: w.lifecycleStage,
		To:   to,
	})
	w.lifecycleStage = to
}

func sizeEvent(hwnd syscall.Handle, e size.Event) {
	theScreen.mu.Lock()
	w := theScreen.windows[hwnd]
	theScreen.mu.Unlock()

	w.Send(e)

	if e != w.sz {
		w.sz = e
		w.Send(paint.Event{})
	}
}

// cmd is used to carry parameters between user code
// and Windows message pump thread.
type cmd struct {
	id  int
	err error

	src2dst f64.Aff3
	sr      image.Rectangle
	dp      image.Point
	dr      image.Rectangle
	color   color.Color
	op      draw.Op
	texture syscall.Handle
	buffer  *bufferImpl
}

const (
	cmdDraw = iota
	cmdFill
	cmdUpload
)

var msgCmd = win32.AddWindowMsg(handleCmd)

func (w *windowImpl) execCmd(c *cmd) {
	win32.SendMessage(w.hwnd, msgCmd, 0, uintptr(unsafe.Pointer(c)))
	if c.err != nil {
		panic(fmt.Sprintf("execCmd faild for cmd.id=%d: %v", c.id, c.err)) // TODO handle errors
	}
}

func handleCmd(hwnd syscall.Handle, uMsg uint32, wParam, lParam uintptr) {
	c := (*cmd)(unsafe.Pointer(lParam))

	dc, err := win32.GetDC(hwnd)
	if err != nil {
		c.err = err
		return
	}
	defer win32.ReleaseDC(hwnd, dc)

	switch c.id {
	case cmdDraw:
		c.err = drawWindow(dc, c.src2dst, c.texture, c.sr, c.op)
	case cmdFill:
		c.err = fill(dc, c.dr, c.color, c.op)
	case cmdUpload:
		// TODO: adjust if dp is outside dst bounds, or sr is outside buffer bounds.
		dr := c.sr.Add(c.dp.Sub(c.sr.Min))
		c.err = copyBitmapToDC(dc, dr, c.buffer.hbitmap, c.sr, draw.Src)
	default:
		c.err = fmt.Errorf("unknown command id=%d", c.id)
	}
	return
}
