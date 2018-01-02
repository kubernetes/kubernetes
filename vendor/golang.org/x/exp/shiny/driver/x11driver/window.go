// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x11driver

// TODO: implement a back buffer.

import (
	"image"
	"image/color"
	"image/draw"
	"sync"

	"github.com/BurntSushi/xgb"
	"github.com/BurntSushi/xgb/render"
	"github.com/BurntSushi/xgb/xproto"

	"golang.org/x/exp/shiny/driver/internal/drawer"
	"golang.org/x/exp/shiny/driver/internal/event"
	"golang.org/x/exp/shiny/driver/internal/lifecycler"
	"golang.org/x/exp/shiny/driver/internal/x11key"
	"golang.org/x/exp/shiny/screen"
	"golang.org/x/image/math/f64"
	"golang.org/x/mobile/event/key"
	"golang.org/x/mobile/event/mouse"
	"golang.org/x/mobile/event/paint"
	"golang.org/x/mobile/event/size"
	"golang.org/x/mobile/geom"
)

type windowImpl struct {
	s *screenImpl

	xw xproto.Window
	xg xproto.Gcontext
	xp render.Picture

	event.Deque
	xevents chan xgb.Event

	// This next group of variables are mutable, but are only modified in the
	// screenImpl.run goroutine.
	width, height int

	lifecycler lifecycler.State

	mu       sync.Mutex
	released bool
}

func (w *windowImpl) Release() {
	w.mu.Lock()
	released := w.released
	w.released = true
	w.mu.Unlock()

	// TODO: call w.lifecycler.SetDead and w.lifecycler.SendEvent, a la
	// handling atomWMDeleteWindow?

	if released {
		return
	}
	render.FreePicture(w.s.xc, w.xp)
	xproto.FreeGC(w.s.xc, w.xg)
	xproto.DestroyWindow(w.s.xc, w.xw)
}

func (w *windowImpl) Upload(dp image.Point, src screen.Buffer, sr image.Rectangle) {
	src.(*bufferImpl).upload(xproto.Drawable(w.xw), w.xg, w.s.xsi.RootDepth, dp, sr)
}

func (w *windowImpl) Fill(dr image.Rectangle, src color.Color, op draw.Op) {
	fill(w.s.xc, w.xp, dr, src, op)
}

func (w *windowImpl) Draw(src2dst f64.Aff3, src screen.Texture, sr image.Rectangle, op draw.Op, opts *screen.DrawOptions) {
	src.(*textureImpl).draw(w.xp, &src2dst, sr, op, opts)
}

func (w *windowImpl) Copy(dp image.Point, src screen.Texture, sr image.Rectangle, op draw.Op, opts *screen.DrawOptions) {
	drawer.Copy(w, dp, src, sr, op, opts)
}

func (w *windowImpl) Scale(dr image.Rectangle, src screen.Texture, sr image.Rectangle, op draw.Op, opts *screen.DrawOptions) {
	drawer.Scale(w, dr, src, sr, op, opts)
}

func (w *windowImpl) Publish() screen.PublishResult {
	// TODO.
	return screen.PublishResult{}
}

func (w *windowImpl) handleConfigureNotify(ev xproto.ConfigureNotifyEvent) {
	// TODO: does the order of these lifecycle and size events matter? Should
	// they really be a single, atomic event?
	w.lifecycler.SetVisible((int(ev.X)+int(ev.Width)) > 0 && (int(ev.Y)+int(ev.Height)) > 0)
	w.lifecycler.SendEvent(w)

	newWidth, newHeight := int(ev.Width), int(ev.Height)
	if w.width == newWidth && w.height == newHeight {
		return
	}
	w.width, w.height = newWidth, newHeight
	w.Send(size.Event{
		WidthPx:     newWidth,
		HeightPx:    newHeight,
		WidthPt:     geom.Pt(newWidth),
		HeightPt:    geom.Pt(newHeight),
		PixelsPerPt: w.s.pixelsPerPt,
	})
}

func (w *windowImpl) handleExpose() {
	w.Send(paint.Event{})
}

func (w *windowImpl) handleKey(detail xproto.Keycode, state uint16, dir key.Direction) {
	r, c := w.s.keysyms.Lookup(uint8(detail), state)
	w.Send(key.Event{
		Rune:      r,
		Code:      c,
		Modifiers: x11key.KeyModifiers(state),
		Direction: dir,
	})
}

func (w *windowImpl) handleMouse(x, y int16, b xproto.Button, state uint16, dir mouse.Direction) {
	// TODO: should a mouse.Event have a separate MouseModifiers field, for
	// which buttons are pressed during a mouse move?
	btn := mouse.Button(b)
	switch b {
	case 4:
		btn = mouse.ButtonWheelUp
	case 5:
		btn = mouse.ButtonWheelDown
	}
	w.Send(mouse.Event{
		X:         float32(x),
		Y:         float32(y),
		Button:    btn,
		Modifiers: x11key.KeyModifiers(state),
		Direction: dir,
	})
}
