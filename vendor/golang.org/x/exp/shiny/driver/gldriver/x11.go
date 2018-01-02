// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux,!android

package gldriver

/*
#cgo LDFLAGS: -lEGL -lGLESv2 -lX11

#include <stdint.h>

void startDriver();
void processEvents();
void makeCurrent(uintptr_t ctx);
void swapBuffers(uintptr_t ctx);
void doCloseWindow(uintptr_t id);
uintptr_t doNewWindow(int width, int height);
uintptr_t doShowWindow(uintptr_t id);
*/
import "C"
import (
	"errors"
	"runtime"
	"time"

	"golang.org/x/exp/shiny/driver/internal/x11key"
	"golang.org/x/exp/shiny/screen"
	"golang.org/x/mobile/event/key"
	"golang.org/x/mobile/event/mouse"
	"golang.org/x/mobile/event/paint"
	"golang.org/x/mobile/event/size"
	"golang.org/x/mobile/geom"
	"golang.org/x/mobile/gl"
)

const useLifecycler = true

var theKeysyms x11key.KeysymTable

func init() {
	// It might not be necessary, but it probably doesn't hurt to try to make
	// 'the main thread' be 'the X11 / OpenGL thread'.
	runtime.LockOSThread()
}

func newWindow(opts *screen.NewWindowOptions) (uintptr, error) {
	width, height := optsSize(opts)
	retc := make(chan uintptr)
	uic <- uiClosure{
		f: func() uintptr {
			return uintptr(C.doNewWindow(C.int(width), C.int(height)))
		},
		retc: retc,
	}
	return <-retc, nil
}

func showWindow(w *windowImpl) {
	retc := make(chan uintptr)
	uic <- uiClosure{
		f: func() uintptr {
			return uintptr(C.doShowWindow(C.uintptr_t(w.id)))
		},
		retc: retc,
	}
	w.ctx = <-retc
	w.glctxMu.Lock()
	w.glctx, w.worker = glctx, worker
	w.glctxMu.Unlock()
	go drawLoop(w)
}

func closeWindow(id uintptr) {
	uic <- uiClosure{
		f: func() uintptr {
			C.doCloseWindow(C.uintptr_t(id))
			return 0
		},
	}
}

func drawLoop(w *windowImpl) {
	glcontextc <- w.ctx.(uintptr)
	go func() {
		for range w.publish {
			publishc <- w
		}
	}()
}

var (
	glcontextc = make(chan uintptr)
	publishc   = make(chan *windowImpl)
	uic        = make(chan uiClosure)

	// TODO: don't assume that there is only one window, and hence only
	// one (global) GL context.
	//
	// TODO: should we be able to make a shiny.Texture before having a
	// shiny.Window's GL context? Should something like gl.IsProgram be a
	// method instead of a function, and have each shiny.Window have its own
	// gl.Context?
	glctx  gl.Context
	worker gl.Worker
)

// uiClosure is a closure to be run on C's UI thread.
type uiClosure struct {
	f    func() uintptr
	retc chan uintptr
}

func main(f func(screen.Screen)) error {
	if gl.Version() == "GL_ES_2_0" {
		return errors.New("gldriver: ES 3 required on X11")
	}
	C.startDriver()
	glctx, worker = gl.NewContext()

	closec := make(chan struct{})
	go func() {
		f(theScreen)
		close(closec)
	}()

	// heartbeat is a channel that, at regular intervals, directs the select
	// below to also consider X11 events, not just Go events (channel
	// communications).
	//
	// TODO: select instead of poll. Note that knowing whether to call
	// C.processEvents needs to select on a file descriptor, and the other
	// cases below select on Go channels.
	heartbeat := time.NewTicker(time.Second / 60)
	workAvailable := worker.WorkAvailable()

	for {
		select {
		case <-closec:
			return nil
		case ctx := <-glcontextc:
			// TODO: do we need to synchronize with seeing a size event for
			// this window's context before or after calling makeCurrent?
			// Otherwise, are we racing with the gl.Viewport call? I've
			// occasionally seen a stale viewport, if the window manager sets
			// the window width and height to something other than that
			// requested by XCreateWindow, but it's not easily reproducible.
			C.makeCurrent(C.uintptr_t(ctx))
		case w := <-publishc:
			C.swapBuffers(C.uintptr_t(w.ctx.(uintptr)))
			w.publishDone <- screen.PublishResult{}
		case req := <-uic:
			ret := req.f()
			if req.retc != nil {
				req.retc <- ret
			}
		case <-heartbeat.C:
			C.processEvents()
		case <-workAvailable:
			worker.DoWork()
		}
	}
}

//export onExpose
func onExpose(id uintptr) {
	theScreen.mu.Lock()
	w := theScreen.windows[id]
	theScreen.mu.Unlock()

	if w == nil {
		return
	}

	w.Send(paint.Event{External: true})
}

//export onKeysym
func onKeysym(k, unshifted, shifted uint32) {
	theKeysyms[k][0] = unshifted
	theKeysyms[k][1] = shifted
}

//export onKey
func onKey(id uintptr, state uint16, detail, dir uint8) {
	theScreen.mu.Lock()
	w := theScreen.windows[id]
	theScreen.mu.Unlock()

	if w == nil {
		return
	}

	r, c := theKeysyms.Lookup(detail, state)
	w.Send(key.Event{
		Rune:      r,
		Code:      c,
		Modifiers: x11key.KeyModifiers(state),
		Direction: key.Direction(dir),
	})
}

//export onMouse
func onMouse(id uintptr, x, y int32, state uint16, button, dir uint8) {
	theScreen.mu.Lock()
	w := theScreen.windows[id]
	theScreen.mu.Unlock()

	if w == nil {
		return
	}

	// TODO: should a mouse.Event have a separate MouseModifiers field, for
	// which buttons are pressed during a mouse move?
	w.Send(mouse.Event{
		X:         float32(x),
		Y:         float32(y),
		Button:    mouse.Button(button),
		Modifiers: x11key.KeyModifiers(state),
		Direction: mouse.Direction(dir),
	})
}

//export onFocus
func onFocus(id uintptr, focused bool) {
	theScreen.mu.Lock()
	w := theScreen.windows[id]
	theScreen.mu.Unlock()

	if w == nil {
		return
	}

	w.lifecycler.SetFocused(focused)
	w.lifecycler.SendEvent(w)
}

//export onConfigure
func onConfigure(id uintptr, x, y, width, height int32) {
	theScreen.mu.Lock()
	w := theScreen.windows[id]
	theScreen.mu.Unlock()

	if w == nil {
		return
	}

	// TODO: should this really be done on the receiving end of the w.Events()
	// channel, in the same goroutine as other GL calls in the app's 'business
	// logic'?
	go func() {
		w.glctxMu.Lock()
		w.glctx.Viewport(0, 0, int(width), int(height))
		w.glctxMu.Unlock()
	}()

	w.lifecycler.SetVisible(x+width > 0 && y+height > 0)
	w.lifecycler.SendEvent(w)

	sz := size.Event{
		WidthPx:  int(width),
		HeightPx: int(height),
		WidthPt:  geom.Pt(width),
		HeightPt: geom.Pt(height),
		// TODO: don't assume 72 DPI. DisplayWidth and DisplayWidthMM is
		// probably the best place to start looking.
		PixelsPerPt: 1,
	}

	w.szMu.Lock()
	w.sz = sz
	w.szMu.Unlock()

	w.Send(sz)
}

//export onDeleteWindow
func onDeleteWindow(id uintptr) {
	theScreen.mu.Lock()
	w := theScreen.windows[id]
	theScreen.mu.Unlock()

	if w == nil {
		return
	}

	w.lifecycler.SetDead(true)
	w.lifecycler.SendEvent(w)
}
