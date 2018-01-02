// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin
// +build 386 amd64
// +build !ios

package gldriver

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Cocoa -framework OpenGL
#include <OpenGL/gl3.h>
#import <Carbon/Carbon.h> // for HIToolbox/Events.h
#import <Cocoa/Cocoa.h>
#include <pthread.h>
#include <stdint.h>

void startDriver();
void stopDriver();
void makeCurrentContext(uintptr_t ctx);
void flushContext(uintptr_t ctx);
uintptr_t doNewWindow(int width, int height);
void doShowWindow(uintptr_t id);
void doCloseWindow(uintptr_t id);
uint64_t threadID();
*/
import "C"

import (
	"fmt"
	"log"
	"runtime"

	"golang.org/x/exp/shiny/screen"
	"golang.org/x/mobile/event/key"
	"golang.org/x/mobile/event/lifecycle"
	"golang.org/x/mobile/event/mouse"
	"golang.org/x/mobile/event/paint"
	"golang.org/x/mobile/event/size"
	"golang.org/x/mobile/geom"
	"golang.org/x/mobile/gl"
)

const useLifecycler = false

var initThreadID C.uint64_t

func init() {
	// Lock the goroutine responsible for initialization to an OS thread.
	// This means the goroutine running main (and calling startDriver below)
	// is locked to the OS thread that started the program. This is
	// necessary for the correct delivery of Cocoa events to the process.
	//
	// A discussion on this topic:
	// https://groups.google.com/forum/#!msg/golang-nuts/IiWZ2hUuLDA/SNKYYZBelsYJ
	runtime.LockOSThread()
	initThreadID = C.threadID()
}

func newWindow(opts *screen.NewWindowOptions) (uintptr, error) {
	width, height := optsSize(opts)
	return uintptr(C.doNewWindow(C.int(width), C.int(height))), nil
}

func showWindow(w *windowImpl) {
	w.glctxMu.Lock()
	w.glctx, w.worker = gl.NewContext()
	w.glctxMu.Unlock()

	C.doShowWindow(C.uintptr_t(w.id))
}

//export preparedOpenGL
func preparedOpenGL(id, ctx, vba uintptr) {
	theScreen.mu.Lock()
	w := theScreen.windows[id]
	theScreen.mu.Unlock()

	w.ctx = ctx
	go drawLoop(w, vba)
}

func closeWindow(id uintptr) {
	C.doCloseWindow(C.uintptr_t(id))
}

var mainCallback func(screen.Screen)

func main(f func(screen.Screen)) error {
	if tid := C.threadID(); tid != initThreadID {
		log.Fatalf("gldriver.Main called on thread %d, but gldriver.init ran on %d", tid, initThreadID)
	}

	mainCallback = f
	C.startDriver()
	return nil
}

//export driverStarted
func driverStarted() {
	go func() {
		mainCallback(theScreen)
		C.stopDriver()
	}()
}

//export drawgl
func drawgl(id uintptr) {
	theScreen.mu.Lock()
	w := theScreen.windows[id]
	theScreen.mu.Unlock()

	if w == nil {
		return // closing window
	}

	sendLifecycle(id, lifecycle.StageVisible)
	w.Send(paint.Event{External: true})
	<-w.drawDone
}

// drawLoop is the primary drawing loop.
//
// After Cocoa has created an NSWindow and called prepareOpenGL,
// it starts drawLoop on a locked goroutine to handle OpenGL calls.
//
// The screen is drawn every time a paint.Event is received, which can be
// triggered either by the user or by Cocoa via drawgl (for example, when
// the window is resized).
func drawLoop(w *windowImpl, vba uintptr) {
	runtime.LockOSThread()
	C.makeCurrentContext(C.uintptr_t(w.ctx.(uintptr)))

	// Starting in OS X 10.11 (El Capitan), the vertex array is
	// occasionally getting unbound when the context changes threads.
	//
	// Avoid this by binding it again.
	C.glBindVertexArray(C.GLuint(vba))
	if errno := C.glGetError(); errno != 0 {
		panic(fmt.Sprintf("gldriver: glBindVertexArray failed: %d", errno))
	}

	workAvailable := w.worker.WorkAvailable()

	// TODO(crawshaw): exit this goroutine on Release.
	for {
		select {
		case <-workAvailable:
			w.worker.DoWork()
		case <-w.publish:
		loop:
			for {
				select {
				case <-workAvailable:
					w.worker.DoWork()
				default:
					break loop
				}
			}
			C.flushContext(C.uintptr_t(w.ctx.(uintptr)))
			w.publishDone <- screen.PublishResult{}
		}
	}
}

//export setGeom
func setGeom(id uintptr, ppp float32, widthPx, heightPx int) {
	theScreen.mu.Lock()
	w := theScreen.windows[id]
	theScreen.mu.Unlock()

	if w == nil {
		return // closing window
	}

	sz := size.Event{
		WidthPx:     widthPx,
		HeightPx:    heightPx,
		WidthPt:     geom.Pt(float32(widthPx) / ppp),
		HeightPt:    geom.Pt(float32(heightPx) / ppp),
		PixelsPerPt: ppp,
	}

	w.szMu.Lock()
	w.sz = sz
	w.szMu.Unlock()

	w.Send(sz)
}

//export windowClosing
func windowClosing(id uintptr) {
	sendLifecycle(id, lifecycle.StageDead)
}

func sendWindowEvent(id uintptr, e interface{}) {
	theScreen.mu.Lock()
	w := theScreen.windows[id]
	theScreen.mu.Unlock()

	if w == nil {
		return // closing window
	}
	w.Send(e)
}

var mods = [...]struct {
	flags uint32
	code  uint16
	mod   key.Modifiers
}{
	// Left and right variants of modifier keys have their own masks,
	// but they are not documented. These were determined empirically.
	{1<<17 | 0x102, C.kVK_Shift, key.ModShift},
	{1<<17 | 0x104, C.kVK_RightShift, key.ModShift},
	{1<<18 | 0x101, C.kVK_Control, key.ModControl},
	// TODO key.ControlRight
	{1<<19 | 0x120, C.kVK_Option, key.ModAlt},
	{1<<19 | 0x140, C.kVK_RightOption, key.ModAlt},
	{1<<20 | 0x108, C.kVK_Command, key.ModMeta},
	{1<<20 | 0x110, C.kVK_Command, key.ModMeta}, // TODO: missing kVK_RightCommand
}

func cocoaMods(flags uint32) (m key.Modifiers) {
	for _, mod := range mods {
		if flags&mod.flags == mod.flags {
			m |= mod.mod
		}
	}
	return m
}

func cocoaMouseDir(ty int32) mouse.Direction {
	switch ty {
	case C.NSLeftMouseDown, C.NSRightMouseDown, C.NSOtherMouseDown:
		return mouse.DirPress
	case C.NSLeftMouseUp, C.NSRightMouseUp, C.NSOtherMouseUp:
		return mouse.DirRelease
	default: // dragged
		return mouse.DirNone
	}
}

func cocoaMouseButton(button int32) mouse.Button {
	switch button {
	case 0:
		return mouse.ButtonLeft
	case 1:
		return mouse.ButtonRight
	case 2:
		return mouse.ButtonMiddle
	default:
		return mouse.ButtonNone
	}
}

//export mouseEvent
func mouseEvent(id uintptr, x, y, dx, dy float32, ty, button int32, flags uint32) {
	cmButton := mouse.ButtonNone
	switch ty {
	default:
		cmButton = cocoaMouseButton(button)
	case C.NSMouseMoved, C.NSLeftMouseDragged, C.NSRightMouseDragged, C.NSOtherMouseDragged:
		// No-op.
	case C.NSScrollWheel:
		// TODO: handle horizontal scrolling
		button := mouse.ButtonWheelDown
		if dy < 0 {
			dy = -dy
			button = mouse.ButtonWheelUp
		}
		for delta := int(dy); delta != 0; delta-- {
			sendWindowEvent(id, mouse.Event{
				X:         x,
				Y:         y,
				Button:    button,
				Direction: mouse.DirNone,
				Modifiers: cocoaMods(flags),
			})
		}
		return
	}
	sendWindowEvent(id, mouse.Event{
		X:         x,
		Y:         y,
		Button:    cmButton,
		Direction: cocoaMouseDir(ty),
		Modifiers: cocoaMods(flags),
	})
}

//export keyEvent
func keyEvent(id uintptr, runeVal rune, dir uint8, code uint16, flags uint32) {
	sendWindowEvent(id, key.Event{
		Rune:      cocoaRune(runeVal),
		Direction: key.Direction(dir),
		Code:      cocoaKeyCode(code),
		Modifiers: cocoaMods(flags),
	})
}

//export flagEvent
func flagEvent(id uintptr, flags uint32) {
	for _, mod := range mods {
		if flags&mod.flags == mod.flags && lastFlags&mod.flags != mod.flags {
			keyEvent(id, -1, C.NSKeyDown, mod.code, flags)
		}
		if lastFlags&mod.flags == mod.flags && flags&mod.flags != mod.flags {
			keyEvent(id, -1, C.NSKeyUp, mod.code, flags)
		}
	}
	lastFlags = flags
}

var lastFlags uint32

// TODO: move sendLifecycle out of cocoa.go, for other platforms to use.
//
// TODO: figure out whether we need to mutex-protect windowImpl.lifecycleStage
// (by re-using glctxMu, if we're also reading windowImpl.glctx??)

func sendLifecycle(id uintptr, to lifecycle.Stage) {
	theScreen.mu.Lock()
	w := theScreen.windows[id]
	theScreen.mu.Unlock()

	if w == nil {
		return
	}
	if w.lifecycleStage == to {
		return
	}
	w.Send(lifecycle.Event{
		From:        w.lifecycleStage,
		To:          to,
		DrawContext: w.glctx,
	})
	w.lifecycleStage = to
}

func sendLifecycleAll(to lifecycle.Stage) {
	theScreen.mu.Lock()
	defer theScreen.mu.Unlock()
	for _, w := range theScreen.windows {
		if w.lifecycleStage == to {
			continue
		}
		w.Send(lifecycle.Event{
			From:        w.lifecycleStage,
			To:          to,
			DrawContext: w.glctx,
		})
		w.lifecycleStage = to
	}
}

//export lifecycleDeadAll
func lifecycleDeadAll() { sendLifecycleAll(lifecycle.StageDead) }

//export lifecycleAliveAll
func lifecycleAliveAll() { sendLifecycleAll(lifecycle.StageAlive) }

//export lifecycleVisible
func lifecycleVisible(id uintptr) { sendLifecycle(id, lifecycle.StageVisible) }

//export lifecycleFocused
func lifecycleFocused(id uintptr) { sendLifecycle(id, lifecycle.StageFocused) }

// cocoaRune marks the Carbon/Cocoa private-range unicode rune representing
// a non-unicode key event to -1, used for Rune in the key package.
//
// http://www.unicode.org/Public/MAPPINGS/VENDORS/APPLE/CORPCHAR.TXT
func cocoaRune(r rune) rune {
	if '\uE000' <= r && r <= '\uF8FF' {
		return -1
	}
	return r
}

// cocoaKeyCode converts a Carbon/Cocoa virtual key code number
// into the standard keycodes used by the key package.
//
// To get a sense of the key map, see the diagram on
//	http://boredzo.org/blog/archives/2007-05-22/virtual-key-codes
func cocoaKeyCode(vkcode uint16) key.Code {
	switch vkcode {
	case C.kVK_ANSI_A:
		return key.CodeA
	case C.kVK_ANSI_B:
		return key.CodeB
	case C.kVK_ANSI_C:
		return key.CodeC
	case C.kVK_ANSI_D:
		return key.CodeD
	case C.kVK_ANSI_E:
		return key.CodeE
	case C.kVK_ANSI_F:
		return key.CodeF
	case C.kVK_ANSI_G:
		return key.CodeG
	case C.kVK_ANSI_H:
		return key.CodeH
	case C.kVK_ANSI_I:
		return key.CodeI
	case C.kVK_ANSI_J:
		return key.CodeJ
	case C.kVK_ANSI_K:
		return key.CodeK
	case C.kVK_ANSI_L:
		return key.CodeL
	case C.kVK_ANSI_M:
		return key.CodeM
	case C.kVK_ANSI_N:
		return key.CodeN
	case C.kVK_ANSI_O:
		return key.CodeO
	case C.kVK_ANSI_P:
		return key.CodeP
	case C.kVK_ANSI_Q:
		return key.CodeQ
	case C.kVK_ANSI_R:
		return key.CodeR
	case C.kVK_ANSI_S:
		return key.CodeS
	case C.kVK_ANSI_T:
		return key.CodeT
	case C.kVK_ANSI_U:
		return key.CodeU
	case C.kVK_ANSI_V:
		return key.CodeV
	case C.kVK_ANSI_W:
		return key.CodeW
	case C.kVK_ANSI_X:
		return key.CodeX
	case C.kVK_ANSI_Y:
		return key.CodeY
	case C.kVK_ANSI_Z:
		return key.CodeZ
	case C.kVK_ANSI_1:
		return key.Code1
	case C.kVK_ANSI_2:
		return key.Code2
	case C.kVK_ANSI_3:
		return key.Code3
	case C.kVK_ANSI_4:
		return key.Code4
	case C.kVK_ANSI_5:
		return key.Code5
	case C.kVK_ANSI_6:
		return key.Code6
	case C.kVK_ANSI_7:
		return key.Code7
	case C.kVK_ANSI_8:
		return key.Code8
	case C.kVK_ANSI_9:
		return key.Code9
	case C.kVK_ANSI_0:
		return key.Code0
	// TODO: move the rest of these codes to constants in key.go
	// if we are happy with them.
	case C.kVK_Return:
		return key.CodeReturnEnter
	case C.kVK_Escape:
		return key.CodeEscape
	case C.kVK_Delete:
		return key.CodeDeleteBackspace
	case C.kVK_Tab:
		return key.CodeTab
	case C.kVK_Space:
		return key.CodeSpacebar
	case C.kVK_ANSI_Minus:
		return key.CodeHyphenMinus
	case C.kVK_ANSI_Equal:
		return key.CodeEqualSign
	case C.kVK_ANSI_LeftBracket:
		return key.CodeLeftSquareBracket
	case C.kVK_ANSI_RightBracket:
		return key.CodeRightSquareBracket
	case C.kVK_ANSI_Backslash:
		return key.CodeBackslash
	// 50: Keyboard Non-US "#" and ~
	case C.kVK_ANSI_Semicolon:
		return key.CodeSemicolon
	case C.kVK_ANSI_Quote:
		return key.CodeApostrophe
	case C.kVK_ANSI_Grave:
		return key.CodeGraveAccent
	case C.kVK_ANSI_Comma:
		return key.CodeComma
	case C.kVK_ANSI_Period:
		return key.CodeFullStop
	case C.kVK_ANSI_Slash:
		return key.CodeSlash
	case C.kVK_CapsLock:
		return key.CodeCapsLock
	case C.kVK_F1:
		return key.CodeF1
	case C.kVK_F2:
		return key.CodeF2
	case C.kVK_F3:
		return key.CodeF3
	case C.kVK_F4:
		return key.CodeF4
	case C.kVK_F5:
		return key.CodeF5
	case C.kVK_F6:
		return key.CodeF6
	case C.kVK_F7:
		return key.CodeF7
	case C.kVK_F8:
		return key.CodeF8
	case C.kVK_F9:
		return key.CodeF9
	case C.kVK_F10:
		return key.CodeF10
	case C.kVK_F11:
		return key.CodeF11
	case C.kVK_F12:
		return key.CodeF12
	// 70: PrintScreen
	// 71: Scroll Lock
	// 72: Pause
	// 73: Insert
	case C.kVK_Home:
		return key.CodeHome
	case C.kVK_PageUp:
		return key.CodePageUp
	case C.kVK_ForwardDelete:
		return key.CodeDeleteForward
	case C.kVK_End:
		return key.CodeEnd
	case C.kVK_PageDown:
		return key.CodePageDown
	case C.kVK_RightArrow:
		return key.CodeRightArrow
	case C.kVK_LeftArrow:
		return key.CodeLeftArrow
	case C.kVK_DownArrow:
		return key.CodeDownArrow
	case C.kVK_UpArrow:
		return key.CodeUpArrow
	case C.kVK_ANSI_KeypadClear:
		return key.CodeKeypadNumLock
	case C.kVK_ANSI_KeypadDivide:
		return key.CodeKeypadSlash
	case C.kVK_ANSI_KeypadMultiply:
		return key.CodeKeypadAsterisk
	case C.kVK_ANSI_KeypadMinus:
		return key.CodeKeypadHyphenMinus
	case C.kVK_ANSI_KeypadPlus:
		return key.CodeKeypadPlusSign
	case C.kVK_ANSI_KeypadEnter:
		return key.CodeKeypadEnter
	case C.kVK_ANSI_Keypad1:
		return key.CodeKeypad1
	case C.kVK_ANSI_Keypad2:
		return key.CodeKeypad2
	case C.kVK_ANSI_Keypad3:
		return key.CodeKeypad3
	case C.kVK_ANSI_Keypad4:
		return key.CodeKeypad4
	case C.kVK_ANSI_Keypad5:
		return key.CodeKeypad5
	case C.kVK_ANSI_Keypad6:
		return key.CodeKeypad6
	case C.kVK_ANSI_Keypad7:
		return key.CodeKeypad7
	case C.kVK_ANSI_Keypad8:
		return key.CodeKeypad8
	case C.kVK_ANSI_Keypad9:
		return key.CodeKeypad9
	case C.kVK_ANSI_Keypad0:
		return key.CodeKeypad0
	case C.kVK_ANSI_KeypadDecimal:
		return key.CodeKeypadFullStop
	case C.kVK_ANSI_KeypadEquals:
		return key.CodeKeypadEqualSign
	case C.kVK_F13:
		return key.CodeF13
	case C.kVK_F14:
		return key.CodeF14
	case C.kVK_F15:
		return key.CodeF15
	case C.kVK_F16:
		return key.CodeF16
	case C.kVK_F17:
		return key.CodeF17
	case C.kVK_F18:
		return key.CodeF18
	case C.kVK_F19:
		return key.CodeF19
	case C.kVK_F20:
		return key.CodeF20
	// 116: Keyboard Execute
	case C.kVK_Help:
		return key.CodeHelp
	// 118: Keyboard Menu
	// 119: Keyboard Select
	// 120: Keyboard Stop
	// 121: Keyboard Again
	// 122: Keyboard Undo
	// 123: Keyboard Cut
	// 124: Keyboard Copy
	// 125: Keyboard Paste
	// 126: Keyboard Find
	case C.kVK_Mute:
		return key.CodeMute
	case C.kVK_VolumeUp:
		return key.CodeVolumeUp
	case C.kVK_VolumeDown:
		return key.CodeVolumeDown
	// 130: Keyboard Locking Caps Lock
	// 131: Keyboard Locking Num Lock
	// 132: Keyboard Locking Scroll Lock
	// 133: Keyboard Comma
	// 134: Keyboard Equal Sign
	// ...: Bunch of stuff
	case C.kVK_Control:
		return key.CodeLeftControl
	case C.kVK_Shift:
		return key.CodeLeftShift
	case C.kVK_Option:
		return key.CodeLeftAlt
	case C.kVK_Command:
		return key.CodeLeftGUI
	case C.kVK_RightControl:
		return key.CodeRightControl
	case C.kVK_RightShift:
		return key.CodeRightShift
	case C.kVK_RightOption:
		return key.CodeRightAlt
	// TODO key.CodeRightGUI
	default:
		return key.CodeUnknown
	}
}
