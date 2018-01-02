// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

// Package win32 implements a partial shiny screen driver using the Win32 API.
// It provides window, lifecycle, key, and mouse management, but no drawing.
// That is left to windriver (using GDI) or gldriver (using DirectX via ANGLE).
package win32 // import "golang.org/x/exp/shiny/driver/internal/win32"

import (
	"fmt"
	"runtime"
	"sync"
	"syscall"
	"unsafe"

	"golang.org/x/exp/shiny/screen"
	"golang.org/x/mobile/event/key"
	"golang.org/x/mobile/event/lifecycle"
	"golang.org/x/mobile/event/mouse"
	"golang.org/x/mobile/event/paint"
	"golang.org/x/mobile/event/size"
	"golang.org/x/mobile/geom"
)

// screenHWND is the handle to the "Screen window".
// The Screen window encapsulates all screen.Screen operations
// in an actual Windows window so they all run on the main thread.
// Since any messages sent to a window will be executed on the
// main thread, we can safely use the messages below.
var screenHWND syscall.Handle

const (
	msgCreateWindow = _WM_USER + iota
	msgMainCallback
	msgShow
	msgQuit
	msgLast
)

// userWM is used to generate private (WM_USER and above) window message IDs
// for use by screenWindowWndProc and windowWndProc.
type userWM struct {
	sync.Mutex
	id uint32
}

func (m *userWM) next() uint32 {
	m.Lock()
	if m.id == 0 {
		m.id = msgLast
	}
	r := m.id
	m.id++
	m.Unlock()
	return r
}

var currentUserWM userWM

func newWindow(opts *screen.NewWindowOptions) (syscall.Handle, error) {
	// TODO(brainman): convert windowClass to *uint16 once (in initWindowClass)
	wcname, err := syscall.UTF16PtrFromString(windowClass)
	if err != nil {
		return 0, err
	}
	title, err := syscall.UTF16PtrFromString("Shiny Window")
	if err != nil {
		return 0, err
	}
	w, h := _CW_USEDEFAULT, _CW_USEDEFAULT
	if opts != nil {
		if opts.Width > 0 {
			w = opts.Width
		}
		if opts.Height > 0 {
			h = opts.Height
		}
	}
	hwnd, err := _CreateWindowEx(0,
		wcname, title,
		_WS_OVERLAPPEDWINDOW,
		_CW_USEDEFAULT, _CW_USEDEFAULT,
		int32(w), int32(h),
		0, 0, hThisInstance, 0)
	if err != nil {
		return 0, err
	}
	// TODO(andlabs): use proper nCmdShow
	// TODO(andlabs): call UpdateWindow()

	return hwnd, nil
}

// Show shows a newly created window.
// It sends the appropriate lifecycle events, makes the window appear
// on the screen, and sends an initial size event.
//
// This is a separate step from NewWindow to give the driver a chance
// to setup its internal state for a window before events start being
// delivered.
func Show(hwnd syscall.Handle) {
	SendMessage(hwnd, msgShow, 0, 0)
}

func Release(hwnd syscall.Handle) {
	// TODO(andlabs): check for errors from this?
	// TODO(andlabs): remove unsafe
	_DestroyWindow(hwnd)
	// TODO(andlabs): what happens if we're still painting?
}

func sendFocus(hwnd syscall.Handle, uMsg uint32, wParam, lParam uintptr) (lResult uintptr) {
	switch uMsg {
	case _WM_SETFOCUS:
		LifecycleEvent(hwnd, lifecycle.StageFocused)
	case _WM_KILLFOCUS:
		LifecycleEvent(hwnd, lifecycle.StageVisible)
	default:
		panic(fmt.Sprintf("unexpected focus message: %d", uMsg))
	}
	return _DefWindowProc(hwnd, uMsg, wParam, lParam)
}

func sendShow(hwnd syscall.Handle, uMsg uint32, wParam, lParam uintptr) (lResult uintptr) {
	LifecycleEvent(hwnd, lifecycle.StageVisible)
	_ShowWindow(hwnd, _SW_SHOWDEFAULT)
	sendSize(hwnd)
	return 0
}

func sendSizeEvent(hwnd syscall.Handle, uMsg uint32, wParam, lParam uintptr) (lResult uintptr) {
	wp := (*_WINDOWPOS)(unsafe.Pointer(lParam))
	if wp.Flags&_SWP_NOSIZE != 0 {
		return 0
	}
	sendSize(hwnd)
	return 0
}

func sendSize(hwnd syscall.Handle) {
	var r _RECT
	if err := _GetClientRect(hwnd, &r); err != nil {
		panic(err) // TODO(andlabs)
	}

	width := int(r.Right - r.Left)
	height := int(r.Bottom - r.Top)

	// TODO(andlabs): don't assume that PixelsPerPt == 1
	SizeEvent(hwnd, size.Event{
		WidthPx:     width,
		HeightPx:    height,
		WidthPt:     geom.Pt(width),
		HeightPt:    geom.Pt(height),
		PixelsPerPt: 1,
	})
}

func sendClose(hwnd syscall.Handle, uMsg uint32, wParam, lParam uintptr) (lResult uintptr) {
	LifecycleEvent(hwnd, lifecycle.StageDead)
	return 0
}

func sendMouseEvent(hwnd syscall.Handle, uMsg uint32, wParam, lParam uintptr) (lResult uintptr) {
	e := mouse.Event{
		X:         float32(_GET_X_LPARAM(lParam)),
		Y:         float32(_GET_Y_LPARAM(lParam)),
		Modifiers: keyModifiers(),
	}

	switch uMsg {
	case _WM_MOUSEMOVE, _WM_MOUSEWHEEL:
		e.Direction = mouse.DirNone
	case _WM_LBUTTONDOWN, _WM_MBUTTONDOWN, _WM_RBUTTONDOWN:
		e.Direction = mouse.DirPress
	case _WM_LBUTTONUP, _WM_MBUTTONUP, _WM_RBUTTONUP:
		e.Direction = mouse.DirRelease
	default:
		panic("sendMouseEvent() called on non-mouse message")
	}

	switch uMsg {
	case _WM_MOUSEMOVE:
		// No-op.
	case _WM_LBUTTONDOWN, _WM_LBUTTONUP:
		e.Button = mouse.ButtonLeft
	case _WM_MBUTTONDOWN, _WM_MBUTTONUP:
		e.Button = mouse.ButtonMiddle
	case _WM_RBUTTONDOWN, _WM_RBUTTONUP:
		e.Button = mouse.ButtonRight
	case _WM_MOUSEWHEEL:
		delta := _GET_WHEEL_DELTA_WPARAM(wParam) / _WHEEL_DELTA
		switch {
		case delta > 0:
			e.Button = mouse.ButtonWheelUp
		case delta < 0:
			e.Button = mouse.ButtonWheelDown
			delta = -delta
		default:
			return
		}
		for delta > 0 {
			MouseEvent(hwnd, e)
			delta--
		}
		return
	}

	MouseEvent(hwnd, e)

	return 0
}

// Precondition: this is called in immediate response to the message that triggered the event (so not after w.Send).
func keyModifiers() (m key.Modifiers) {
	down := func(x int32) bool {
		// GetKeyState gets the key state at the time of the message, so this is what we want.
		return _GetKeyState(x)&0x80 != 0
	}

	if down(_VK_CONTROL) {
		m |= key.ModControl
	}
	if down(_VK_MENU) {
		m |= key.ModAlt
	}
	if down(_VK_SHIFT) {
		m |= key.ModShift
	}
	if down(_VK_LWIN) || down(_VK_RWIN) {
		m |= key.ModMeta
	}
	return m
}

var (
	MouseEvent     func(hwnd syscall.Handle, e mouse.Event)
	PaintEvent     func(hwnd syscall.Handle, e paint.Event)
	SizeEvent      func(hwnd syscall.Handle, e size.Event)
	KeyEvent       func(hwnd syscall.Handle, e key.Event)
	LifecycleEvent func(hwnd syscall.Handle, e lifecycle.Stage)

	// TODO: use the golang.org/x/exp/shiny/driver/internal/lifecycler package
	// instead of or together with the LifecycleEvent callback?
)

func sendPaint(hwnd syscall.Handle, uMsg uint32, wParam, lParam uintptr) (lResult uintptr) {
	PaintEvent(hwnd, paint.Event{})
	return _DefWindowProc(hwnd, uMsg, wParam, lParam)
}

var screenMsgs = map[uint32]func(hwnd syscall.Handle, uMsg uint32, wParam, lParam uintptr) (lResult uintptr){}

func AddScreenMsg(fn func(hwnd syscall.Handle, uMsg uint32, wParam, lParam uintptr)) uint32 {
	uMsg := currentUserWM.next()
	screenMsgs[uMsg] = func(hwnd syscall.Handle, uMsg uint32, wParam, lParam uintptr) uintptr {
		fn(hwnd, uMsg, wParam, lParam)
		return 0
	}
	return uMsg
}

func screenWindowWndProc(hwnd syscall.Handle, uMsg uint32, wParam uintptr, lParam uintptr) (lResult uintptr) {
	switch uMsg {
	case msgCreateWindow:
		p := (*newWindowParams)(unsafe.Pointer(lParam))
		p.w, p.err = newWindow(p.opts)
	case msgMainCallback:
		go func() {
			mainCallback()
			SendScreenMessage(msgQuit, 0, 0)
		}()
	case msgQuit:
		_PostQuitMessage(0)
	}
	fn := screenMsgs[uMsg]
	if fn != nil {
		return fn(hwnd, uMsg, wParam, lParam)
	}
	return _DefWindowProc(hwnd, uMsg, wParam, lParam)
}

func SendScreenMessage(uMsg uint32, wParam uintptr, lParam uintptr) (lResult uintptr) {
	return SendMessage(screenHWND, uMsg, wParam, lParam)
}

var windowMsgs = map[uint32]func(hwnd syscall.Handle, uMsg uint32, wParam, lParam uintptr) (lResult uintptr){
	_WM_SETFOCUS:         sendFocus,
	_WM_KILLFOCUS:        sendFocus,
	_WM_PAINT:            sendPaint,
	msgShow:              sendShow,
	_WM_WINDOWPOSCHANGED: sendSizeEvent,
	_WM_CLOSE:            sendClose,

	_WM_LBUTTONDOWN: sendMouseEvent,
	_WM_LBUTTONUP:   sendMouseEvent,
	_WM_MBUTTONDOWN: sendMouseEvent,
	_WM_MBUTTONUP:   sendMouseEvent,
	_WM_RBUTTONDOWN: sendMouseEvent,
	_WM_RBUTTONUP:   sendMouseEvent,
	_WM_MOUSEMOVE:   sendMouseEvent,
	_WM_MOUSEWHEEL:  sendMouseEvent,

	_WM_KEYDOWN: sendKeyEvent,
	_WM_KEYUP:   sendKeyEvent,
	// TODO case _WM_SYSKEYDOWN, _WM_SYSKEYUP:
}

func AddWindowMsg(fn func(hwnd syscall.Handle, uMsg uint32, wParam, lParam uintptr)) uint32 {
	uMsg := currentUserWM.next()
	windowMsgs[uMsg] = func(hwnd syscall.Handle, uMsg uint32, wParam, lParam uintptr) uintptr {
		fn(hwnd, uMsg, wParam, lParam)
		return 0
	}
	return uMsg
}

func windowWndProc(hwnd syscall.Handle, uMsg uint32, wParam uintptr, lParam uintptr) (lResult uintptr) {
	fn := windowMsgs[uMsg]
	if fn != nil {
		return fn(hwnd, uMsg, wParam, lParam)
	}
	return _DefWindowProc(hwnd, uMsg, wParam, lParam)
}

type newWindowParams struct {
	opts *screen.NewWindowOptions
	w    syscall.Handle
	err  error
}

func NewWindow(opts *screen.NewWindowOptions) (syscall.Handle, error) {
	var p newWindowParams
	p.opts = opts
	SendScreenMessage(msgCreateWindow, 0, uintptr(unsafe.Pointer(&p)))
	return p.w, p.err
}

const windowClass = "shiny_Window"

func initWindowClass() (err error) {
	wcname, err := syscall.UTF16PtrFromString(windowClass)
	if err != nil {
		return err
	}
	_, err = _RegisterClass(&_WNDCLASS{
		LpszClassName: wcname,
		LpfnWndProc:   syscall.NewCallback(windowWndProc),
		HIcon:         hDefaultIcon,
		HCursor:       hDefaultCursor,
		HInstance:     hThisInstance,
		// TODO(andlabs): change this to something else? NULL? the hollow brush?
		HbrBackground: syscall.Handle(_COLOR_BTNFACE + 1),
	})
	return err
}

func initScreenWindow() (err error) {
	const screenWindowClass = "shiny_ScreenWindow"
	swc, err := syscall.UTF16PtrFromString(screenWindowClass)
	if err != nil {
		return err
	}
	emptyString, err := syscall.UTF16PtrFromString("")
	if err != nil {
		return err
	}
	wc := _WNDCLASS{
		LpszClassName: swc,
		LpfnWndProc:   syscall.NewCallback(screenWindowWndProc),
		HIcon:         hDefaultIcon,
		HCursor:       hDefaultCursor,
		HInstance:     hThisInstance,
		HbrBackground: syscall.Handle(_COLOR_BTNFACE + 1),
	}
	_, err = _RegisterClass(&wc)
	if err != nil {
		return err
	}
	screenHWND, err = _CreateWindowEx(0,
		swc, emptyString,
		_WS_OVERLAPPEDWINDOW,
		_CW_USEDEFAULT, _CW_USEDEFAULT,
		_CW_USEDEFAULT, _CW_USEDEFAULT,
		_HWND_MESSAGE, 0, hThisInstance, 0)
	if err != nil {
		return err
	}
	return nil
}

var (
	hDefaultIcon   syscall.Handle
	hDefaultCursor syscall.Handle
	hThisInstance  syscall.Handle
)

func initCommon() (err error) {
	hDefaultIcon, err = _LoadIcon(0, _IDI_APPLICATION)
	if err != nil {
		return err
	}
	hDefaultCursor, err = _LoadCursor(0, _IDC_ARROW)
	if err != nil {
		return err
	}
	// TODO(andlabs) hThisInstance
	return nil
}

var mainCallback func()

func Main(f func()) (retErr error) {
	// It does not matter which OS thread we are on.
	// All that matters is that we confine all UI operations
	// to the thread that created the respective window.
	runtime.LockOSThread()

	if err := initCommon(); err != nil {
		return err
	}

	if err := initScreenWindow(); err != nil {
		return err
	}
	defer func() {
		// TODO(andlabs): log an error if this fails?
		_DestroyWindow(screenHWND)
		// TODO(andlabs): unregister window class
	}()

	if err := initWindowClass(); err != nil {
		return err
	}

	// Prime the pump.
	mainCallback = f
	_PostMessage(screenHWND, msgMainCallback, 0, 0)

	// Main message pump.
	var m _MSG
	for {
		done, err := _GetMessage(&m, 0, 0, 0)
		if err != nil {
			return fmt.Errorf("win32 GetMessage failed: %v", err)
		}
		if done == 0 { // WM_QUIT
			break
		}
		_TranslateMessage(&m)
		_DispatchMessage(&m)
	}

	return nil
}
