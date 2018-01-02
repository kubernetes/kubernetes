// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x11driver

import (
	"fmt"
	"image"
	"log"
	"sync"

	"github.com/BurntSushi/xgb"
	"github.com/BurntSushi/xgb/render"
	"github.com/BurntSushi/xgb/shm"
	"github.com/BurntSushi/xgb/xproto"

	"golang.org/x/exp/shiny/driver/internal/x11key"
	"golang.org/x/exp/shiny/screen"
	"golang.org/x/mobile/event/key"
	"golang.org/x/mobile/event/mouse"
)

// TODO: check that xgb is safe to use concurrently from multiple goroutines.
// For example, its Conn.WaitForEvent concept is a method, not a channel, so
// it's not obvious how to interrupt it to service a NewWindow request.

type screenImpl struct {
	xc      *xgb.Conn
	xsi     *xproto.ScreenInfo
	keysyms x11key.KeysymTable

	atomWMDeleteWindow xproto.Atom
	atomWMProtocols    xproto.Atom
	atomWMTakeFocus    xproto.Atom

	pixelsPerPt  float32
	pictformat24 render.Pictformat
	pictformat32 render.Pictformat

	// window32 and its related X11 resources is an unmapped window so that we
	// have a depth-32 window to create depth-32 pixmaps from, i.e. pixmaps
	// with an alpha channel. The root window isn't guaranteed to be depth-32.
	gcontext32 xproto.Gcontext
	window32   xproto.Window

	mu              sync.Mutex
	buffers         map[shm.Seg]*bufferImpl
	uploads         map[uint16]chan struct{}
	windows         map[xproto.Window]*windowImpl
	nPendingUploads int
	completionKeys  []uint16
}

func newScreenImpl(xc *xgb.Conn) (*screenImpl, error) {
	s := &screenImpl{
		xc:      xc,
		xsi:     xproto.Setup(xc).DefaultScreen(xc),
		buffers: map[shm.Seg]*bufferImpl{},
		uploads: map[uint16]chan struct{}{},
		windows: map[xproto.Window]*windowImpl{},
	}
	if err := s.initAtoms(); err != nil {
		return nil, err
	}
	if err := s.initKeyboardMapping(); err != nil {
		return nil, err
	}
	const (
		mmPerInch = 25.4
		ptPerInch = 72
	)
	pixelsPerMM := float32(s.xsi.WidthInPixels) / float32(s.xsi.WidthInMillimeters)
	s.pixelsPerPt = pixelsPerMM * mmPerInch / ptPerInch
	if err := s.initPictformats(); err != nil {
		return nil, err
	}
	if err := s.initWindow32(); err != nil {
		return nil, err
	}
	go s.run()
	return s, nil
}

func (s *screenImpl) run() {
	for {
		ev, err := s.xc.WaitForEvent()
		if err != nil {
			log.Printf("x11driver: xproto.WaitForEvent: %v", err)
			continue
		}

		noWindowFound := false
		switch ev := ev.(type) {
		case xproto.DestroyNotifyEvent:
			s.mu.Lock()
			delete(s.windows, ev.Window)
			s.mu.Unlock()

		case shm.CompletionEvent:
			s.mu.Lock()
			s.completionKeys = append(s.completionKeys, ev.Sequence)
			s.handleCompletions()
			s.mu.Unlock()

		case xproto.ClientMessageEvent:
			if ev.Type != s.atomWMProtocols || ev.Format != 32 {
				break
			}
			switch xproto.Atom(ev.Data.Data32[0]) {
			case s.atomWMDeleteWindow:
				if w := s.findWindow(ev.Window); w != nil {
					w.lifecycler.SetDead(true)
					w.lifecycler.SendEvent(w)
				} else {
					noWindowFound = true
				}
			case s.atomWMTakeFocus:
				xproto.SetInputFocus(s.xc, xproto.InputFocusParent, ev.Window, xproto.Timestamp(ev.Data.Data32[1]))
			}

		case xproto.ConfigureNotifyEvent:
			if w := s.findWindow(ev.Window); w != nil {
				w.handleConfigureNotify(ev)
			} else {
				noWindowFound = true
			}

		case xproto.ExposeEvent:
			if w := s.findWindow(ev.Window); w != nil {
				// A non-zero Count means that there are more expose events
				// coming. For example, a non-rectangular exposure (e.g. from a
				// partially overlapped window) will result in multiple expose
				// events whose dirty rectangles combine to define the dirty
				// region. Go's paint events do not provide dirty regions, so
				// we only pass on the final X11 expose event.
				if ev.Count == 0 {
					w.handleExpose()
				}
			} else {
				noWindowFound = true
			}

		case xproto.FocusInEvent:
			if w := s.findWindow(ev.Event); w != nil {
				w.lifecycler.SetFocused(true)
				w.lifecycler.SendEvent(w)
			} else {
				noWindowFound = true
			}

		case xproto.FocusOutEvent:
			if w := s.findWindow(ev.Event); w != nil {
				w.lifecycler.SetFocused(false)
				w.lifecycler.SendEvent(w)
			} else {
				noWindowFound = true
			}

		case xproto.KeyPressEvent:
			if w := s.findWindow(ev.Event); w != nil {
				w.handleKey(ev.Detail, ev.State, key.DirPress)
			} else {
				noWindowFound = true
			}

		case xproto.KeyReleaseEvent:
			if w := s.findWindow(ev.Event); w != nil {
				w.handleKey(ev.Detail, ev.State, key.DirRelease)
			} else {
				noWindowFound = true
			}

		case xproto.ButtonPressEvent:
			if w := s.findWindow(ev.Event); w != nil {
				w.handleMouse(ev.EventX, ev.EventY, ev.Detail, ev.State, mouse.DirPress)
			} else {
				noWindowFound = true
			}

		case xproto.ButtonReleaseEvent:
			if w := s.findWindow(ev.Event); w != nil {
				w.handleMouse(ev.EventX, ev.EventY, ev.Detail, ev.State, mouse.DirRelease)
			} else {
				noWindowFound = true
			}

		case xproto.MotionNotifyEvent:
			if w := s.findWindow(ev.Event); w != nil {
				w.handleMouse(ev.EventX, ev.EventY, 0, ev.State, mouse.DirNone)
			} else {
				noWindowFound = true
			}
		}

		if noWindowFound {
			log.Printf("x11driver: no window found for event %T", ev)
		}
	}
}

// TODO: is findBuffer and the s.buffers field unused? Delete?

func (s *screenImpl) findBuffer(key shm.Seg) *bufferImpl {
	s.mu.Lock()
	b := s.buffers[key]
	s.mu.Unlock()
	return b
}

func (s *screenImpl) findWindow(key xproto.Window) *windowImpl {
	s.mu.Lock()
	w := s.windows[key]
	s.mu.Unlock()
	return w
}

// handleCompletions must only be called while holding s.mu.
func (s *screenImpl) handleCompletions() {
	if s.nPendingUploads != 0 {
		return
	}
	for _, ck := range s.completionKeys {
		completion, ok := s.uploads[ck]
		if !ok {
			log.Printf("x11driver: no matching upload for a SHM completion event")
			continue
		}
		delete(s.uploads, ck)
		close(completion)
	}
	s.completionKeys = s.completionKeys[:0]
}

const (
	maxShmSide = 0x00007fff // 32,767 pixels.
	maxShmSize = 0x10000000 // 268,435,456 bytes.
)

func (s *screenImpl) NewBuffer(size image.Point) (retBuf screen.Buffer, retErr error) {
	// TODO: detect if the X11 server or connection cannot support SHM pixmaps,
	// and fall back to regular pixmaps.

	w, h := int64(size.X), int64(size.Y)
	if w < 0 || maxShmSide < w || h < 0 || maxShmSide < h || maxShmSize < 4*w*h {
		return nil, fmt.Errorf("x11driver: invalid buffer size %v", size)
	}

	b := &bufferImpl{
		s: s,
		rgba: image.RGBA{
			Stride: 4 * size.X,
			Rect:   image.Rectangle{Max: size},
		},
		size: size,
	}

	if size.X == 0 || size.Y == 0 {
		// No-op, but we can't take the else path because the minimum shmget
		// size is 1.
	} else {
		xs, err := shm.NewSegId(s.xc)
		if err != nil {
			return nil, fmt.Errorf("x11driver: shm.NewSegId: %v", err)
		}

		bufLen := 4 * size.X * size.Y
		shmid, addr, err := shmOpen(bufLen)
		if err != nil {
			return nil, fmt.Errorf("x11driver: shmOpen: %v", err)
		}
		defer func() {
			if retErr != nil {
				shmClose(addr)
			}
		}()
		a := (*[maxShmSize]byte)(addr)
		b.buf = (*a)[:bufLen:bufLen]
		b.rgba.Pix = b.buf
		b.addr = addr

		// readOnly is whether the shared memory is read-only from the X11 server's
		// point of view. We need false to use SHM pixmaps.
		const readOnly = false
		shm.Attach(s.xc, xs, uint32(shmid), readOnly)
		b.xs = xs
	}

	s.mu.Lock()
	s.buffers[b.xs] = b
	s.mu.Unlock()

	return b, nil
}

func (s *screenImpl) NewTexture(size image.Point) (screen.Texture, error) {
	w, h := int64(size.X), int64(size.Y)
	if w < 0 || maxShmSide < w || h < 0 || maxShmSide < h || maxShmSize < 4*w*h {
		return nil, fmt.Errorf("x11driver: invalid texture size %v", size)
	}
	if w == 0 || h == 0 {
		return &textureImpl{
			s:    s,
			size: size,
		}, nil
	}

	xm, err := xproto.NewPixmapId(s.xc)
	if err != nil {
		return nil, fmt.Errorf("x11driver: xproto.NewPixmapId failed: %v", err)
	}
	xp, err := render.NewPictureId(s.xc)
	if err != nil {
		return nil, fmt.Errorf("x11driver: xproto.NewPictureId failed: %v", err)
	}
	opaqueM, err := xproto.NewPixmapId(s.xc)
	if err != nil {
		return nil, fmt.Errorf("x11driver: xproto.NewPixmapId failed: %v", err)
	}
	opaqueP, err := render.NewPictureId(s.xc)
	if err != nil {
		return nil, fmt.Errorf("x11driver: xproto.NewPictureId failed: %v", err)
	}
	xproto.CreatePixmap(s.xc, textureDepth, xm, xproto.Drawable(s.window32), uint16(w), uint16(h))
	render.CreatePicture(s.xc, xp, xproto.Drawable(xm), s.pictformat32, render.CpRepeat, []uint32{render.RepeatPad})
	render.SetPictureFilter(s.xc, xp, uint16(len("bilinear")), "bilinear", nil)
	// The X11 server doesn't zero-initialize the pixmap. We do it ourselves.
	render.FillRectangles(s.xc, render.PictOpSrc, xp, render.Color{}, []xproto.Rectangle{{
		Width:  uint16(w),
		Height: uint16(h),
	}})

	return &textureImpl{
		s:       s,
		size:    size,
		xm:      xm,
		xp:      xp,
		opaqueM: opaqueM,
		opaqueP: opaqueP,
	}, nil
}

func (s *screenImpl) NewWindow(opts *screen.NewWindowOptions) (screen.Window, error) {
	width, height := 1024, 768
	if opts != nil {
		if opts.Width > 0 {
			width = opts.Width
		}
		if opts.Height > 0 {
			height = opts.Height
		}
	}

	xw, err := xproto.NewWindowId(s.xc)
	if err != nil {
		return nil, fmt.Errorf("x11driver: xproto.NewWindowId failed: %v", err)
	}
	xg, err := xproto.NewGcontextId(s.xc)
	if err != nil {
		return nil, fmt.Errorf("x11driver: xproto.NewGcontextId failed: %v", err)
	}
	xp, err := render.NewPictureId(s.xc)
	if err != nil {
		return nil, fmt.Errorf("x11driver: render.NewPictureId failed: %v", err)
	}
	pictformat := render.Pictformat(0)
	switch s.xsi.RootDepth {
	default:
		return nil, fmt.Errorf("x11driver: unsupported root depth %d", s.xsi.RootDepth)
	case 24:
		pictformat = s.pictformat24
	case 32:
		pictformat = s.pictformat32
	}

	w := &windowImpl{
		s:       s,
		xw:      xw,
		xg:      xg,
		xp:      xp,
		xevents: make(chan xgb.Event),
	}

	s.mu.Lock()
	s.windows[xw] = w
	s.mu.Unlock()

	w.lifecycler.SendEvent(w)

	xproto.CreateWindow(s.xc, s.xsi.RootDepth, xw, s.xsi.Root,
		0, 0, uint16(width), uint16(height), 0,
		xproto.WindowClassInputOutput, s.xsi.RootVisual,
		xproto.CwEventMask,
		[]uint32{0 |
			xproto.EventMaskKeyPress |
			xproto.EventMaskKeyRelease |
			xproto.EventMaskButtonPress |
			xproto.EventMaskButtonRelease |
			xproto.EventMaskPointerMotion |
			xproto.EventMaskExposure |
			xproto.EventMaskStructureNotify |
			xproto.EventMaskFocusChange,
		},
	)
	s.setProperty(xw, s.atomWMProtocols, s.atomWMDeleteWindow, s.atomWMTakeFocus)
	xproto.CreateGC(s.xc, xg, xproto.Drawable(xw), 0, nil)
	render.CreatePicture(s.xc, xp, xproto.Drawable(xw), pictformat, 0, nil)
	xproto.MapWindow(s.xc, xw)

	return w, nil
}

func (s *screenImpl) initAtoms() (err error) {
	s.atomWMDeleteWindow, err = s.internAtom("WM_DELETE_WINDOW")
	if err != nil {
		return err
	}
	s.atomWMProtocols, err = s.internAtom("WM_PROTOCOLS")
	if err != nil {
		return err
	}
	s.atomWMTakeFocus, err = s.internAtom("WM_TAKE_FOCUS")
	if err != nil {
		return err
	}
	return nil
}

func (s *screenImpl) internAtom(name string) (xproto.Atom, error) {
	r, err := xproto.InternAtom(s.xc, false, uint16(len(name)), name).Reply()
	if err != nil {
		return 0, fmt.Errorf("x11driver: xproto.InternAtom failed: %v", err)
	}
	if r == nil {
		return 0, fmt.Errorf("x11driver: xproto.InternAtom failed")
	}
	return r.Atom, nil
}

func (s *screenImpl) initKeyboardMapping() error {
	const keyLo, keyHi = 8, 255
	km, err := xproto.GetKeyboardMapping(s.xc, keyLo, keyHi-keyLo+1).Reply()
	if err != nil {
		return err
	}
	n := int(km.KeysymsPerKeycode)
	if n < 2 {
		return fmt.Errorf("x11driver: too few keysyms per keycode: %d", n)
	}
	for i := keyLo; i <= keyHi; i++ {
		s.keysyms[i][0] = uint32(km.Keysyms[(i-keyLo)*n+0])
		s.keysyms[i][1] = uint32(km.Keysyms[(i-keyLo)*n+1])
	}
	return nil
}

func (s *screenImpl) initPictformats() error {
	pformats, err := render.QueryPictFormats(s.xc).Reply()
	if err != nil {
		return fmt.Errorf("x11driver: render.QueryPictFormats failed: %v", err)
	}
	s.pictformat24, err = findPictformat(pformats.Formats, 24)
	if err != nil {
		return err
	}
	s.pictformat32, err = findPictformat(pformats.Formats, 32)
	if err != nil {
		return err
	}
	return nil
}

func findPictformat(fs []render.Pictforminfo, depth byte) (render.Pictformat, error) {
	// This presumes little-endian BGRA.
	want := render.Directformat{
		RedShift:   16,
		RedMask:    0xff,
		GreenShift: 8,
		GreenMask:  0xff,
		BlueShift:  0,
		BlueMask:   0xff,
		AlphaShift: 24,
		AlphaMask:  0xff,
	}
	if depth == 24 {
		want.AlphaShift = 0
		want.AlphaMask = 0x00
	}
	for _, f := range fs {
		if f.Type == render.PictTypeDirect && f.Depth == depth && f.Direct == want {
			return f.Id, nil
		}
	}
	return 0, fmt.Errorf("x11driver: no matching Pictformat for depth %d", depth)
}

func (s *screenImpl) initWindow32() error {
	visualid, err := findVisual(s.xsi, 32)
	if err != nil {
		return err
	}
	colormap, err := xproto.NewColormapId(s.xc)
	if err != nil {
		return fmt.Errorf("x11driver: xproto.NewColormapId failed: %v", err)
	}
	if err := xproto.CreateColormapChecked(
		s.xc, xproto.ColormapAllocNone, colormap, s.xsi.Root, visualid).Check(); err != nil {
		return fmt.Errorf("x11driver: xproto.CreateColormap failed: %v", err)
	}
	s.window32, err = xproto.NewWindowId(s.xc)
	if err != nil {
		return fmt.Errorf("x11driver: xproto.NewWindowId failed: %v", err)
	}
	s.gcontext32, err = xproto.NewGcontextId(s.xc)
	if err != nil {
		return fmt.Errorf("x11driver: xproto.NewGcontextId failed: %v", err)
	}
	const depth = 32
	xproto.CreateWindow(s.xc, depth, s.window32, s.xsi.Root,
		0, 0, 1, 1, 0,
		xproto.WindowClassInputOutput, visualid,
		// The CwBorderPixel attribute seems necessary for depth == 32. See
		// http://stackoverflow.com/questions/3645632/how-to-create-a-window-with-a-bit-depth-of-32
		xproto.CwBorderPixel|xproto.CwColormap,
		[]uint32{0, uint32(colormap)},
	)
	xproto.CreateGC(s.xc, s.gcontext32, xproto.Drawable(s.window32), 0, nil)
	return nil
}

func findVisual(xsi *xproto.ScreenInfo, depth byte) (xproto.Visualid, error) {
	for _, d := range xsi.AllowedDepths {
		if d.Depth != depth {
			continue
		}
		for _, v := range d.Visuals {
			if v.RedMask == 0xff0000 && v.GreenMask == 0xff00 && v.BlueMask == 0xff {
				return v.VisualId, nil
			}
		}
	}
	return 0, fmt.Errorf("x11driver: no matching Visualid")
}

func (s *screenImpl) setProperty(xw xproto.Window, prop xproto.Atom, values ...xproto.Atom) {
	b := make([]byte, len(values)*4)
	for i, v := range values {
		b[4*i+0] = uint8(v >> 0)
		b[4*i+1] = uint8(v >> 8)
		b[4*i+2] = uint8(v >> 16)
		b[4*i+3] = uint8(v >> 24)
	}
	xproto.ChangeProperty(s.xc, xproto.PropModeReplace, xw, prop, xproto.AtomAtom, 32, uint32(len(values)), b)
}
