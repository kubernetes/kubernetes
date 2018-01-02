// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package screen provides interfaces for portable two-dimensional graphics and
// input events.
//
// Screens are not created directly. Instead, driver packages provide access to
// the screen through a Main function that is designed to be called by the
// program's main function. The golang.org/x/exp/shiny/driver package provides
// the default driver for the system, such as the X11 driver for desktop Linux,
// but other drivers, such as the OpenGL driver, can be explicitly invoked by
// calling that driver's Main function. To use the default driver:
//
//	package main
//
//	import (
//		"golang.org/x/exp/shiny/driver"
//		"golang.org/x/exp/shiny/screen"
//		"golang.org/x/mobile/event/lifecycle"
//	)
//
//	func main() {
//		driver.Main(func(s screen.Screen) {
//			w, err := s.NewWindow(nil)
//			if err != nil {
//				handleError(err)
//				return
//			}
//			defer w.Release()
//
//			for {
//				switch e := w.NextEvent().(type) {
//				case lifecycle.Event:
//					if e.To == lifecycle.StageDead {
//						return
//					}
//					etc
//				case etc:
//					etc
//				}
//			}
//		})
//	}
//
// Complete examples can be found in the shiny/example directory.
//
// Each driver package provides Screen, Buffer, Texture and Window
// implementations that work together. Such types are interface types because
// this package is driver-independent, but those interfaces aren't expected to
// be implemented outside of drivers. For example, a driver's Window
// implementation will generally work only with that driver's Buffer
// implementation, and will not work with an arbitrary type that happens to
// implement the Buffer methods.
package screen // import "golang.org/x/exp/shiny/screen"

import (
	"image"
	"image/color"
	"image/draw"

	"golang.org/x/image/math/f64"
)

// TODO: specify image format (Alpha or Gray, not just RGBA) for NewBuffer
// and/or NewTexture?

// Screen creates Buffers, Textures and Windows.
type Screen interface {
	// NewBuffer returns a new Buffer for this screen.
	NewBuffer(size image.Point) (Buffer, error)

	// NewTexture returns a new Texture for this screen.
	NewTexture(size image.Point) (Texture, error)

	// NewWindow returns a new Window for this screen.
	//
	// A nil opts is valid and means to use the default option values.
	NewWindow(opts *NewWindowOptions) (Window, error)
}

// TODO: rename Buffer to Image, to be less confusing with a Window's back and
// front buffers.

// Buffer is an in-memory pixel buffer. Its pixels can be modified by any Go
// code that takes an *image.RGBA, such as the standard library's image/draw
// package. A Buffer is essentially an *image.RGBA, but not all *image.RGBA
// values (including those returned by image.NewRGBA) are valid Buffers, as a
// driver may assume that the memory backing a Buffer's pixels are specially
// allocated.
//
// To see a Buffer's contents on a screen, upload it to a Texture (and then
// draw the Texture on a Window) or upload it directly to a Window.
//
// When specifying a sub-Buffer via Upload, a Buffer's top-left pixel is always
// (0, 0) in its own coordinate space.
type Buffer interface {
	// Release releases the Buffer's resources, after all pending uploads and
	// draws resolve.
	//
	// The behavior of the Buffer after Release, whether calling its methods or
	// passing it as an argument, is undefined.
	Release()

	// Size returns the size of the Buffer's image.
	Size() image.Point

	// Bounds returns the bounds of the Buffer's image. It is equal to
	// image.Rectangle{Max: b.Size()}.
	Bounds() image.Rectangle

	// RGBA returns the pixel buffer as an *image.RGBA.
	//
	// Its contents should not be accessed while the Buffer is uploading.
	//
	// The contents of the returned *image.RGBA's Pix field (of type []byte)
	// can be modified at other times, but that Pix slice itself (i.e. its
	// underlying pointer, length and capacity) should not be modified at any
	// time.
	//
	// The following is valid:
	//	m := buffer.RGBA()
	//	if len(m.Pix) >= 4 {
	//		m.Pix[0] = 0xff
	//		m.Pix[1] = 0x00
	//		m.Pix[2] = 0x00
	//		m.Pix[3] = 0xff
	//	}
	// or, equivalently:
	//	m := buffer.RGBA()
	//	m.SetRGBA(m.Rect.Min.X, m.Rect.Min.Y, color.RGBA{0xff, 0x00, 0x00, 0xff})
	// and using the standard library's image/draw package is also valid:
	//	dst := buffer.RGBA()
	//	draw.Draw(dst, dst.Bounds(), etc)
	// but the following is invalid:
	//	m := buffer.RGBA()
	//	m.Pix = anotherByteSlice
	// and so is this:
	//	*buffer.RGBA() = anotherImageRGBA
	RGBA() *image.RGBA
}

// Texture is a pixel buffer, but not one that is directly accessible as a
// []byte. Conceptually, it could live on a GPU, in another process or even be
// across a network, instead of on a CPU in this process.
//
// Buffers can be uploaded to Textures, and Textures can be drawn on Windows.
//
// When specifying a sub-Texture via Draw, a Texture's top-left pixel is always
// (0, 0) in its own coordinate space.
type Texture interface {
	// Release releases the Texture's resources, after all pending uploads and
	// draws resolve.
	//
	// The behavior of the Texture after Release, whether calling its methods
	// or passing it as an argument, is undefined.
	Release()

	// Size returns the size of the Texture's image.
	Size() image.Point

	// Bounds returns the bounds of the Texture's image. It is equal to
	// image.Rectangle{Max: t.Size()}.
	Bounds() image.Rectangle

	Uploader

	// TODO: also implement Drawer? If so, merge the Uploader and Drawer
	// interfaces??
}

// EventDeque is an infinitely buffered double-ended queue of events.
type EventDeque interface {
	// Send adds an event to the end of the deque. They are returned by
	// NextEvent in FIFO order.
	Send(event interface{})

	// SendFirst adds an event to the start of the deque. They are returned by
	// NextEvent in LIFO order, and have priority over events sent via Send.
	SendFirst(event interface{})

	// NextEvent returns the next event in the deque. It blocks until such an
	// event has been sent.
	//
	// Typical event types include:
	//	- lifecycle.Event
	//	- size.Event
	//	- paint.Event
	//	- key.Event
	//	- mouse.Event
	//	- touch.Event
	// from the golang.org/x/mobile/event/... packages. Other packages may send
	// events, of those types above or of other types, via Send or SendFirst.
	NextEvent() interface{}

	// TODO: LatestLifecycleEvent? Is that still worth it if the
	// lifecycle.Event struct type loses its DrawContext field?

	// TODO: LatestSizeEvent?
}

// Window is a top-level, double-buffered GUI window.
type Window interface {
	// Release closes the window.
	//
	// The behavior of the Window after Release, whether calling its methods or
	// passing it as an argument, is undefined.
	Release()

	EventDeque

	Uploader

	Drawer

	// Publish flushes any pending Upload and Draw calls to the window, and
	// swaps the back buffer to the front.
	Publish() PublishResult
}

// PublishResult is the result of an Window.Publish call.
type PublishResult struct {
	// BackBufferPreserved is whether the contents of the back buffer was
	// preserved. If false, the contents are undefined.
	BackBufferPreserved bool
}

// NewWindowOptions are optional arguments to NewWindow.
type NewWindowOptions struct {
	// Width and Height specify the dimensions of the new window. If Width
	// or Height are zero, a driver-dependent default will be used for each
	// zero value dimension.
	Width, Height int

	// TODO: fullscreen, title, icon, cursorHidden?
}

// Uploader is something you can upload a Buffer to.
type Uploader interface {
	// Upload uploads the sub-Buffer defined by src and sr to the destination
	// (the method receiver), such that sr.Min in src-space aligns with dp in
	// dst-space. The destination's contents are overwritten; the draw operator
	// is implicitly draw.Src.
	//
	// It is valid to upload a Buffer while another upload of the same Buffer
	// is in progress, but a Buffer's image.RGBA pixel contents should not be
	// accessed while it is uploading. A Buffer is re-usable, in that its pixel
	// contents can be further modified, once all outstanding calls to Upload
	// have returned.
	//
	// When uploading to a Window, there will not be any visible effect until
	// Publish is called.
	Upload(dp image.Point, src Buffer, sr image.Rectangle)

	// Fill fills that part of the destination (the method receiver) defined by
	// dr with the given color.
	//
	// When filling a Window, there will not be any visible effect until
	// Publish is called.
	Fill(dr image.Rectangle, src color.Color, op draw.Op)
}

// TODO: have a Downloader interface? Not every graphical app needs to be
// interactive or involve a window. You could use the GPU for hardware-
// accelerated image manipulation: upload a buffer, do some texture ops, then
// download the result.

// Drawer is something you can draw Textures on.
//
// Draw is the most general purpose of this interface's methods. It supports
// arbitrary affine transformations, such as translations, scales and
// rotations.
//
// Copy and Scale are more specific versions of Draw. The affected dst pixels
// are an axis-aligned rectangle, quantized to the pixel grid. Copy copies
// pixels in a 1:1 manner, Scale is more general. They have simpler parameters
// than Draw, using ints instead of float64s.
//
// When drawing on a Window, there will not be any visible effect until Publish
// is called.
type Drawer interface {
	// Draw draws the sub-Texture defined by src and sr to the destination (the
	// method receiver). src2dst defines how to transform src coordinates to
	// dst coordinates. For example, if src2dst is the matrix
	//
	// m00 m01 m02
	// m10 m11 m12
	//
	// then the src-space point (sx, sy) maps to the dst-space point
	// (m00*sx + m01*sy + m02, m10*sx + m11*sy + m12).
	Draw(src2dst f64.Aff3, src Texture, sr image.Rectangle, op draw.Op, opts *DrawOptions)

	// Copy copies the sub-Texture defined by src and sr to the destination
	// (the method receiver), such that sr.Min in src-space aligns with dp in
	// dst-space.
	Copy(dp image.Point, src Texture, sr image.Rectangle, op draw.Op, opts *DrawOptions)

	// Scale scales the sub-Texture defined by src and sr to the destination
	// (the method receiver), such that sr in src-space is mapped to dr in
	// dst-space.
	Scale(dr image.Rectangle, src Texture, sr image.Rectangle, op draw.Op, opts *DrawOptions)
}

// These draw.Op constants are provided so that users of this package don't
// have to explicitly import "image/draw".
const (
	Over = draw.Over
	Src  = draw.Src
)

// DrawOptions are optional arguments to Draw.
type DrawOptions struct {
	// TODO: transparency in [0x0000, 0xffff]?
	// TODO: scaler (nearest neighbor vs linear)?
}
