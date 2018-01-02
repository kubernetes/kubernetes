// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x11driver

import (
	"image"
	"image/color"
	"image/draw"
	"math"
	"sync"

	"github.com/BurntSushi/xgb/render"
	"github.com/BurntSushi/xgb/xproto"

	"golang.org/x/exp/shiny/screen"
	"golang.org/x/image/math/f64"
)

const textureDepth = 32

type textureImpl struct {
	s *screenImpl

	size image.Point
	xm   xproto.Pixmap
	xp   render.Picture

	// renderMu is a mutex that enforces the atomicity of methods like
	// Window.Draw that are conceptually one operation but are implemented by
	// multiple X11/Render calls. X11/Render is a stateful API, so interleaving
	// X11/Render calls from separate higher-level operations causes
	// inconsistencies.
	//
	// It also protects the opaqueXxx fields, which hold a lazily created,
	// fully opaque mask picture.
	renderMu      sync.Mutex
	opaqueM       xproto.Pixmap
	opaqueP       render.Picture
	opaqueCreated bool

	releasedMu sync.Mutex
	released   bool
}

func (t *textureImpl) degenerate() bool        { return t.size.X == 0 || t.size.Y == 0 }
func (t *textureImpl) Size() image.Point       { return t.size }
func (t *textureImpl) Bounds() image.Rectangle { return image.Rectangle{Max: t.size} }

func (t *textureImpl) Release() {
	t.releasedMu.Lock()
	released := t.released
	t.released = true
	t.releasedMu.Unlock()

	if released || t.degenerate() {
		return
	}
	if t.opaqueCreated {
		render.FreePicture(t.s.xc, t.opaqueP)
		xproto.FreePixmap(t.s.xc, t.opaqueM)
	}
	render.FreePicture(t.s.xc, t.xp)
	xproto.FreePixmap(t.s.xc, t.xm)
}

func (t *textureImpl) Upload(dp image.Point, src screen.Buffer, sr image.Rectangle) {
	if t.degenerate() {
		return
	}
	src.(*bufferImpl).upload(xproto.Drawable(t.xm), t.s.gcontext32, textureDepth, dp, sr)
}

func (t *textureImpl) Fill(dr image.Rectangle, src color.Color, op draw.Op) {
	if t.degenerate() {
		return
	}
	fill(t.s.xc, t.xp, dr, src, op)
}

// f64ToFixed converts from float64 to X11/Render's 16.16 fixed point.
func f64ToFixed(x float64) render.Fixed {
	return render.Fixed(x * 65536)
}

func inv(x *f64.Aff3) f64.Aff3 {
	invDet := 1 / (x[0]*x[4] - x[1]*x[3])
	return f64.Aff3{
		+x[4] * invDet,
		-x[1] * invDet,
		(x[1]*x[5] - x[2]*x[4]) * invDet,
		-x[3] * invDet,
		+x[0] * invDet,
		(x[2]*x[3] - x[0]*x[5]) * invDet,
	}
}

func (t *textureImpl) draw(xp render.Picture, src2dst *f64.Aff3, sr image.Rectangle, op draw.Op, opts *screen.DrawOptions) {
	sr = sr.Intersect(t.Bounds())
	if sr.Empty() {
		return
	}

	t.renderMu.Lock()
	defer t.renderMu.Unlock()

	// For simple copies and scales, the inverse matrix is trivial to compute,
	// and we do not need the "Src becomes OutReverse plus Over" dance (see
	// below). Thus, draw can be one render.SetPictureTransform call and then
	// one render.Composite call, regardless of whether or not op is Src.
	if src2dst[1] == 0 && src2dst[3] == 0 {
		dstXMin := float64(sr.Min.X)*src2dst[0] + src2dst[2]
		dstXMax := float64(sr.Max.X)*src2dst[0] + src2dst[2]
		if dstXMin > dstXMax {
			// TODO: check if this (and below) works when src2dst[0] < 0.
			dstXMin, dstXMax = dstXMax, dstXMin
		}
		dXMin := int(math.Floor(dstXMin))
		dXMax := int(math.Ceil(dstXMax))

		dstYMin := float64(sr.Min.Y)*src2dst[4] + src2dst[5]
		dstYMax := float64(sr.Max.Y)*src2dst[4] + src2dst[5]
		if dstYMin > dstYMax {
			// TODO: check if this (and below) works when src2dst[4] < 0.
			dstYMin, dstYMax = dstYMax, dstYMin
		}
		dYMin := int(math.Floor(dstYMin))
		dYMax := int(math.Ceil(dstYMax))

		render.SetPictureTransform(t.s.xc, t.xp, render.Transform{
			f64ToFixed(1 / src2dst[0]), 0, 0,
			0, f64ToFixed(1 / src2dst[4]), 0,
			0, 0, 1 << 16,
		})
		render.Composite(t.s.xc, renderOp(op), t.xp, 0, xp,
			int16(sr.Min.X), int16(sr.Min.Y), // SrcX, SrcY,
			0, 0, // MaskX, MaskY,
			int16(dXMin), int16(dYMin), // DstX, DstY,
			uint16(dXMax-dXMin), uint16(dYMax-dYMin), // Width, Height,
		)
		return
	}

	// The X11/Render transform matrix maps from destination pixels to source
	// pixels, so we invert src2dst.
	dst2src := inv(src2dst)
	render.SetPictureTransform(t.s.xc, t.xp, render.Transform{
		f64ToFixed(dst2src[0]), f64ToFixed(dst2src[1]), render.Fixed(sr.Min.X << 16),
		f64ToFixed(dst2src[3]), f64ToFixed(dst2src[4]), render.Fixed(sr.Min.Y << 16),
		0, 0, 1 << 16,
	})

	minX := float64(sr.Min.X)
	maxX := float64(sr.Max.X)
	minY := float64(sr.Min.Y)
	maxY := float64(sr.Max.Y)
	points := [4]render.Pointfix{{
		f64ToFixed(src2dst[0]*minX + src2dst[1]*minY + src2dst[2]),
		f64ToFixed(src2dst[3]*minX + src2dst[4]*minY + src2dst[5]),
	}, {
		f64ToFixed(src2dst[0]*maxX + src2dst[1]*minY + src2dst[2]),
		f64ToFixed(src2dst[3]*maxX + src2dst[4]*minY + src2dst[5]),
	}, {
		f64ToFixed(src2dst[0]*maxX + src2dst[1]*maxY + src2dst[2]),
		f64ToFixed(src2dst[3]*maxX + src2dst[4]*maxY + src2dst[5]),
	}, {
		f64ToFixed(src2dst[0]*minX + src2dst[1]*maxY + src2dst[2]),
		f64ToFixed(src2dst[3]*minX + src2dst[4]*maxY + src2dst[5]),
	}}

	if op == draw.Src {
		// Lazily create the opaque mask picture.
		if !t.opaqueCreated {
			t.opaqueCreated = true
			xproto.CreatePixmap(t.s.xc, textureDepth, t.opaqueM, xproto.Drawable(t.s.window32), 1, 1)
			render.CreatePicture(t.s.xc, t.opaqueP, xproto.Drawable(t.opaqueM), t.s.pictformat32, 0, nil)
			render.FillRectangles(t.s.xc, render.PictOpSrc, t.opaqueP, render.Color{
				Red:   0xffff,
				Green: 0xffff,
				Blue:  0xffff,
				Alpha: 0xffff,
			}, []xproto.Rectangle{{
				Width:  1,
				Height: 1,
			}})
		}

		// render.TriFan visits every dst-space pixel in the axis-aligned
		// bounding box (AABB) containing the transformation of the sr
		// rectangle in src-space to a quad in dst-space.
		//
		// render.TriFan is like render.Composite, except that the AABB is
		// defined implicitly by the transformed triangle vertices instead of
		// being passed explicitly as arguments. It implies the minimal AABB.
		//
		// In any case, for arbitrary src2dst affine transformations, which
		// include rotations, this means that a naive render.TriFan call will
		// affect those pixels inside the AABB but outside the quad. For the
		// draw.Src operator, this means that pixels in that AABB can be
		// incorrectly set to zero.
		//
		// Instead, we implement the draw.Src operator as two render.TriFan
		// calls. The first one (using the PictOpOutReverse operator and a
		// fully opaque source) clears the dst-space quad but leaves pixels
		// outside that quad (but inside the AABB) untouched. The second one
		// (using the PictOpOver operator and the texture t as source) fills in
		// the quad and again does not touch the pixels outside.
		//
		// What X11/Render calls PictOpOutReverse is also known as dst-out. See
		// http://www.w3.org/TR/SVGCompositing/examples/compop-porterduff-examples.png
		// for a visualization.
		invW := 1 / float64(sr.Dx())
		invH := 1 / float64(sr.Dy())
		render.SetPictureTransform(t.s.xc, t.opaqueP, render.Transform{
			f64ToFixed(invW * dst2src[0]), f64ToFixed(invW * dst2src[1]), 0,
			f64ToFixed(invH * dst2src[3]), f64ToFixed(invH * dst2src[4]), 0,
			0, 0, 1 << 16,
		})
		render.TriFan(t.s.xc, render.PictOpOutReverse, t.opaqueP, xp, 0, 0, 0, points[:])
	}
	render.TriFan(t.s.xc, render.PictOpOver, t.xp, xp, 0, 0, 0, points[:])
}

func renderOp(op draw.Op) byte {
	if op == draw.Src {
		return render.PictOpSrc
	}
	return render.PictOpOver
}
