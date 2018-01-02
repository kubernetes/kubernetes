// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package widget

import (
	"image"
	"image/draw"

	"golang.org/x/exp/shiny/text"
	"golang.org/x/exp/shiny/unit"
	"golang.org/x/exp/shiny/widget/node"
	"golang.org/x/exp/shiny/widget/theme"
	"golang.org/x/image/font"
	"golang.org/x/image/math/fixed"
)

// Text is a leaf widget that holds a text label.
type Text struct {
	node.LeafEmbed

	// TODO: add NaturalWidth and NaturalHeight fields a la Uniform?
	//
	// Should they go in a Sizer shell widget instead?

	frame   text.Frame
	faceSet bool

	// TODO: scrolling, although should that be the responsibility of this
	// widget, the parent widget or something else?
}

// NewText returns a new Text widget.
func NewText(text string) *Text {
	w := &Text{}
	w.Wrapper = w
	if text != "" {
		c := w.frame.NewCaret()
		c.WriteString(text)
		c.Close()
	}
	return w
}

func (w *Text) Measure(t *theme.Theme) {
	// TODO: implement. Should the Measure method include a width hint?
	w.MeasuredSize = image.Point{}
}

func (w *Text) Layout(t *theme.Theme) {
	// TODO: can a theme change at runtime, or can it be set only once, at
	// start-up?
	if !w.faceSet {
		w.faceSet = true
		// TODO: when is face released? Should we just unconditionally call
		// SetFace for every Measure, Layout and Paint? How do we avoid
		// excessive re-calculation of soft returns when re-using the same
		// logical face (as in "Times New Roman 12pt") even if using different
		// physical font.Face values (as each Face may have its own caches)?
		face := t.AcquireFontFace(theme.FontFaceOptions{})
		w.frame.SetFace(face)
	}

	// TODO: should padding (and/or margin and border) be a universal concept
	// and part of the node.Embed type instead of having each widget implement
	// its own?
	padding := t.Pixels(unit.Ems(0.5)).Ceil()
	w.frame.SetMaxWidth(fixed.I(w.Rect.Dx() - 2*padding))
}

func (w *Text) Paint(t *theme.Theme, dst *image.RGBA, origin image.Point) {
	dst = dst.SubImage(w.Rect.Add(origin)).(*image.RGBA)
	if dst.Bounds().Empty() {
		return
	}

	face := t.AcquireFontFace(theme.FontFaceOptions{})
	defer t.ReleaseFontFace(theme.FontFaceOptions{}, face)
	m := face.Metrics()
	ascent := m.Ascent.Ceil()
	height := m.Height.Ceil()

	padding := t.Pixels(unit.Ems(0.5)).Ceil()

	draw.Draw(dst, dst.Bounds(), t.GetPalette().Background, image.Point{}, draw.Src)

	x0 := fixed.I(origin.X + w.Rect.Min.X + padding)
	d := font.Drawer{
		Dst:  dst,
		Src:  t.GetPalette().Foreground,
		Face: face,
		Dot: fixed.Point26_6{
			X: x0,
			Y: fixed.I(origin.Y + w.Rect.Min.Y + padding + ascent),
		},
	}
	f := &w.frame
	for p := f.FirstParagraph(); p != nil; p = p.Next(f) {
		// TODO: bail out early if the paragraph or line isn't visible. For
		// long (high) passages of text, this should save many DrawBytes calls.
		for l := p.FirstLine(f); l != nil; l = l.Next(f) {
			for b := l.FirstBox(f); b != nil; b = b.Next(f) {
				d.DrawBytes(b.TrimmedText(f))
				// TODO: adjust d.Dot.X for any ligatures?
			}
			d.Dot.X = x0
			d.Dot.Y += fixed.I(height)
		}
	}
}
