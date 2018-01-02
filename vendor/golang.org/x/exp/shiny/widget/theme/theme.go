// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package theme provides widget themes.
package theme

import (
	"image"
	"image/color"

	"golang.org/x/exp/shiny/unit"
	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

// FontFaceOptions allows asking for font face variants, such as style (e.g.
// italic) or weight (e.g. bold).
//
// TODO: include font.Hinting and font.Stretch typed fields?
//
// TODO: include font size? If so, directly as "12pt" or indirectly as an enum
// (Heading1, Heading2, Body, etc)?
type FontFaceOptions struct {
	Style  font.Style
	Weight font.Weight
}

// FontFaceCatalog provides a theme's font faces.
//
// AcquireFontFace returns a font.Face. ReleaseFontFace should be called, with
// the same options, once a widget's measure, layout or paint is done with the
// font.Face returned.
//
// A FontFaceCatalog is safe for use by multiple goroutines simultaneously, but
// in general, a font.Face is not safe for concurrent use, as its methods may
// re-use implementation-specific caches and mask image buffers.
type FontFaceCatalog interface {
	AcquireFontFace(FontFaceOptions) font.Face
	ReleaseFontFace(FontFaceOptions, font.Face)

	// TODO: add a "Metrics(FontFaceOptions) font.Metrics" method?
}

// Palette provides a theme's color palette.
//
// The colors are expressed as *image.Uniform values so that they can be easily
// passed as the src argument to image/draw functions.
type Palette struct {
	// Light, Neutral and Dark are three color tones used to fill in widgets
	// such as buttons, menu bars and panels.
	Light   *image.Uniform
	Neutral *image.Uniform
	Dark    *image.Uniform

	// Accent is the color used to accentuate selections or suggestions.
	Accent *image.Uniform

	// Foreground is the color used for text, dividers and icons.
	Foreground *image.Uniform

	// Background is the color used behind large blocks of text. Short,
	// non-editable label text will typically be on the Neutral color.
	Background *image.Uniform
}

// DefaultDPI is the fallback value of a theme's DPI, if the underlying context
// does not provide a DPI value.
const DefaultDPI = 72.0

var (
	// DefaultFontFaceCatalog is a catalog for a basic font face.
	DefaultFontFaceCatalog FontFaceCatalog = defaultFontFaceCatalog{}

	// DefaultPalette is the default theme's palette.
	DefaultPalette = Palette{
		Light:      &image.Uniform{C: color.RGBA{0xf5, 0xf5, 0xf5, 0xff}}, // Material Design "Grey 100".
		Neutral:    &image.Uniform{C: color.RGBA{0xee, 0xee, 0xee, 0xff}}, // Material Design "Grey 200".
		Dark:       &image.Uniform{C: color.RGBA{0xe0, 0xe0, 0xe0, 0xff}}, // Material Design "Grey 300".
		Accent:     &image.Uniform{C: color.RGBA{0x21, 0x96, 0xf3, 0xff}}, // Material Design "Blue 500".
		Foreground: &image.Uniform{C: color.RGBA{0x00, 0x00, 0x00, 0xff}}, // Material Design "Black".
		Background: &image.Uniform{C: color.RGBA{0xff, 0xff, 0xff, 0xff}}, // Material Design "White".
	}

	// Default uses the default DPI, FontFaceCatalog and Palette.
	//
	// The nil-valued pointer is a valid receiver for a Theme's methods.
	Default *Theme
)

// Note that a basicfont.Face is stateless and safe to use concurrently, so
// defaultFontFaceCatalog.ReleaseFontFace can be a no-op.

type defaultFontFaceCatalog struct{}

func (defaultFontFaceCatalog) AcquireFontFace(FontFaceOptions) font.Face  { return basicfont.Face7x13 }
func (defaultFontFaceCatalog) ReleaseFontFace(FontFaceOptions, font.Face) {}

// Theme is used for measuring, laying out and painting widgets. It consists of
// a screen DPI resolution, a set of font faces and colors.
type Theme struct {
	// DPI is the screen resolution, in dots (i.e. pixels) per inch.
	//
	// A zero value means to use the DefaultDPI.
	DPI float64

	// FontFaceCatalog provides a theme's font faces.
	//
	// A zero value means to use the DefaultFontFaceCatalog.
	FontFaceCatalog FontFaceCatalog

	// Palette provides a theme's color palette.
	//
	// A zero value means to use the DefaultPalette.
	Palette *Palette
}

// GetDPI returns the theme's DPI, or the default DPI if the field value is
// zero.
func (t *Theme) GetDPI() float64 {
	if t != nil && t.DPI != 0 {
		return t.DPI
	}
	return DefaultDPI
}

// GetFontFaceCatalog returns the theme's font face catalog, or the default
// catalog if the field value is zero.
func (t *Theme) GetFontFaceCatalog() FontFaceCatalog {
	if t != nil && t.FontFaceCatalog != nil {
		return t.FontFaceCatalog
	}
	return DefaultFontFaceCatalog
}

// GetPalette returns the theme's palette, or the default palette if the field
// value is zero.
func (t *Theme) GetPalette() *Palette {
	if t != nil && t.Palette != nil {
		return t.Palette
	}
	return &DefaultPalette
}

// AcquireFontFace calls the same method on the result of GetFontFaceCatalog.
func (t *Theme) AcquireFontFace(o FontFaceOptions) font.Face {
	return t.GetFontFaceCatalog().AcquireFontFace(o)
}

// ReleaseFontFace calls the same method on the result of GetFontFaceCatalog.
func (t *Theme) ReleaseFontFace(o FontFaceOptions, f font.Face) {
	t.GetFontFaceCatalog().ReleaseFontFace(o, f)
}

// Pixels implements the unit.Converter interface.
func (t *Theme) Pixels(v unit.Value) fixed.Int26_6 {
	c := t.Convert(v, unit.Px)
	return fixed.Int26_6(c.F * 64)
}

// Convert implements the unit.Converter interface.
func (t *Theme) Convert(v unit.Value, to unit.Unit) unit.Value {
	if v.U == to {
		return v
	}
	return unit.Value{
		F: v.F * t.pixelsPer(v.U) / t.pixelsPer(to),
		U: to,
	}
}

// pixelsPer returns the number of pixels in the unit u.
func (t *Theme) pixelsPer(u unit.Unit) float64 {
	switch u {
	case unit.Px:
		return 1
	case unit.Dp:
		return t.GetDPI() / unit.DensityIndependentPixelsPerInch
	case unit.Pt:
		return t.GetDPI() / unit.PointsPerInch
	case unit.Mm:
		return t.GetDPI() / unit.MillimetresPerInch
	case unit.In:
		return t.GetDPI()
	}

	f := t.AcquireFontFace(FontFaceOptions{})
	defer t.ReleaseFontFace(FontFaceOptions{}, f)

	// The 64 is because Height is in 26.6 fixed-point units.
	h := float64(f.Metrics().Height) / 64
	switch u {
	case unit.Em:
		return h
	case unit.Ex:
		return h / 2
	case unit.Ch:
		if advance, ok := f.GlyphAdvance('0'); ok {
			return float64(advance) / 64
		}
		return h / 2
	}
	return 1
}
