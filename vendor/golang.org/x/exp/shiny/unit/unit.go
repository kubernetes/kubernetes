// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package unit defines units of length such as inches or pixels.
//
// Functions like Inches and Pixels return a Value in the corresponding unit.
// For example:
//
//	v := unit.Inches(4.5)
//
// represents four and a half inches.
//
// Converting between pixels (px), physical units (dp, pt, in, mm) and
// font-face-relative measures (em, ex, ch) depends on the context, such as the
// screen's DPI resolution and the active font face. That context is
// represented by the Converter type.
//
// Conversions may be lossy. Converting 4.5 inches to pixels and back may
// result in something slightly different than 4.5. Similarly, converting 4
// inches and 0.5 inches to pixels and then adding the results won't
// necessarily equal the conversion of 4.5 inches to pixels.
//
// Note that what CSS (Cascading Style Sheets) calls "px" differs from what
// this package calls "px". For legacy reasons, the CSS semantics are that 1
// inch should roughly equal 96csspx regardless of the actual DPI resolution,
// as per https://developer.mozilla.org/en/docs/Web/CSS/length. This package's
// semantics are that 1px means exactly one physical pixel, always. This
// package represents 1csspx as 1.666666667dp, since there are 160 density
// independent pixels per inch, the same definition as Android.
package unit

import (
	"fmt"

	"golang.org/x/image/math/fixed"
)

const (
	DensityIndependentPixelsPerInch = 160
	MillimetresPerInch              = 25.4
	PointsPerInch                   = 72
)

// Converter converts values from one unit to another. Conversions may be
// lossy.
type Converter interface {
	// Convert converts v to the given unit.
	Convert(v Value, to Unit) Value

	// Pixels converts v to a 26.6 fixed-point number of physical pixels.
	Pixels(v Value) fixed.Int26_6
}

// Value is a number and a unit.
type Value struct {
	F float64
	U Unit
}

// String implements the fmt.Stringer interface.
func (v Value) String() string {
	return fmt.Sprintf("%f%s", v.F, names[v.U])
}

var names = [...]string{
	Px: "px",
	Dp: "dp",
	Pt: "pt",
	In: "in",
	Mm: "mm",
	Em: "em",
	Ex: "ex",
	Ch: "ch",
}

// Unit is a unit of length, such as inches or pixels.
type Unit uint8

const (
	// Px is a physical pixel, regardless of the DPI resolution.
	Px Unit = iota

	// Dp is 1 density independent pixel: 1/160th of an inch.
	Dp
	// Pt is 1 point: 1/72th of an inch.
	Pt
	// Mm is 1 millimetre: 1/25.4th of an inch.
	Mm
	// In is 1 inch.
	//
	// If the context does not specify a DPI resolution, the recommended
	// fallback value for conversion is 72 pixels per inch.
	In

	// Em is the height of the active font face, disregarding extra leading
	// such as from double-spaced lines of text.
	//
	// If the context does not specify an active font face, the recommended
	// fallback value for conversion is 12pt.
	Em
	// Ex is the x-height of the active font face.
	//
	// If the context does not specify an x-height, the recommended fallback
	// value for conversion is 0.5em.
	Ex
	// Ch is the character width of the numeral zero glyph '0' of the active
	// font face.
	//
	// If the context does not specify a '0' glyph, the recommended fallback
	// value for conversion is 0.5em.
	Ch
)

// Pixels returns the given number of Px as a Value.
func Pixels(f float64) Value { return Value{f, Px} }

// DIPs returns the given number of Dp as a Value.
func DIPs(f float64) Value { return Value{f, Dp} }

// Points returns the given number of Pt as a Value.
func Points(f float64) Value { return Value{f, Pt} }

// Millimetres returns the given number of Mm as a Value.
func Millimetres(f float64) Value { return Value{f, Mm} }

// Inches returns the given number of In as a Value.
func Inches(f float64) Value { return Value{f, In} }

// Ems returns the given number of Em as a Value.
func Ems(f float64) Value { return Value{f, Em} }

// Exs returns the given number of Ex as a Value.
func Exs(f float64) Value { return Value{f, Ex} }

// Chs returns the given number of Ch as a Value.
func Chs(f float64) Value { return Value{f, Ch} }
