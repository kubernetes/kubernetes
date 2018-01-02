// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package widget

import (
	"image"
	"image/color"
	"image/draw"

	"golang.org/x/exp/shiny/unit"
	"golang.org/x/exp/shiny/widget/node"
	"golang.org/x/exp/shiny/widget/theme"
)

// Uniform is a leaf widget that paints a uniform color, analogous to an
// image.Uniform.
type Uniform struct {
	node.LeafEmbed
	Uniform       image.Uniform
	NaturalWidth  unit.Value
	NaturalHeight unit.Value
}

// NewUniform returns a new Uniform widget of the given color and natural size.
// Its parent widget may lay it out at a different size than its natural size,
// such as expanding to fill a panel's width.
func NewUniform(c color.Color, naturalWidth, naturalHeight unit.Value) *Uniform {
	w := &Uniform{
		Uniform:       image.Uniform{c},
		NaturalWidth:  naturalWidth,
		NaturalHeight: naturalHeight,
	}
	w.Wrapper = w
	return w
}

func (w *Uniform) Measure(t *theme.Theme) {
	w.MeasuredSize.X = t.Pixels(w.NaturalWidth).Round()
	w.MeasuredSize.Y = t.Pixels(w.NaturalHeight).Round()
}

func (w *Uniform) Paint(t *theme.Theme, dst *image.RGBA, origin image.Point) {
	if w.Uniform.C == nil {
		return
	}
	draw.Draw(dst, w.Rect.Add(origin), &w.Uniform, image.Point{}, draw.Src)
}
