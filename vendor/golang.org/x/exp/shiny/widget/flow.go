// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package widget

import (
	"image"

	"golang.org/x/exp/shiny/widget/node"
	"golang.org/x/exp/shiny/widget/theme"
)

// TODO: padding, alignment.

// Flow is a container widget that lays out its children in sequence along an
// axis, either horizontally or vertically. The children's laid out size may
// differ from their natural size, along or across that axis, if a child's
// LayoutData is a FlowLayoutData.
type Flow struct {
	node.ContainerEmbed
	Axis Axis
}

// NewFlow returns a new Flow widget containing the given children.
func NewFlow(a Axis, children ...node.Node) *Flow {
	w := &Flow{
		Axis: a,
	}
	w.Wrapper = w
	for _, c := range children {
		w.Insert(c, nil)
	}
	return w
}

func (w *Flow) Measure(t *theme.Theme) {
	if w.Axis != AxisHorizontal && w.Axis != AxisVertical {
		w.ContainerEmbed.Measure(t)
		return
	}

	mSize := image.Point{}
	for c := w.FirstChild; c != nil; c = c.NextSibling {
		c.Wrapper.Measure(t)
		if w.Axis == AxisHorizontal {
			mSize.X += c.MeasuredSize.X
			if mSize.Y < c.MeasuredSize.Y {
				mSize.Y = c.MeasuredSize.Y
			}
		} else {
			mSize.Y += c.MeasuredSize.Y
			if mSize.X < c.MeasuredSize.X {
				mSize.X = c.MeasuredSize.X
			}
		}
	}
	w.MeasuredSize = mSize
}

func (w *Flow) Layout(t *theme.Theme) {
	if w.Axis != AxisHorizontal && w.Axis != AxisVertical {
		w.ContainerEmbed.Layout(t)
		return
	}

	eaExtra, eaWeight := 0, 0
	if w.Axis == AxisHorizontal {
		eaExtra = w.Rect.Dx()
	} else {
		eaExtra = w.Rect.Dy()
	}
	for c := w.FirstChild; c != nil; c = c.NextSibling {
		if d, ok := c.LayoutData.(FlowLayoutData); ok && d.ExpandAlongWeight > 0 {
			eaWeight += d.ExpandAlongWeight
		}
		if w.Axis == AxisHorizontal {
			eaExtra -= c.MeasuredSize.X
		} else {
			eaExtra -= c.MeasuredSize.Y
		}
	}
	if eaExtra < 0 {
		eaExtra = 0
	}

	p := image.Point{}
	for c := w.FirstChild; c != nil; c = c.NextSibling {
		q := p.Add(c.MeasuredSize)
		if d, ok := c.LayoutData.(FlowLayoutData); ok {
			if d.ExpandAlongWeight > 0 {
				delta := eaExtra * d.ExpandAlongWeight / eaWeight
				eaExtra -= delta
				eaWeight -= d.ExpandAlongWeight
				if w.Axis == AxisHorizontal {
					q.X += delta
				} else {
					q.Y += delta
				}
			}
			if d.ExpandAcross {
				if w.Axis == AxisHorizontal {
					q.Y = max(q.Y, w.Rect.Dy())
				} else {
					q.X = max(q.X, w.Rect.Dx())
				}
			}
		}
		c.Rect = image.Rectangle{
			Min: p,
			Max: q,
		}
		c.Wrapper.Layout(t)
		if w.Axis == AxisHorizontal {
			p.X = q.X
		} else {
			p.Y = q.Y
		}
	}
}

// FlowLayoutData is the node LayoutData type for a Flow's children.
type FlowLayoutData struct {
	// ExpandAlongWeight is the relative weight for distributing any excess
	// space along the Flow's axis. For example, if an AxisHorizontal Flow's
	// Rect width was 100 pixels greater than the sum of its children's natural
	// widths, and three children had non-zero FlowLayoutData.ExpandAlongWeight
	// values 6, 3 and 1, then those children's laid out widths would be larger
	// than their natural widths by 60, 30 and 10 pixels.
	ExpandAlongWeight int

	// ExpandAcross is whether the child's laid out size should expand to fill
	// the Flow's cross-axis. For example, if an AxisHorizontal Flow's Rect
	// height was 80 pixels, any child whose FlowLayoutData.ExpandAcross was
	// true would also be laid out with at least an 80 pixel height.
	ExpandAcross bool
}
