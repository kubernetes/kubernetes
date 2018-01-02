// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build example
//
// This build tag means that "go install golang.org/x/exp/shiny/..." doesn't
// install this example program. Use "go run main.go board.go xy.go" to run it
// or "go install -tags=example" to install it.

package main

import (
	"fmt"
	"image"
)

// Note that a "square" is not a drawn square on the board, but
// the rectangle (sic) centered on the game point.
type Dims struct {
	dim          int // Size of board, always square. Full size is 19.
	percent      int // Scale.
	xInset       int // From left edge to left edge of first square.
	yInset       int // From top edge to top edge of first square.
	stoneDiam    int // Diameter of stone in pixels.
	stoneRad2    int // Stone radius squared.
	squareWidth  int // Width of square in pixels.
	squareHeight int // Height of square in pixels.
	lineWidth    int // Width of lines.
}

// Initial values, in pixels. These numbers match the images that are loaded; they are resized.
const (
	// These numbers are tuned for the images in asset/.
	xInset0    = 100 // Distance from the edge of the board image to the left side of the first stone.
	yInset0    = 115 // Distance from the top of the board image to the top side of the first stone.
	stoneSize0 = 256 // Size of a stone on the board.
	stoneDiam0 = 225 // Diameter of the circular part of the stone within the square.
	// The square is a little smaller than the stone, for crowding.
	squareWidth0  = 215 // Width of a square on the board.
	squareHeight0 = 218 // Height of a square on the board.
	stoneRad2     = stoneDiam0 * stoneDiam0 / 4
)

func (d *Dims) Init(dim, percent int) {
	d.dim = dim
	d.Resize(percent)
}

func (d *Dims) Resize(percent int) {
	d.percent = percent
	d.xInset = xInset0 * percent / 100
	d.yInset = yInset0 * percent / 100
	d.stoneDiam = stoneDiam0 * percent / 100
	d.squareWidth = squareWidth0 * percent / 100
	d.squareHeight = squareHeight0 * percent / 100
	d.stoneRad2 = d.stoneDiam * d.stoneDiam / 4
	d.lineWidth = 4 * percent / 100
	if d.lineWidth < 2 {
		d.lineWidth = 2
	}
	if d.lineWidth > 2 {
		d.lineWidth = 4
	}
}

// An XY represents a pixel in the board image. Y is downwards.
type XY struct {
	x, y int
}

// An IJ represents a position on the board. It is 1-indexed and J is upwards.
type IJ struct {
	i, j int
}

func (ij IJ) String() string {
	return fmt.Sprintf("%c%d", 'A'+int(ij.i-1), ij.j)
}

// A go board is played on centers, not on squares, but we think of it internally
// as IJ squares (1, 1) through (19, 19), and draw the lines through the middle
// of those squares. That is how the arithmetic is done converting between
// IJ and XY.

// IJ converts a position in the board image to a Go position (IJ) on the board.
// The boolean is false if the point does not represent a Go position.
func (xy XY) IJ(d *Dims) (IJ, bool) {
	x, y := xy.x, xy.y
	x -= d.xInset
	y -= d.yInset
	i := x / d.squareWidth
	j := y / d.squareHeight
	// Now zero indexed. Make j goes up and switch to 1-indexed.
	j = d.dim - 1 - j
	i++
	j++
	if 1 <= i && i <= d.dim && 1 <= j && j <= d.dim {
		return IJ{i, j}, true
	}
	return IJ{}, false
}

// XY converts a Go position to the upper left corner of the square for that position
// on the board image.
func (ij IJ) XY(d *Dims) XY {
	// Change to 0-indexed.
	ij.i--
	ij.j--
	// j goes down.
	ij.j = (d.dim - 1) - ij.j
	return XY{d.xInset + ij.i*d.squareWidth, d.yInset + ij.j*d.squareHeight}
}

// IJtoXYCenter converts a Go position to the center of the square for that position
// on the board image.
func (ij IJ) XYCenter(d *Dims) XY {
	xy := ij.XY(d)
	xy.x += d.squareWidth / 2
	xy.y += d.squareHeight / 2
	return xy
}

// IJtoXYStone converts a Go position to the square holding the stone for that position
// on the board image.
func (ij IJ) XYStone(d *Dims) image.Rectangle {
	center := ij.XYCenter(d)
	min := image.Point{center.x - d.stoneDiam/2, center.y - d.stoneDiam/2}
	max := image.Point{center.x + d.stoneDiam/2, center.y + d.stoneDiam/2}
	return image.Rectangle{Min: min, Max: max}
}
