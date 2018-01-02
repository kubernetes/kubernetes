// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build example
//
// Use "go test -tags=example" to run this test.

package main

import "testing"

func TestIJXYCorners(t *testing.T) {
	var d Dims
	d.Init(19, 100)
	xSize := (d.dim - 1) * d.squareWidth
	ySize := (d.dim - 1) * d.squareHeight
	ij := IJ{1, 1}
	xy := ij.XY(&d)
	if xy.x != d.xInset || xy.y != d.yInset+ySize {
		t.Errorf("%d got %d", ij, xy)
	}
	ij = IJ{1, d.dim}
	xy = ij.XY(&d)
	if xy.x != d.xInset || xy.y != d.yInset {
		t.Errorf("%d got %d", ij, xy)
	}
	ij = IJ{d.dim, 1}
	xy = ij.XY(&d)
	if xy.x != d.xInset+xSize || xy.y != d.yInset+ySize {
		t.Errorf("%d got %d", ij, xy)
	}
	ij = IJ{d.dim, d.dim}
	xy = ij.XY(&d)
	if xy.x != d.xInset+xSize || xy.y != d.yInset {
		t.Errorf("%d got %d", ij, xy)
	}
}

func TestIJXYCenterRoundTrip(t *testing.T) {
	var d Dims
	d.Init(19, 100)
	for i := 1; i <= d.dim; i++ {
		for j := 1; j <= d.dim; j++ {
			ij := IJ{i, j}
			xy := ij.XYCenter(&d)
			ij2, ok := xy.IJ(&d)
			if !ok {
				t.Error("failed to round trip")
			}
			if ij2 != ij {
				t.Errorf("%d round trip got %d\n", ij, ij2)
			}
		}
	}
}

func TestIJString(t *testing.T) {
	var d Dims
	d.Init(19, 100)
	ij := IJ{1, 1}
	if ij.String() != "A1" {
		t.Errorf("%#v prints as %q", ij, ij.String())
	}
	ij = IJ{1, 19}
	if ij.String() != "A19" {
		t.Errorf("%#v prints as %q", ij, ij.String())
	}
	ij = IJ{19, 1}
	if ij.String() != "S1" {
		t.Errorf("%#v prints as %q", ij, ij.String())
	}
	ij = IJ{19, 19}
	if ij.String() != "S19" {
		t.Errorf("%#v prints as %q", ij, ij.String())
	}
}
