// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build example
//
// This build tag means that "go install golang.org/x/exp/shiny/..." doesn't
// install this example program. Use "go run main.go" to run it or "go install
// -tags=example" to install it.

// Gallery demonstrates the shiny/widget package.
package main

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"log"

	"golang.org/x/exp/shiny/driver"
	"golang.org/x/exp/shiny/screen"
	"golang.org/x/exp/shiny/widget"
	"golang.org/x/exp/shiny/widget/node"
	"golang.org/x/exp/shiny/widget/theme"
)

var red = image.NewUniform(color.RGBA{0xff, 0x00, 0x00, 0xff})

// custom is a custom widget.
type custom struct {
	node.LeafEmbed
}

func newCustom() *custom {
	w := &custom{}
	w.Wrapper = w
	return w
}

func (w *custom) OnInputEvent(e interface{}, origin image.Point) node.EventHandled {
	// TODO: do something more interesting.
	fmt.Printf("%T %v\n", e, e)
	return node.Handled
}

func (w *custom) Paint(t *theme.Theme, dst *image.RGBA, origin image.Point) {
	// TODO: do something more interesting.
	draw.Draw(dst, w.Rect.Add(origin), red, image.Point{}, draw.Src)
}

func main() {
	log.SetFlags(0)
	driver.Main(func(s screen.Screen) {
		// TODO: create a bunch of standard widgets: buttons, labels, etc.
		w := newCustom()
		if err := widget.RunWindow(s, w, nil); err != nil {
			log.Fatal(err)
		}
	})
}
