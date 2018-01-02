// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build example
//
// This build tag means that "go install golang.org/x/exp/shiny/..." doesn't
// install this example program. Use "go run main.go" to run it or "go install
// -tags=example" to install it.

// Imageview is a basic image viewer. Supported image formats include BMP, GIF,
// JPEG, PNG, TIFF and WEBP.
package main

import (
	"fmt"
	"image"
	"log"
	"os"

	"golang.org/x/exp/shiny/driver"
	"golang.org/x/exp/shiny/screen"
	"golang.org/x/exp/shiny/widget"

	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"

	_ "golang.org/x/image/bmp"
	_ "golang.org/x/image/tiff"
	_ "golang.org/x/image/webp"
)

// TODO: scrolling, such as when images are larger than the window.

func decode(filename string) (image.Image, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	m, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("could not decode %s: %v", filename, err)
	}
	return m, nil
}

func main() {
	log.SetFlags(0)
	driver.Main(func(s screen.Screen) {
		if len(os.Args) < 2 {
			log.Fatal("no image file specified")
		}
		// TODO: view multiple images.
		src, err := decode(os.Args[1])
		if err != nil {
			log.Fatal(err)
		}
		m := widget.NewImage(src, src.Bounds())
		if err := widget.RunWindow(s, m, nil); err != nil {
			log.Fatal(err)
		}
	})
}
