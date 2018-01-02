// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// Custom image resizer. Saved for posterity.

package main

import (
	"flag"
	"fmt"
	"image"
	"image/jpeg"
	_ "image/png"
	"log"
	"os"

	"golang.org/x/image/draw"
)

var n = flag.Int("n", 0, "number of pixels to inset before scaling")

func main() {
	flag.Parse()
	fmt.Println(flag.Args())
	for _, s := range flag.Args() {
		resize(s)
	}
}

func resize(s string) {
	in, err := os.Open(s)
	if err != nil {
		log.Fatal(err)
	}
	defer in.Close()
	src, _, err := image.Decode(in)
	if err != nil {
		log.Fatal(err)
	}
	name := "../" + s
	out, err := os.Create(name)
	fmt.Println(name)
	if err != nil {
		log.Fatal(err)
	}
	defer out.Close()
	dst := image.NewGray(image.Rect(0, 0, 256, 256))
	draw.Draw(dst, dst.Bounds(), image.White, image.ZP, draw.Src)
	dr := image.Rect(*n, *n, 256-*n, 256-*n)
	draw.ApproxBiLinear.Scale(dst, dr, src, src.Bounds(), draw.Src, nil)
	err = jpeg.Encode(out, dst, nil)
	if err != nil {
		log.Fatal(err)
	}
}
