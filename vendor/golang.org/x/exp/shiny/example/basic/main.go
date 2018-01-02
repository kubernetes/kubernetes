// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build example
//
// This build tag means that "go install golang.org/x/exp/shiny/..." doesn't
// install this example program. Use "go run main.go" to run it or "go install
// -tags=example" to install it.

// Basic is a basic example of a graphical application.
package main

import (
	"fmt"
	"image"
	"image/color"
	"log"
	"math"

	"golang.org/x/exp/shiny/driver"
	"golang.org/x/exp/shiny/screen"
	"golang.org/x/image/math/f64"
	"golang.org/x/mobile/event/key"
	"golang.org/x/mobile/event/lifecycle"
	"golang.org/x/mobile/event/paint"
	"golang.org/x/mobile/event/size"
)

var (
	blue0    = color.RGBA{0x00, 0x00, 0x1f, 0xff}
	blue1    = color.RGBA{0x00, 0x00, 0x3f, 0xff}
	darkGray = color.RGBA{0x3f, 0x3f, 0x3f, 0xff}
	red      = color.RGBA{0x7f, 0x00, 0x00, 0x7f}

	cos30 = math.Cos(math.Pi / 6)
	sin30 = math.Sin(math.Pi / 6)
)

func main() {
	driver.Main(func(s screen.Screen) {
		w, err := s.NewWindow(nil)
		if err != nil {
			log.Fatal(err)
		}
		defer w.Release()

		winSize := image.Point{256, 256}
		b, err := s.NewBuffer(winSize)
		if err != nil {
			log.Fatal(err)
		}
		defer b.Release()
		drawGradient(b.RGBA())

		t, err := s.NewTexture(winSize)
		if err != nil {
			log.Fatal(err)
		}
		defer t.Release()
		t.Upload(image.Point{}, b, b.Bounds())

		var sz size.Event
		for {
			e := w.NextEvent()

			// This print message is to help programmers learn what events this
			// example program generates. A real program shouldn't print such
			// messages; they're not important to end users.
			format := "got %#v\n"
			if _, ok := e.(fmt.Stringer); ok {
				format = "got %v\n"
			}
			fmt.Printf(format, e)

			switch e := e.(type) {
			case lifecycle.Event:
				if e.To == lifecycle.StageDead {
					return
				}

			case key.Event:
				if e.Code == key.CodeEscape {
					return
				}

			case paint.Event:
				w.Fill(sz.Bounds(), blue0, screen.Src)
				w.Fill(sz.Bounds().Inset(10), blue1, screen.Src)
				w.Upload(image.Point{20, 0}, b, b.Bounds())
				w.Fill(image.Rect(50, 50, 350, 120), red, screen.Over)

				// By default, draw the entirety of the texture using the Over
				// operator. Uncomment one or both of the lines below to see
				// their different effects.
				op := screen.Over
				// op = screen.Src
				tRect := t.Bounds()
				// tRect = image.Rect(16, 0, 240, 100)

				// Draw the texture t twice, as a 1:1 copy and under the
				// transform src2dst.
				w.Copy(image.Point{150, 100}, t, tRect, op, nil)
				src2dst := f64.Aff3{
					+0.5 * cos30, -1.0 * sin30, 100,
					+0.5 * sin30, +1.0 * cos30, 200,
				}
				w.Draw(src2dst, t, tRect, op, nil)

				// Draw crosses at the transformed corners of tRect.
				for _, sx := range []int{tRect.Min.X, tRect.Max.X} {
					for _, sy := range []int{tRect.Min.Y, tRect.Max.Y} {
						dx := int(src2dst[0]*float64(sx) + src2dst[1]*float64(sy) + src2dst[2])
						dy := int(src2dst[3]*float64(sx) + src2dst[4]*float64(sy) + src2dst[5])
						w.Fill(image.Rect(dx-0, dy-1, dx+1, dy+2), darkGray, screen.Src)
						w.Fill(image.Rect(dx-1, dy-0, dx+2, dy+1), darkGray, screen.Src)
					}
				}

				w.Publish()

			case size.Event:
				sz = e

			case error:
				log.Print(e)
			}
		}
	})
}

func drawGradient(m *image.RGBA) {
	b := m.Bounds()
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			if x%64 == 0 || y%64 == 0 {
				m.SetRGBA(x, y, color.RGBA{0xff, 0xff, 0xff, 0xff})
			} else if x%64 == 63 || y%64 == 63 {
				m.SetRGBA(x, y, color.RGBA{0x00, 0x00, 0xff, 0xff})
			} else {
				m.SetRGBA(x, y, color.RGBA{uint8(x), uint8(y), 0x00, 0xff})
			}
		}
	}

	// Round off the corners.
	const radius = 64
	lox := b.Min.X + radius - 1
	loy := b.Min.Y + radius - 1
	hix := b.Max.X - radius
	hiy := b.Max.Y - radius
	for y := 0; y < radius; y++ {
		for x := 0; x < radius; x++ {
			if x*x+y*y <= radius*radius {
				continue
			}
			m.SetRGBA(lox-x, loy-y, color.RGBA{})
			m.SetRGBA(hix+x, loy-y, color.RGBA{})
			m.SetRGBA(lox-x, hiy+y, color.RGBA{})
			m.SetRGBA(hix+x, hiy+y, color.RGBA{})
		}
	}
}
