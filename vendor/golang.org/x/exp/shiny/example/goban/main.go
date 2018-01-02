// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build example
//
// This build tag means that "go install golang.org/x/exp/shiny/..." doesn't
// install this example program. Use "go run main.go board.go xy.go" to run it
// or "go install -tags=example" to install it.

// Goban is a simple example of a graphics program using shiny.
// It implements a Go board that two people can use to play the game.
// TODO: Improve the main function.
// TODO: Provide more functionality.
package main

import (
	"flag"
	"image"
	"log"
	"math/rand"
	"time"

	"golang.org/x/exp/shiny/driver"
	"golang.org/x/exp/shiny/screen"
	"golang.org/x/mobile/event/key"
	"golang.org/x/mobile/event/lifecycle"
	"golang.org/x/mobile/event/mouse"
	"golang.org/x/mobile/event/paint"
	"golang.org/x/mobile/event/size"
)

var scale = flag.Int("scale", 35, "`percent` to scale images (TODO: a poor design)")

func main() {
	flag.Parse()

	rand.Seed(int64(time.Now().Nanosecond()))
	board := NewBoard(9, *scale)

	driver.Main(func(s screen.Screen) {
		w, err := s.NewWindow(nil)
		if err != nil {
			log.Fatal(err)
		}
		defer w.Release()

		var b screen.Buffer
		defer func() {
			if b != nil {
				b.Release()
			}
		}()

		for {
			switch e := w.NextEvent().(type) {
			case lifecycle.Event:
				if e.To == lifecycle.StageDead {
					return
				}

			case key.Event:
				if e.Code == key.CodeEscape {
					return
				}

			case mouse.Event:
				if e.Direction != mouse.DirRelease {
					break
				}

				// Re-map control-click to middle-click, etc, for computers with one-button mice.
				if e.Modifiers&key.ModControl != 0 {
					e.Button = mouse.ButtonMiddle
				} else if e.Modifiers&key.ModAlt != 0 {
					e.Button = mouse.ButtonRight
				} else if e.Modifiers&key.ModMeta != 0 {
					e.Button = mouse.ButtonMiddle
				}

				if board.click(b.RGBA(), int(e.X), int(e.Y), int(e.Button)) {
					w.Send(paint.Event{})
				}

			case paint.Event:
				w.Upload(image.Point{}, b, b.Bounds())
				w.Publish()

			case size.Event:
				// TODO: Set board size.
				if b != nil {
					b.Release()
				}
				b, err = s.NewBuffer(e.Size())
				if err != nil {
					log.Fatal(err)
				}
				render(b.RGBA(), board)

			case error:
				log.Print(e)
			}
		}
	})
}

func render(m *image.RGBA, board *Board) {
	board.Draw(m)
}
