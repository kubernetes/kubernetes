// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build example
//
// This build tag means that "go install golang.org/x/exp/shiny/..." doesn't
// install this example program. Use "go run main.go" to run it or "go install
// -tags=example" to install it.

// Tile demonstrates tiling a screen with textures.
package main

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"log"
	"sync"

	"golang.org/x/exp/shiny/driver"
	"golang.org/x/exp/shiny/screen"
	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
	"golang.org/x/mobile/event/key"
	"golang.org/x/mobile/event/lifecycle"
	"golang.org/x/mobile/event/mouse"
	"golang.org/x/mobile/event/paint"
	"golang.org/x/mobile/event/size"
)

func main() {
	driver.Main(func(s screen.Screen) {
		w, err := s.NewWindow(nil)
		if err != nil {
			log.Fatal(err)
		}
		defer w.Release()

		var (
			pool = &tilePool{
				screen:   s,
				drawRGBA: drawRGBA,
				m:        map[image.Point]*tilePoolEntry{},
			}
			dragging     bool
			paintPending bool
			drag         image.Point
			origin       image.Point
			sz           size.Event
		)
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
				p := image.Point{X: int(e.X), Y: int(e.Y)}
				if e.Button == mouse.ButtonLeft && e.Direction != mouse.DirNone {
					dragging = e.Direction == mouse.DirPress
					drag = p
				}
				if !dragging {
					break
				}
				origin = origin.Sub(p.Sub(drag))
				drag = p
				if origin.X < 0 {
					origin.X = 0
				}
				if origin.Y < 0 {
					origin.Y = 0
				}
				if !paintPending {
					paintPending = true
					w.Send(paint.Event{})
				}

			case paint.Event:
				generation++
				var wg sync.WaitGroup
				for y := -(origin.Y & 0xff); y < sz.HeightPx; y += 256 {
					for x := -(origin.X & 0xff); x < sz.WidthPx; x += 256 {
						wg.Add(1)
						go drawTile(&wg, w, pool, origin, x, y)
					}
				}
				wg.Wait()
				w.Publish()
				paintPending = false
				pool.releaseUnused()

			case size.Event:
				sz = e

			case error:
				log.Print(e)
			}
		}
	})
}

func drawTile(wg *sync.WaitGroup, w screen.Window, pool *tilePool, origin image.Point, x, y int) {
	defer wg.Done()
	tp := image.Point{
		(x + origin.X) >> 8,
		(y + origin.Y) >> 8,
	}
	tex, err := pool.get(tp)
	if err != nil {
		log.Println(err)
		return
	}
	w.Copy(image.Point{x, y}, tex, tileBounds, screen.Src, nil)
}

func drawRGBA(m *image.RGBA, tp image.Point) {
	draw.Draw(m, m.Bounds(), image.White, image.Point{}, draw.Src)
	for _, p := range crossPoints {
		m.SetRGBA(p.X, p.Y, crossColor)
	}
	d := font.Drawer{
		Dst:  m,
		Src:  image.Black,
		Face: basicfont.Face7x13,
		Dot: fixed.Point26_6{
			Y: basicfont.Face7x13.Metrics().Ascent,
		},
	}
	d.DrawString(fmt.Sprint(tp))
}

var (
	crossColor  = color.RGBA{0x7f, 0x00, 0x00, 0xff}
	crossPoints = []image.Point{
		{0x00, 0xfe},
		{0x00, 0xff},
		{0xfe, 0x00},
		{0xff, 0x00},
		{0x00, 0x00},
		{0x01, 0x00},
		{0x02, 0x00},
		{0x00, 0x01},
		{0x00, 0x02},

		{0x80, 0x7f},
		{0x7f, 0x80},
		{0x80, 0x80},
		{0x81, 0x80},
		{0x80, 0x81},

		{0x80, 0x00},

		{0x00, 0x80},
	}

	generation int

	tileSize   = image.Point{256, 256}
	tileBounds = image.Rectangle{Max: tileSize}
)

type tilePoolEntry struct {
	tex screen.Texture
	gen int
}

type tilePool struct {
	screen   screen.Screen
	drawRGBA func(*image.RGBA, image.Point)

	mu sync.Mutex
	m  map[image.Point]*tilePoolEntry
}

func (p *tilePool) get(tp image.Point) (screen.Texture, error) {
	p.mu.Lock()
	v, ok := p.m[tp]
	if v != nil {
		v.gen = generation
	}
	p.mu.Unlock()

	if ok {
		return v.tex, nil
	}
	tex, err := p.screen.NewTexture(tileSize)
	if err != nil {
		return nil, err
	}
	buf, err := p.screen.NewBuffer(tileSize)
	if err != nil {
		tex.Release()
		return nil, err
	}
	p.drawRGBA(buf.RGBA(), tp)
	tex.Upload(image.Point{}, buf, tileBounds)
	buf.Release()

	p.mu.Lock()
	p.m[tp] = &tilePoolEntry{
		tex: tex,
		gen: generation,
	}
	n := len(p.m)
	p.mu.Unlock()

	fmt.Printf("%4d textures; created  %v\n", n, tp)
	return tex, nil
}

func (p *tilePool) releaseUnused() {
	p.mu.Lock()
	defer p.mu.Unlock()

	for tp, v := range p.m {
		if v.gen == generation {
			continue
		}
		v.tex.Release()
		delete(p.m, tp)
		fmt.Printf("%4d textures; released %v\n", len(p.m), tp)
	}
}
