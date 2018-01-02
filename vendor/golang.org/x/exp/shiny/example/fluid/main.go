// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build example
//
// This build tag means that "go install golang.org/x/exp/shiny/..." doesn't
// install this example program. Use "go run main.go" to run it or "go install
// -tags=example" to install it.

// Fluid is a fluid dynamics simulator. It is based on Jos Stam, "Real-Time
// Fluid Dynamics for Games", Proceedings of the Game Developer Conference,
// March 2003. See
// http://www.dgp.toronto.edu/people/stam/reality/Research/pub.html
package main

import (
	"image"
	"image/color"
	"image/draw"
	"log"
	"sync"
	"time"

	"golang.org/x/exp/shiny/driver"
	"golang.org/x/exp/shiny/screen"
	"golang.org/x/mobile/event/lifecycle"
	"golang.org/x/mobile/event/mouse"
	"golang.org/x/mobile/event/paint"
	"golang.org/x/mobile/event/size"
)

const (
	N = 128 // The grid of cells has size NxN.

	tickDuration = time.Second / 60

	// These remaining numbers have magic values, determined by trial and error
	// to look good, rather than being derived from first principles.
	iterations = 20
	dt         = 0.1
	diff       = 0
	visc       = 0
	force      = 5
	source     = 20
	fade       = 0.89
)

func main() {
	driver.Main(func(s screen.Screen) {
		w, err := s.NewWindow(nil)
		if err != nil {
			log.Fatal(err)
		}
		buf, tex := screen.Buffer(nil), screen.Texture(nil)
		defer func() {
			if buf != nil {
				tex.Release()
				buf.Release()
			}
			w.Release()
		}()

		go simulate(w)

		var (
			buttonDown bool
			sz         size.Event
		)
		for {
			publish := false

			switch e := w.NextEvent().(type) {
			case lifecycle.Event:
				if e.To == lifecycle.StageDead {
					return
				}

				switch e.Crosses(lifecycle.StageVisible) {
				case lifecycle.CrossOn:
					pauseChan <- play
					var err error
					buf, err = s.NewBuffer(image.Point{N, N})
					if err != nil {
						log.Fatal(err)
					}
					tex, err = s.NewTexture(image.Point{N, N})
					if err != nil {
						log.Fatal(err)
					}
					tex.Fill(tex.Bounds(), color.White, draw.Src)

				case lifecycle.CrossOff:
					pauseChan <- pause
					tex.Release()
					tex = nil
					buf.Release()
					buf = nil
				}

			case mouse.Event:
				if e.Button == mouse.ButtonLeft {
					buttonDown = e.Direction == mouse.DirPress
				}
				if !buttonDown {
					break
				}
				z := sz.Size()
				x := int(e.X) * N / z.X
				y := int(e.Y) * N / z.Y
				if x < 0 || N <= x || y < 0 || N <= y {
					break
				}

				shared.mu.Lock()
				shared.mouseEvents = append(shared.mouseEvents, image.Point{x, y})
				shared.mu.Unlock()

			case paint.Event:
				publish = buf != nil

			case size.Event:
				sz = e

			case uploadEvent:
				shared.mu.Lock()
				if buf != nil {
					copy(buf.RGBA().Pix, shared.pix)
					publish = true
				}
				shared.uploadEventSent = false
				shared.mu.Unlock()

				if publish {
					tex.Upload(image.Point{}, buf, buf.Bounds())
				}

			case error:
				log.Print(e)
			}

			if publish {
				w.Scale(sz.Bounds(), tex, tex.Bounds(), draw.Src, nil)
				w.Publish()
			}
		}
	})
}

const (
	pause = false
	play  = true
)

// pauseChan lets the UI event goroutine pause and play the CPU-intensive
// simulation goroutine depending on whether the window is visible (e.g.
// minimized). 64 should be large enough, in typical use, so that the former
// doesn't ever block on the latter.
var pauseChan = make(chan bool, 64)

// uploadEvent signals that the shared pix slice should be uploaded to the
// screen.Texture via the screen.Buffer.
type uploadEvent struct{}

var shared = struct {
	mu              sync.Mutex
	uploadEventSent bool
	mouseEvents     []image.Point
	pix             []byte
}{
	pix: make([]byte, 4*N*N),
}

func simulate(q screen.EventQueue) {
	var (
		dens, densPrev array
		u, uPrev       array
		v, vPrev       array
		xPrev, yPrev   int
		havePrevLoc    bool
	)

	ticker := time.NewTicker(tickDuration)
	var tickerC <-chan time.Time
	for {
		select {
		case p := <-pauseChan:
			if p == pause {
				tickerC = nil
			} else {
				tickerC = ticker.C
			}
			continue
		case <-tickerC:
		}

		shared.mu.Lock()
		for _, p := range shared.mouseEvents {
			dens[p.X+1][p.Y] = source
			if havePrevLoc {
				u[p.X+1][p.Y+1] = force * float32(p.X-xPrev)
				v[p.X+1][p.Y+1] = force * float32(p.Y-yPrev)
			}
			xPrev, yPrev, havePrevLoc = p.X, p.Y, true
		}
		shared.mouseEvents = shared.mouseEvents[:0]
		shared.mu.Unlock()

		velStep(&u, &v, &uPrev, &vPrev)
		densStep(&dens, &densPrev, &u, &v)

		// This fade isn't part of Stam's GDC03 paper, but it looks nice.
		for i := range dens {
			for j := range dens[i] {
				dens[i][j] *= fade
			}
		}

		shared.mu.Lock()
		for y := 0; y < N; y++ {
			for x := 0; x < N; x++ {
				d := int32(dens[x+1][y+1] * 0xff)
				if d < 0 {
					d = 0
				} else if d > 0xff {
					d = 0xff
				}
				v := 255 - uint8(d)
				p := (N*y + x) * 4
				shared.pix[p+0] = v
				shared.pix[p+1] = v
				shared.pix[p+2] = v
				shared.pix[p+3] = 0xff
			}
		}
		uploadEventSent := shared.uploadEventSent
		shared.uploadEventSent = true
		shared.mu.Unlock()

		if !uploadEventSent {
			q.Send(uploadEvent{})
		}
	}
}

// All of the remaining code more or less comes from Stam's GDC03 paper.

type array [N + 2][N + 2]float32

func addSource(x, s *array) {
	for i := range x {
		for j := range x[i] {
			x[i][j] += dt * s[i][j]
		}
	}
}

func setBnd(b int, x *array) {
	switch b {
	case 0:
		for i := 1; i <= N; i++ {
			x[0+0][i] = +x[1][i]
			x[N+1][i] = +x[N][i]
			x[i][0+0] = +x[i][1]
			x[i][N+1] = +x[i][N]
		}
	case 1:
		for i := 1; i <= N; i++ {
			x[0+0][i] = -x[1][i]
			x[N+1][i] = -x[N][i]
			x[i][0+0] = +x[i][1]
			x[i][N+1] = +x[i][N]
		}
	case 2:
		for i := 1; i <= N; i++ {
			x[0+0][i] = +x[1][i]
			x[N+1][i] = +x[N][i]
			x[i][0+0] = -x[i][1]
			x[i][N+1] = -x[i][N]
		}
	}
	x[0+0][0+0] = 0.5 * (x[1][0+0] + x[0+0][1])
	x[0+0][N+1] = 0.5 * (x[1][N+1] + x[0+0][N])
	x[N+1][0+0] = 0.5 * (x[N][0+0] + x[N+1][1])
	x[N+1][N+1] = 0.5 * (x[N][N+1] + x[N+1][N])
}

func linSolve(b int, x, x0 *array, a, c float32) {
	// This if block isn't part of Stam's GDC03 paper, but it's a nice
	// optimization when the diff diffusion parameter is zero.
	if a == 0 && c == 1 {
		for i := 1; i <= N; i++ {
			for j := 1; j <= N; j++ {
				x[i][j] = x0[i][j]
			}
		}
		setBnd(b, x)
		return
	}

	invC := 1 / c
	for k := 0; k < iterations; k++ {
		for i := 1; i <= N; i++ {
			for j := 1; j <= N; j++ {
				x[i][j] = (x0[i][j] + a*(x[i-1][j]+x[i+1][j]+x[i][j-1]+x[i][j+1])) * invC
			}
		}
		setBnd(b, x)
	}
}

func diffuse(b int, x, x0 *array, diff float32) {
	a := dt * diff * N * N
	linSolve(b, x, x0, a, 1+4*a)
}

func advect(b int, d, d0, u, v *array) {
	const dt0 = dt * N
	for i := 1; i <= N; i++ {
		for j := 1; j <= N; j++ {
			x := float32(i) - dt0*u[i][j]
			if x < 0.5 {
				x = 0.5
			}
			if x > N+0.5 {
				x = N + 0.5
			}
			i0 := int(x)
			i1 := i0 + 1

			y := float32(j) - dt0*v[i][j]
			if y < 0.5 {
				y = 0.5
			}
			if y > N+0.5 {
				y = N + 0.5
			}
			j0 := int(y)
			j1 := j0 + 1

			s1 := x - float32(i0)
			s0 := 1 - s1
			t1 := y - float32(j0)
			t0 := 1 - t1
			d[i][j] = s0*(t0*d0[i0][j0]+t1*d0[i0][j1]) + s1*(t0*d0[i1][j0]+t1*d0[i1][j1])
		}
	}
	setBnd(b, d)
}

func project(u, v, p, div *array) {
	for i := 1; i <= N; i++ {
		for j := 1; j <= N; j++ {
			div[i][j] = (u[i+1][j] - u[i-1][j] + v[i][j+1] - v[i][j-1]) / (-2 * N)
			p[i][j] = 0
		}
	}
	setBnd(0, div)
	setBnd(0, p)
	linSolve(0, p, div, 1, 4)
	for i := 1; i <= N; i++ {
		for j := 1; j <= N; j++ {
			u[i][j] -= (N / 2) * (p[i+1][j+0] - p[i-1][j+0])
			v[i][j] -= (N / 2) * (p[i+0][j+1] - p[i+0][j-1])
		}
	}
	setBnd(1, u)
	setBnd(2, v)
}

func velStep(u, v, u0, v0 *array) {
	addSource(u, u0)
	addSource(v, v0)
	u0, u = u, u0
	diffuse(1, u, u0, visc)
	v0, v = v, v0
	diffuse(2, v, v0, visc)
	project(u, v, u0, v0)
	u0, u = u, u0
	v0, v = v, v0
	advect(1, u, u0, u0, v0)
	advect(2, v, v0, u0, v0)
	project(u, v, u0, v0)
}

func densStep(x, x0, u, v *array) {
	addSource(x, x0)
	x0, x = x, x0
	diffuse(0, x, x0, diff)
	x0, x = x, x0
	advect(0, x, x0, u, v)
}
