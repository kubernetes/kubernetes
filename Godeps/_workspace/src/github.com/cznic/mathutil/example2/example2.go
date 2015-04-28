// Copyright (c) 2011 CZ.NIC z.s.p.o. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// blame: jnml, labs.nic.cz

// +build ignore

package main

import (
	"bytes"
	"github.com/cznic/mathutil"
	"image"
	"image/png"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
)

// $ go run example2.go # view rand.png and rnd.png by your favorite pic viewer
//
// see http://www.boallen.com/random-numbers.html
func main() {
	sqr := image.Rect(0, 0, 511, 511)
	r, err := mathutil.NewFC32(math.MinInt32, math.MaxInt32, true)
	if err != nil {
		log.Fatal("NewFC32", err)
	}

	img := image.NewGray(sqr)
	for y := 0; y < 512; y++ {
		for x := 0; x < 512; x++ {
			if r.Next()&1 != 0 {
				img.Set(x, y, image.White)
			}
		}
	}
	buf := bytes.NewBuffer(nil)
	if err := png.Encode(buf, img); err != nil {
		log.Fatal("Encode rnd.png ", err)
	}

	if err := ioutil.WriteFile("rnd.png", buf.Bytes(), 0666); err != nil {
		log.Fatal("ioutil.WriteFile/rnd.png ", err)
	}

	r2 := rand.New(rand.NewSource(0))
	img = image.NewGray(sqr)
	for y := 0; y < 512; y++ {
		for x := 0; x < 512; x++ {
			if r2.Int()&1 != 0 {
				img.Set(x, y, image.White)
			}
		}
	}
	buf = bytes.NewBuffer(nil)
	if err := png.Encode(buf, img); err != nil {
		log.Fatal("Encode rand.png ", err)
	}

	if err := ioutil.WriteFile("rand.png", buf.Bytes(), 0666); err != nil {
		log.Fatal("ioutil.WriteFile/rand.png ", err)
	}
}
