// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore
//
// This build tag means that "go build" does not build this file. Use "go run
// manual_test_program.go" to run it.
//
// You will also need to change "debug = false" to "debug = true" in mmap_*.go.

package main

import (
	"log"
	"math/rand"
	"time"

	"golang.org/x/exp/mmap"
)

var garbage []byte

func main() {
	const filename = "manual_test_program.go"

	for _, explicitClose := range []bool{false, true} {
		r, err := mmap.Open(filename)
		if err != nil {
			log.Fatalf("Open: %v", err)
		}
		if explicitClose {
			r.Close()
		} else {
			// Leak the *mmap.ReaderAt returned by mmap.Open. The finalizer
			// should pick it up, if finalizers run at all.
		}
	}

	println("Finished all explicit Close calls.")
	println("Creating and collecting garbage.")
	println("Look for two munmap log messages.")
	println("Hit Ctrl-C to exit.")

	rng := rand.New(rand.NewSource(1))
	now := time.Now()
	for {
		garbage = make([]byte, rng.Intn(1<<20))
		if time.Since(now) > 1*time.Second {
			now = time.Now()
			print(".")
		}
	}
}
