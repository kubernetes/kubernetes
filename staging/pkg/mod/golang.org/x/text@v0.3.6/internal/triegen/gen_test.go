// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build generate
// +build generate

package triegen_test

// The code in this file generates captures and writes the tries generated in
// the examples to data_test.go. To invoke it, run:
// 		go test -tags=generate
//
// Making the generation code a "test" allows us to link in the necessary test
// code.

import (
	"log"
	"os"
	"os/exec"
)

func init() {
	const tmpfile = "tmpout"
	const dstfile = "data_test.go"

	f, err := os.Create(tmpfile)
	if err != nil {
		log.Fatalf("Could not create output file: %v", err)
	}
	defer os.Remove(tmpfile)
	defer f.Close()

	// We exit before this function returns, regardless of success or failure,
	// so there's no need to save (and later restore) the existing genWriter
	// value.
	genWriter = f

	f.Write([]byte(header))

	Example_build()
	ExampleGen_build()

	if err := exec.Command("gofmt", "-w", tmpfile).Run(); err != nil {
		log.Fatal(err)
	}
	os.Remove(dstfile)
	os.Rename(tmpfile, dstfile)

	os.Exit(0)
}

const header = `// This file is generated with "go test -tags generate". DO NOT EDIT!
// +build !generate

package triegen_test
`

// Stubs for generated tries. These are needed as we exclude data_test.go if
// the generate flag is set. This will clearly make the tests fail, but that
// is okay. It allows us to bootstrap.

type trie struct{}

func (t *trie) lookupString(string) (uint8, int) { return 0, 1 }
func (t *trie) lookupStringUnsafe(string) uint64 { return 0 }

func newRandTrie(i int) *trie  { return &trie{} }
func newMultiTrie(i int) *trie { return &trie{} }
