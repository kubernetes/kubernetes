// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// mkpost processes the output of cgo -godefs to
// modify the generated types. It is used to clean up
// the sys API in an architecture specific manner.
//
// mkpost is run after cgo -godefs by mkall.sh.
package main

import (
	"fmt"
	"go/format"
	"io/ioutil"
	"log"
	"os"
	"regexp"
)

func main() {
	b, err := ioutil.ReadAll(os.Stdin)
	if err != nil {
		log.Fatal(err)
	}
	s := string(b)

	goarch := os.Getenv("GOARCH")
	goos := os.Getenv("GOOS")
	if goarch == "s390x" && goos == "linux" {
		// Export the types of PtraceRegs fields.
		re := regexp.MustCompile("ptrace(Psw|Fpregs|Per)")
		s = re.ReplaceAllString(s, "Ptrace$1")

		// Replace padding fields inserted by cgo with blank identifiers.
		re = regexp.MustCompile("Pad_cgo[A-Za-z0-9_]*")
		s = re.ReplaceAllString(s, "_")

		// Replace other unwanted fields with blank identifiers.
		re = regexp.MustCompile("X_[A-Za-z0-9_]*")
		s = re.ReplaceAllString(s, "_")

		// Replace the control_regs union with a blank identifier for now.
		re = regexp.MustCompile("(Control_regs)\\s+\\[0\\]uint64")
		s = re.ReplaceAllString(s, "_ [0]uint64")
	}

	// gofmt
	b, err = format.Source([]byte(s))
	if err != nil {
		log.Fatal(err)
	}

	// Append this command to the header to show where the new file
	// came from.
	re := regexp.MustCompile("(cgo -godefs [a-zA-Z0-9_]+\\.go.*)")
	b = re.ReplaceAll(b, []byte("$1 | go run mkpost.go"))

	fmt.Printf("%s", b)
}
