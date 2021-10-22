// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pflag_test

import (
	"fmt"

	"github.com/spf13/pflag"
)

func ExampleShorthandLookup() {
	name := "verbose"
	short := name[:1]

	pflag.BoolP(name, short, false, "verbose output")

	// len(short) must be == 1
	flag := pflag.ShorthandLookup(short)

	fmt.Println(flag.Name)
}

func ExampleFlagSet_ShorthandLookup() {
	name := "verbose"
	short := name[:1]

	fs := pflag.NewFlagSet("Example", pflag.ContinueOnError)
	fs.BoolP(name, short, false, "verbose output")

	// len(short) must be == 1
	flag := fs.ShorthandLookup(short)

	fmt.Println(flag.Name)
}
