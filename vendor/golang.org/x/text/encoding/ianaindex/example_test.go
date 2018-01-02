// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ianaindex_test

import (
	"fmt"

	"golang.org/x/text/encoding/charmap"
	"golang.org/x/text/encoding/ianaindex"
)

func ExampleIndex() {
	fmt.Println(ianaindex.MIME.Name(charmap.ISO8859_7))
	fmt.Println(ianaindex.IANA.Name(charmap.ISO8859_7))
	fmt.Println(ianaindex.MIB.Name(charmap.ISO8859_7))

	e, _ := ianaindex.IANA.Encoding("cp437")
	fmt.Println(ianaindex.IANA.Name(e))

	// Output:
	// ISO-8859-7 <nil>
	// ISO_8859-7:1987 <nil>
	// ISOLatinGreek <nil>
	// IBM437 <nil>
}
