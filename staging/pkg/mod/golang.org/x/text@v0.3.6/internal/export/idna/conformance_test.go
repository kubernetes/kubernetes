// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package idna

import (
	"fmt"
	"strings"
	"testing"

	"golang.org/x/text/internal/gen"
	"golang.org/x/text/internal/testtext"
	"golang.org/x/text/internal/ucd"
)

func TestConformance(t *testing.T) {
	testtext.SkipIfNotLong(t)

	r := gen.OpenUnicodeFile("idna", "10.0.0", "IdnaTest.txt")
	defer r.Close()

	section := "main"
	p := ucd.New(r)
	transitional := New(Transitional(true), VerifyDNSLength(true), BidiRule(), MapForLookup())
	nonTransitional := New(VerifyDNSLength(true), BidiRule(), MapForLookup())
	for p.Next() {
		// What to test
		profiles := []*Profile{}
		switch p.String(0) {
		case "T":
			profiles = append(profiles, transitional)
		case "N":
			profiles = append(profiles, nonTransitional)
		case "B":
			profiles = append(profiles, transitional)
			profiles = append(profiles, nonTransitional)
		}

		src := unescape(p.String(1))

		wantToUnicode := unescape(p.String(2))
		if wantToUnicode == "" {
			wantToUnicode = src
		}
		wantToASCII := unescape(p.String(3))
		if wantToASCII == "" {
			wantToASCII = wantToUnicode
		}
		wantErrToUnicode := ""
		if strings.HasPrefix(wantToUnicode, "[") {
			wantErrToUnicode = wantToUnicode
			wantToUnicode = ""
		}
		wantErrToASCII := ""
		if strings.HasPrefix(wantToASCII, "[") {
			wantErrToASCII = wantToASCII
			wantToASCII = ""
		}

		// TODO: also do IDNA tests.
		// invalidInIDNA2008 := p.String(4) == "NV8"

		for _, p := range profiles {
			name := fmt.Sprintf("%s:%s", section, p)
			doTest(t, p.ToUnicode, name+":ToUnicode", src, wantToUnicode, wantErrToUnicode)
			doTest(t, p.ToASCII, name+":ToASCII", src, wantToASCII, wantErrToASCII)
		}
	}
}
