// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package idna_test

import (
	"fmt"

	"golang.org/x/text/internal/export/idna"
)

func ExampleProfile() {
	// Raw Punycode has no restrictions and does no mappings.
	fmt.Println(idna.ToASCII(""))
	fmt.Println(idna.ToASCII("*.faß.com"))
	fmt.Println(idna.Punycode.ToASCII("*.faß.com"))

	// Rewrite IDN for lookup. This (currently) uses transitional mappings to
	// find a balance between IDNA2003 and IDNA2008 compatibility.
	fmt.Println(idna.Lookup.ToASCII(""))
	fmt.Println(idna.Lookup.ToASCII("www.faß.com"))

	// Convert an IDN to ASCII for registration purposes. This changes the
	// encoding, but reports an error if the input was illformed.
	fmt.Println(idna.Registration.ToASCII(""))
	fmt.Println(idna.Registration.ToASCII("www.faß.com"))

	// Output:
	//  <nil>
	// *.xn--fa-hia.com <nil>
	// *.xn--fa-hia.com <nil>
	//  <nil>
	// www.fass.com <nil>
	//  idna: invalid label ""
	// www.xn--fa-hia.com <nil>
}

func ExampleNew() {
	var p *idna.Profile

	// Raw Punycode has no restrictions and does no mappings.
	p = idna.New()
	fmt.Println(p.ToASCII("*.faß.com"))

	// Do mappings. Note that star is not allowed in a DNS lookup.
	p = idna.New(
		idna.MapForLookup(),
		idna.Transitional(true)) // Map ß -> ss
	fmt.Println(p.ToASCII("*.faß.com"))

	// Lookup for registration. Also does not allow '*'.
	p = idna.New(idna.ValidateForRegistration())
	fmt.Println(p.ToUnicode("*.faß.com"))

	// Set up a profile maps for lookup, but allows wild cards.
	p = idna.New(
		idna.MapForLookup(),
		idna.Transitional(true),      // Map ß -> ss
		idna.StrictDomainName(false)) // Set more permissive ASCII rules.
	fmt.Println(p.ToASCII("*.faß.com"))

	// Output:
	// *.xn--fa-hia.com <nil>
	// *.fass.com idna: disallowed rune U+002A
	// *.faß.com idna: disallowed rune U+002A
	// *.fass.com <nil>
}
