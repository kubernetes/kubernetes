// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package publicsuffix_test

import (
	"fmt"
	"strings"

	"golang.org/x/net/publicsuffix"
)

// This example demonstrates looking up several domains' eTLDs (effective Top
// Level Domains) in the PSL (Public Suffix List) snapshot. For each eTLD, the
// example also determines whether the eTLD is ICANN managed, privately
// managed, or unmanaged (not explicitly in the PSL).
//
// See https://publicsuffix.org/ for the underlying PSL data.
func ExamplePublicSuffix_manager() {
	domains := []string{
		"amazon.co.uk",
		"books.amazon.co.uk",
		"www.books.amazon.co.uk",
		"amazon.com",
		"",
		"example0.debian.net",
		"example1.debian.org",
		"",
		"golang.dev",
		"golang.net",
		"play.golang.org",
		"gophers.in.space.museum",
		"",
		"0emm.com",
		"a.0emm.com",
		"b.c.d.0emm.com",
		"",
		"there.is.no.such-tld",
		"",
		// Examples from the PublicSuffix function's documentation.
		"foo.org",
		"foo.co.uk",
		"foo.dyndns.org",
		"foo.blogspot.co.uk",
		"cromulent",
	}

	for _, domain := range domains {
		if domain == "" {
			fmt.Println(">")
			continue
		}
		eTLD, icann := publicsuffix.PublicSuffix(domain)

		// Only ICANN managed domains can have a single label. Privately
		// managed domains must have multiple labels.
		manager := "Unmanaged"
		if icann {
			manager = "ICANN Managed"
		} else if strings.IndexByte(eTLD, '.') >= 0 {
			manager = "Privately Managed"
		}

		fmt.Printf("> %24s%16s  is  %s\n", domain, eTLD, manager)
	}

	// Output:
	// >             amazon.co.uk           co.uk  is  ICANN Managed
	// >       books.amazon.co.uk           co.uk  is  ICANN Managed
	// >   www.books.amazon.co.uk           co.uk  is  ICANN Managed
	// >               amazon.com             com  is  ICANN Managed
	// >
	// >      example0.debian.net      debian.net  is  Privately Managed
	// >      example1.debian.org             org  is  ICANN Managed
	// >
	// >               golang.dev             dev  is  ICANN Managed
	// >               golang.net             net  is  ICANN Managed
	// >          play.golang.org             org  is  ICANN Managed
	// >  gophers.in.space.museum    space.museum  is  ICANN Managed
	// >
	// >                 0emm.com             com  is  ICANN Managed
	// >               a.0emm.com      a.0emm.com  is  Privately Managed
	// >           b.c.d.0emm.com      d.0emm.com  is  Privately Managed
	// >
	// >     there.is.no.such-tld        such-tld  is  Unmanaged
	// >
	// >                  foo.org             org  is  ICANN Managed
	// >                foo.co.uk           co.uk  is  ICANN Managed
	// >           foo.dyndns.org      dyndns.org  is  Privately Managed
	// >       foo.blogspot.co.uk  blogspot.co.uk  is  Privately Managed
	// >                cromulent       cromulent  is  Unmanaged
}
