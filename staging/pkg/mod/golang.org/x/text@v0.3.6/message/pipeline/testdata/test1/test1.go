// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "golang.org/x/text/message"

func main() {
	p := message.NewPrinter(message.MatchLanguage("en"))

	// NOT EXTRACTED: strings passed to Println are not extracted.
	p.Println("Hello world!")

	// NOT EXTRACTED: strings passed to Print are not extracted.
	p.Print("Hello world!\n")

	// Extract and trim whitespace (TODO).
	p.Printf("Hello world!\n")

	// NOT EXTRACTED: city is not used as a pattern or passed to %m.
	city := "Amsterdam"
	// This comment is extracted.
	p.Printf("Hello %s!\n", city)

	person := "Sheila"
	place := "Zürich"

	// Substitutions replaced by variable names.
	p.Printf("%s is visiting %s!\n",
		person, // The person of matter.
		place,  // Place the person is visiting.
	)

	pp := struct {
		Person string // The person of matter. // TODO: get this comment.
		Place  string
		extra  int
	}{
		person, place, 4,
	}

	// extract will drop this comment in favor of the one below.
	p.Printf("%[1]s is visiting %[3]s!\n", // Field names are placeholders.
		pp.Person,
		pp.extra,
		pp.Place, // Place the person is visiting.
	)

	// Numeric literal becomes placeholder.
	p.Printf("%d files remaining!", 2)

	const n = 2

	// Constant identifier becomes placeholder.
	p.Printf("%d more files remaining!", n)

	// Infer better names from type names.
	type referralCode int

	const c = referralCode(5)

	// Use type name as placeholder.
	p.Printf("Use the following code for your discount: %d\n", c)

	// Use constant name as message ID.
	const msgOutOfOrder = "%s is out of order!" // This comment wins.
	const device = "Soda machine"
	// This message has two IDs.
	p.Printf(msgOutOfOrder, device)

	// Multiple substitutions for same argument.
	miles := 1.2345
	p.Printf("%.2[1]f miles traveled (%[1]f)", miles)
}
