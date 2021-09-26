// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:generate gotext update -out catalog.go

import (
	"golang.org/x/text/language"
	"golang.org/x/text/message"
)

func main() {
	p := message.NewPrinter(language.English)

	p.Print("Hello world!\n")

	p.Println("Hello", "world!")

	person := "Sheila"
	place := "Zürich"

	p.Print("Hello ", person, " in ", place, "!\n")

	// Greet everyone.
	p.Printf("Hello world!\n")

	city := "Amsterdam"
	// Greet a city.
	p.Printf("Hello %s!\n", city)

	town := "Amsterdam"
	// Greet a town.
	p.Printf("Hello %s!\n",
		town, // Town
	)

	// Person visiting a place.
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
	// argument is added as a placeholder.
	p.Printf("%[1]s is visiting %[3]s!\n", // Person visiting a place.
		pp.Person,
		pp.extra,
		pp.Place, // Place the person is visiting.
	)

	// Numeric literal
	p.Printf("%d files remaining!", 2)

	const n = 2

	// Numeric var
	p.Printf("%d more files remaining!", n)

	// Infer better names from type names.
	type referralCode int

	const c = referralCode(5)
	p.Printf("Use the following code for your discount: %d\n", c)

	// Using a constant for a message will cause the constant name to be
	// added as an identifier, allowing for stable message identifiers.

	// Explain that a device is out of order.
	const msgOutOfOrder = "%s is out of order!" // FOO
	const device = "Soda machine"
	p.Printf(msgOutOfOrder, device)

	// Double arguments.
	miles := 1.2345
	p.Printf("%.2[1]f miles traveled (%[1]f)", miles)
}
