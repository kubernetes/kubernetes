// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"golang.org/x/text/language"
	"golang.org/x/text/message"
)

func main() {
	var nPizzas = 4
	// The following call gets replaced by a call to the globally
	// defined printer.
	fmt.Println("We ate", nPizzas, "pizzas.")

	p := message.NewPrinter(language.English)

	// Prevent build failure, although it is okay for gotext.
	p.Println(1024)

	// Replaced by a call to p.
	fmt.Println("Example punctuation:", "$%^&!")

	{
		q := message.NewPrinter(language.French)

		const leaveAnIdentBe = "Don't expand me."
		fmt.Print(leaveAnIdentBe)
		q.Println() // Prevent build failure, although it is okay for gotext.
	}

	fmt.Printf("Hello %s\n", "City")
}
