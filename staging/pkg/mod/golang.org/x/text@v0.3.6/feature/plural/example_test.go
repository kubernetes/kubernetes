// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package plural_test

import (
	"golang.org/x/text/feature/plural"
	"golang.org/x/text/language"
	"golang.org/x/text/message"
)

func ExampleSelect() {
	// Manually set some translations. This is typically done programmatically.
	message.Set(language.English, "%d files remaining",
		plural.Selectf(1, "%d",
			"=0", "done!",
			plural.One, "one file remaining",
			plural.Other, "%[1]d files remaining",
		))
	message.Set(language.Dutch, "%d files remaining",
		plural.Selectf(1, "%d",
			"=0", "klaar!",
			// One can also use a string instead of a Kind
			"one", "nog één bestand te gaan",
			"other", "nog %[1]d bestanden te gaan",
		))

	p := message.NewPrinter(language.English)
	p.Printf("%d files remaining", 5)
	p.Println()
	p.Printf("%d files remaining", 1)
	p.Println()

	p = message.NewPrinter(language.Dutch)
	p.Printf("%d files remaining", 1)
	p.Println()
	p.Printf("%d files remaining", 0)
	p.Println()

	// Output:
	// 5 files remaining
	// one file remaining
	// nog één bestand te gaan
	// klaar!
}
