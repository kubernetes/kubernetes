// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package message_test

import (
	"fmt"
	"net/http"

	"golang.org/x/text/language"
	"golang.org/x/text/message"
)

func Example_http() {
	// languages supported by this service:
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		lang, _ := r.Cookie("lang")
		accept := r.Header.Get("Accept-Language")
		fallback := "en"
		tag := message.MatchLanguage(lang.String(), accept, fallback)

		p := message.NewPrinter(tag)

		p.Fprintln(w, "User language is", tag)
	})
}

func ExamplePrinter_numbers() {
	for _, lang := range []string{"en", "de", "de-CH", "fr", "bn"} {
		p := message.NewPrinter(language.Make(lang))
		p.Printf("%-6s %g\n", lang, 123456.78)
	}

	// Output:
	// en     123,456.78
	// de     123.456,78
	// de-CH  123’456.78
	// fr     123 456,78
	// bn     ১,২৩,৪৫৬.৭৮
}

func ExamplePrinter_mVerb() {
	message.SetString(language.Dutch, "You have chosen to play %m.", "U heeft ervoor gekozen om %m te spelen.")
	message.SetString(language.Dutch, "basketball", "basketbal")
	message.SetString(language.Dutch, "hockey", "ijshockey")
	message.SetString(language.Dutch, "soccer", "voetbal")
	message.SetString(language.BritishEnglish, "soccer", "football")

	for _, sport := range []string{"soccer", "basketball", "hockey"} {
		for _, lang := range []string{"en", "en-GB", "nl"} {
			p := message.NewPrinter(language.Make(lang))
			fmt.Printf("%-6s %s\n", lang, p.Sprintf("You have chosen to play %m.", sport))
		}
		fmt.Println()
	}

	// Output:
	// en     You have chosen to play soccer.
	// en-GB  You have chosen to play football.
	// nl     U heeft ervoor gekozen om voetbal te spelen.
	//
	// en     You have chosen to play basketball.
	// en-GB  You have chosen to play basketball.
	// nl     U heeft ervoor gekozen om basketbal te spelen.
	//
	// en     You have chosen to play hockey.
	// en-GB  You have chosen to play hockey.
	// nl     U heeft ervoor gekozen om ijshockey te spelen.
}
