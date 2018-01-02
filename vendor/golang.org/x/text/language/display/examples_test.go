// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package display_test

import (
	"fmt"

	"golang.org/x/text/language"
	"golang.org/x/text/language/display"
)

func ExampleNamer() {
	supported := []string{
		"en-US", "en-GB", "ja", "zh", "zh-Hans", "zh-Hant", "pt", "pt-PT", "ko", "ar", "el", "ru", "uk", "pa",
	}

	en := display.English.Languages()

	for _, s := range supported {
		t := language.MustParse(s)
		fmt.Printf("%-20s (%s)\n", en.Name(t), display.Self.Name(t))
	}

	// Output:
	// American English     (American English)
	// British English      (British English)
	// Japanese             (日本語)
	// Chinese              (中文)
	// Simplified Chinese   (简体中文)
	// Traditional Chinese  (繁體中文)
	// Portuguese           (português)
	// European Portuguese  (português europeu)
	// Korean               (한국어)
	// Arabic               (العربية)
	// Greek                (Ελληνικά)
	// Russian              (русский)
	// Ukrainian            (українська)
	// Punjabi              (ਪੰਜਾਬੀ)
}

func ExampleTags() {
	n := display.Tags(language.English)
	fmt.Println(n.Name(language.Make("nl")))
	fmt.Println(n.Name(language.Make("nl-BE")))
	fmt.Println(n.Name(language.Make("nl-CW")))
	fmt.Println(n.Name(language.Make("nl-Arab")))
	fmt.Println(n.Name(language.Make("nl-Cyrl-RU")))

	// Output:
	// Dutch
	// Flemish
	// Dutch (Curaçao)
	// Dutch (Arabic)
	// Dutch (Cyrillic, Russia)
}

// ExampleDictionary shows how to reduce the amount of data linked into your
// binary by only using the predefined Dictionary variables of the languages you
// wish to support.
func ExampleDictionary() {
	tags := []language.Tag{
		language.English,
		language.German,
		language.Japanese,
		language.Russian,
	}
	dicts := []*display.Dictionary{
		display.English,
		display.German,
		display.Japanese,
		display.Russian,
	}

	m := language.NewMatcher(tags)

	getDict := func(t language.Tag) *display.Dictionary {
		_, i, confidence := m.Match(t)
		// Skip this check if you want to support a fall-back language, which
		// will be the first one passed to NewMatcher.
		if confidence == language.No {
			return nil
		}
		return dicts[i]
	}

	// The matcher will match Swiss German to German.
	n := getDict(language.Make("gsw")).Languages()
	fmt.Println(n.Name(language.German))
	fmt.Println(n.Name(language.Make("de-CH")))
	fmt.Println(n.Name(language.Make("gsw")))

	// Output:
	// Deutsch
	// Schweizer Hochdeutsch
	// Schweizerdeutsch
}
