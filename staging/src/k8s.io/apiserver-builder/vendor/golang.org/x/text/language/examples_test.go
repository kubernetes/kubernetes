// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package language_test

import (
	"fmt"

	"golang.org/x/text/language"
)

func ExampleCanonType() {
	p := func(id string) {
		fmt.Printf("Default(%s) -> %s\n", id, language.Make(id))
		fmt.Printf("BCP47(%s) -> %s\n", id, language.BCP47.Make(id))
		fmt.Printf("Macro(%s) -> %s\n", id, language.Macro.Make(id))
		fmt.Printf("All(%s) -> %s\n", id, language.All.Make(id))
	}
	p("en-Latn")
	p("sh")
	p("zh-cmn")
	p("bjd")
	p("iw-Latn-fonipa-u-cu-usd")
	// Output:
	// Default(en-Latn) -> en-Latn
	// BCP47(en-Latn) -> en
	// Macro(en-Latn) -> en-Latn
	// All(en-Latn) -> en
	// Default(sh) -> sr-Latn
	// BCP47(sh) -> sh
	// Macro(sh) -> sh
	// All(sh) -> sr-Latn
	// Default(zh-cmn) -> cmn
	// BCP47(zh-cmn) -> cmn
	// Macro(zh-cmn) -> zh
	// All(zh-cmn) -> zh
	// Default(bjd) -> drl
	// BCP47(bjd) -> drl
	// Macro(bjd) -> bjd
	// All(bjd) -> drl
	// Default(iw-Latn-fonipa-u-cu-usd) -> he-Latn-fonipa-u-cu-usd
	// BCP47(iw-Latn-fonipa-u-cu-usd) -> he-Latn-fonipa-u-cu-usd
	// Macro(iw-Latn-fonipa-u-cu-usd) -> iw-Latn-fonipa-u-cu-usd
	// All(iw-Latn-fonipa-u-cu-usd) -> he-Latn-fonipa-u-cu-usd
}

func ExampleTag_Base() {
	fmt.Println(language.Make("und").Base())
	fmt.Println(language.Make("und-US").Base())
	fmt.Println(language.Make("und-NL").Base())
	fmt.Println(language.Make("und-419").Base()) // Latin America
	fmt.Println(language.Make("und-ZZ").Base())
	// Output:
	// en Low
	// en High
	// nl High
	// es Low
	// en Low
}

func ExampleTag_Script() {
	en := language.Make("en")
	sr := language.Make("sr")
	sr_Latn := language.Make("sr_Latn")
	fmt.Println(en.Script())
	fmt.Println(sr.Script())
	// Was a script explicitly specified?
	_, c := sr.Script()
	fmt.Println(c == language.Exact)
	_, c = sr_Latn.Script()
	fmt.Println(c == language.Exact)
	// Output:
	// Latn High
	// Cyrl Low
	// false
	// true
}

func ExampleTag_Region() {
	ru := language.Make("ru")
	en := language.Make("en")
	fmt.Println(ru.Region())
	fmt.Println(en.Region())
	// Output:
	// RU Low
	// US Low
}

func ExampleRegion_TLD() {
	us := language.MustParseRegion("US")
	gb := language.MustParseRegion("GB")
	uk := language.MustParseRegion("UK")
	bu := language.MustParseRegion("BU")

	fmt.Println(us.TLD())
	fmt.Println(gb.TLD())
	fmt.Println(uk.TLD())
	fmt.Println(bu.TLD())

	fmt.Println(us.Canonicalize().TLD())
	fmt.Println(gb.Canonicalize().TLD())
	fmt.Println(uk.Canonicalize().TLD())
	fmt.Println(bu.Canonicalize().TLD())
	// Output:
	// US <nil>
	// UK <nil>
	// UK <nil>
	// ZZ language: region is not a valid ccTLD
	// US <nil>
	// UK <nil>
	// UK <nil>
	// MM <nil>
}

func ExampleCompose() {
	nl, _ := language.ParseBase("nl")
	us, _ := language.ParseRegion("US")
	de := language.Make("de-1901-u-co-phonebk")
	jp := language.Make("ja-JP")
	fi := language.Make("fi-x-ing")

	u, _ := language.ParseExtension("u-nu-arabic")
	x, _ := language.ParseExtension("x-piglatin")

	// Combine a base language and region.
	fmt.Println(language.Compose(nl, us))
	// Combine a base language and extension.
	fmt.Println(language.Compose(nl, x))
	// Replace the region.
	fmt.Println(language.Compose(jp, us))
	// Combine several tags.
	fmt.Println(language.Compose(us, nl, u))

	// Replace the base language of a tag.
	fmt.Println(language.Compose(de, nl))
	fmt.Println(language.Compose(de, nl, u))
	// Remove the base language.
	fmt.Println(language.Compose(de, language.Base{}))
	// Remove all variants.
	fmt.Println(language.Compose(de, []language.Variant{}))
	// Remove all extensions.
	fmt.Println(language.Compose(de, []language.Extension{}))
	fmt.Println(language.Compose(fi, []language.Extension{}))
	// Remove all variants and extensions.
	fmt.Println(language.Compose(de.Raw()))

	// An error is gobbled or returned if non-nil.
	fmt.Println(language.Compose(language.ParseRegion("ZA")))
	fmt.Println(language.Compose(language.ParseRegion("HH")))

	// Compose uses the same Default canonicalization as Make.
	fmt.Println(language.Compose(language.Raw.Parse("en-Latn-UK")))

	// Call compose on a different CanonType for different results.
	fmt.Println(language.All.Compose(language.Raw.Parse("en-Latn-UK")))

	// Output:
	// nl-US <nil>
	// nl-x-piglatin <nil>
	// ja-US <nil>
	// nl-US-u-nu-arabic <nil>
	// nl-1901-u-co-phonebk <nil>
	// nl-1901-u-nu-arabic <nil>
	// und-1901-u-co-phonebk <nil>
	// de-u-co-phonebk <nil>
	// de-1901 <nil>
	// fi <nil>
	// de <nil>
	// und-ZA <nil>
	// und language: subtag "HH" is well-formed but unknown
	// en-Latn-GB <nil>
	// en-GB <nil>
}

func ExampleParse_errors() {
	for _, s := range []string{"Foo", "Bar", "Foobar"} {
		_, err := language.Parse(s)
		if err != nil {
			if inv, ok := err.(language.ValueError); ok {
				fmt.Println(inv.Subtag())
			} else {
				fmt.Println(s)
			}
		}
	}
	for _, s := range []string{"en", "aa-Uuuu", "AC", "ac-u"} {
		_, err := language.Parse(s)
		switch e := err.(type) {
		case language.ValueError:
			fmt.Printf("%s: culprit %q\n", s, e.Subtag())
		case nil:
			// No error.
		default:
			// A syntax error.
			fmt.Printf("%s: ill-formed\n", s)
		}
	}
	// Output:
	// foo
	// Foobar
	// aa-Uuuu: culprit "Uuuu"
	// AC: culprit "ac"
	// ac-u: ill-formed
}

func ExampleParent() {
	p := func(tag string) {
		fmt.Printf("parent(%v): %v\n", tag, language.Make(tag).Parent())
	}
	p("zh-CN")

	// Australian English inherits from World English.
	p("en-AU")

	// If the tag has a different maximized script from its parent, a tag with
	// this maximized script is inserted. This allows different language tags
	// which have the same base language and script in common to inherit from
	// a common set of settings.
	p("zh-HK")

	// If the maximized script of the parent is not identical, CLDR will skip
	// inheriting from it, as it means there will not be many entries in common
	// and inheriting from it is nonsensical.
	p("zh-Hant")

	// The parent of a tag with variants and extensions is the tag with all
	// variants and extensions removed.
	p("de-1994-u-co-phonebk")

	// Remove default script.
	p("de-Latn-LU")

	// Output:
	// parent(zh-CN): zh
	// parent(en-AU): en-001
	// parent(zh-HK): zh-Hant
	// parent(zh-Hant): und
	// parent(de-1994-u-co-phonebk): de
	// parent(de-Latn-LU): de
}

// ExampleMatcher_bestMatch gives some examples of getting the best match of
// a set of tags to any of the tags of given set.
func ExampleMatcher() {
	// This is the set of tags from which we want to pick the best match. These
	// can be, for example, the supported languages for some package.
	tags := []language.Tag{
		language.English,
		language.BritishEnglish,
		language.French,
		language.Afrikaans,
		language.BrazilianPortuguese,
		language.EuropeanPortuguese,
		language.Croatian,
		language.SimplifiedChinese,
		language.Raw.Make("iw-IL"),
		language.Raw.Make("iw"),
		language.Raw.Make("he"),
	}
	m := language.NewMatcher(tags)

	// A simple match.
	fmt.Println(m.Match(language.Make("fr")))

	// Australian English is closer to British than American English.
	fmt.Println(m.Match(language.Make("en-AU")))

	// Default to the first tag passed to the Matcher if there is no match.
	fmt.Println(m.Match(language.Make("ar")))

	// Get the default tag.
	fmt.Println(m.Match())

	fmt.Println("----")

	// Croatian speakers will likely understand Serbian written in Latin script.
	fmt.Println(m.Match(language.Make("sr-Latn")))

	// We match SimplifiedChinese, but with Low confidence.
	fmt.Println(m.Match(language.TraditionalChinese))

	// Serbian in Latin script is a closer match to Croatian than Traditional
	// Chinese to Simplified Chinese.
	fmt.Println(m.Match(language.TraditionalChinese, language.Make("sr-Latn")))

	fmt.Println("----")

	// In case a multiple variants of a language are available, the most spoken
	// variant is typically returned.
	fmt.Println(m.Match(language.Portuguese))

	// Pick the first value passed to Match in case of a tie.
	fmt.Println(m.Match(language.Dutch, language.Make("fr-BE"), language.Make("af-NA")))
	fmt.Println(m.Match(language.Dutch, language.Make("af-NA"), language.Make("fr-BE")))

	fmt.Println("----")

	// If a Matcher is initialized with a language and it's deprecated version,
	// it will distinguish between them.
	fmt.Println(m.Match(language.Raw.Make("iw")))

	// However, for non-exact matches, it will treat deprecated versions as
	// equivalent and consider other factors first.
	fmt.Println(m.Match(language.Raw.Make("he-IL")))

	fmt.Println("----")

	// User settings passed to the Unicode extension are ignored for matching
	// and preserved in the returned tag.
	fmt.Println(m.Match(language.Make("de-u-co-phonebk"), language.Make("fr-u-cu-frf")))

	// Even if the matching language is different.
	fmt.Println(m.Match(language.Make("de-u-co-phonebk"), language.Make("br-u-cu-frf")))

	// If there is no matching language, the options of the first preferred tag are used.
	fmt.Println(m.Match(language.Make("de-u-co-phonebk")))

	// Output:
	// fr 2 Exact
	// en-GB 1 High
	// en 0 No
	// en 0 No
	// ----
	// hr 6 High
	// zh-Hans 7 Low
	// hr 6 High
	// ----
	// pt-BR 4 High
	// fr 2 High
	// af 3 High
	// ----
	// iw 9 Exact
	// iw-IL 8 Exact
	// ----
	// fr-u-cu-frf 2 Exact
	// fr-u-cu-frf 2 High
	// en-u-co-phonebk 0 No
}

func ExampleComprehends() {
	// Various levels of comprehensibility.
	fmt.Println(language.Comprehends(language.English, language.English))
	fmt.Println(language.Comprehends(language.AmericanEnglish, language.BritishEnglish))

	// An explicit Und results in no match.
	fmt.Println(language.Comprehends(language.English, language.Und))

	fmt.Println("----")

	// There is usually no mutual comprehensibility between different scripts.
	fmt.Println(language.Comprehends(language.Make("en-Dsrt"), language.English))

	// One exception is for Traditional versus Simplified Chinese, albeit with
	// a low confidence.
	fmt.Println(language.Comprehends(language.TraditionalChinese, language.SimplifiedChinese))

	fmt.Println("----")

	// A Swiss German speaker will often understand High German.
	fmt.Println(language.Comprehends(language.Make("gsw"), language.Make("de")))

	// The converse is not generally the case.
	fmt.Println(language.Comprehends(language.Make("de"), language.Make("gsw")))

	// Output:
	// Exact
	// High
	// No
	// ----
	// No
	// Low
	// ----
	// High
	// No
}

func ExampleTag_values() {
	us := language.MustParseRegion("US")
	en := language.MustParseBase("en")

	lang, _, region := language.AmericanEnglish.Raw()
	fmt.Println(lang == en, region == us)

	lang, _, region = language.BritishEnglish.Raw()
	fmt.Println(lang == en, region == us)

	// Tags can be compared for exact equivalence using '=='.
	en_us, _ := language.Compose(en, us)
	fmt.Println(en_us == language.AmericanEnglish)

	// Output:
	// true true
	// true false
	// true
}
