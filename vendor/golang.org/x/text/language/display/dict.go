// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package display

// This file contains sets of data for specific languages. Users can use these
// to create smaller collections of supported languages and reduce total table
// size.

// The variable names defined here correspond to those in package language.

var (
	Afrikaans            *Dictionary = &af        // af
	Amharic              *Dictionary = &am        // am
	Arabic               *Dictionary = &ar        // ar
	ModernStandardArabic *Dictionary = Arabic     // ar-001
	Azerbaijani          *Dictionary = &az        // az
	Bulgarian            *Dictionary = &bg        // bg
	Bengali              *Dictionary = &bn        // bn
	Catalan              *Dictionary = &ca        // ca
	Czech                *Dictionary = &cs        // cs
	Danish               *Dictionary = &da        // da
	German               *Dictionary = &de        // de
	Greek                *Dictionary = &el        // el
	English              *Dictionary = &en        // en
	AmericanEnglish      *Dictionary = English    // en-US
	BritishEnglish       *Dictionary = English    // en-GB
	Spanish              *Dictionary = &es        // es
	EuropeanSpanish      *Dictionary = Spanish    // es-ES
	LatinAmericanSpanish *Dictionary = Spanish    // es-419
	Estonian             *Dictionary = &et        // et
	Persian              *Dictionary = &fa        // fa
	Finnish              *Dictionary = &fi        // fi
	Filipino             *Dictionary = &fil       // fil
	French               *Dictionary = &fr        // fr
	Gujarati             *Dictionary = &gu        // gu
	Hebrew               *Dictionary = &he        // he
	Hindi                *Dictionary = &hi        // hi
	Croatian             *Dictionary = &hr        // hr
	Hungarian            *Dictionary = &hu        // hu
	Armenian             *Dictionary = &hy        // hy
	Indonesian           *Dictionary = &id        // id
	Icelandic            *Dictionary = &is        // is
	Italian              *Dictionary = &it        // it
	Japanese             *Dictionary = &ja        // ja
	Georgian             *Dictionary = &ka        // ka
	Kazakh               *Dictionary = &kk        // kk
	Khmer                *Dictionary = &km        // km
	Kannada              *Dictionary = &kn        // kn
	Korean               *Dictionary = &ko        // ko
	Kirghiz              *Dictionary = &ky        // ky
	Lao                  *Dictionary = &lo        // lo
	Lithuanian           *Dictionary = &lt        // lt
	Latvian              *Dictionary = &lv        // lv
	Macedonian           *Dictionary = &mk        // mk
	Malayalam            *Dictionary = &ml        // ml
	Mongolian            *Dictionary = &mn        // mn
	Marathi              *Dictionary = &mr        // mr
	Malay                *Dictionary = &ms        // ms
	Burmese              *Dictionary = &my        // my
	Nepali               *Dictionary = &ne        // ne
	Dutch                *Dictionary = &nl        // nl
	Norwegian            *Dictionary = &no        // no
	Punjabi              *Dictionary = &pa        // pa
	Polish               *Dictionary = &pl        // pl
	Portuguese           *Dictionary = &pt        // pt
	BrazilianPortuguese  *Dictionary = Portuguese // pt-BR
	EuropeanPortuguese   *Dictionary = &ptPT      // pt-PT
	Romanian             *Dictionary = &ro        // ro
	Russian              *Dictionary = &ru        // ru
	Sinhala              *Dictionary = &si        // si
	Slovak               *Dictionary = &sk        // sk
	Slovenian            *Dictionary = &sl        // sl
	Albanian             *Dictionary = &sq        // sq
	Serbian              *Dictionary = &sr        // sr
	SerbianLatin         *Dictionary = &srLatn    // sr
	Swedish              *Dictionary = &sv        // sv
	Swahili              *Dictionary = &sw        // sw
	Tamil                *Dictionary = &ta        // ta
	Telugu               *Dictionary = &te        // te
	Thai                 *Dictionary = &th        // th
	Turkish              *Dictionary = &tr        // tr
	Ukrainian            *Dictionary = &uk        // uk
	Urdu                 *Dictionary = &ur        // ur
	Uzbek                *Dictionary = &uz        // uz
	Vietnamese           *Dictionary = &vi        // vi
	Chinese              *Dictionary = &zh        // zh
	SimplifiedChinese    *Dictionary = Chinese    // zh-Hans
	TraditionalChinese   *Dictionary = &zhHant    // zh-Hant
	Zulu                 *Dictionary = &zu        // zu
)
