// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package language

type matchTest struct {
	comment   string
	supported string
	test      []struct{ match, desired string }
}

var matchTests = []matchTest{
	{
		"basics",
		"fr, en-GB, en",
		[]struct{ match, desired string }{
			{"en-GB", "en-GB"},
			{"en", "en-US"},
			{"fr", "fr-FR"},
			{"fr", "ja-JP"},
		},
	},
	{
		"script fallbacks",
		"zh-CN, zh-TW, iw",
		[]struct{ match, desired string }{
			{"zh-TW", "zh-Hant"},
			{"zh-CN", "zh"},
			{"zh-CN", "zh-Hans-CN"},
			{"zh-TW", "zh-Hant-HK"},
			{"iw", "he-IT"},
		},
	},
	{
		"language-specific script fallbacks 1",
		"en, sr, nl",
		[]struct{ match, desired string }{
			{"sr", "sr-Latn"},
			{"en", "sh"},
			{"en", "hr"},
			{"en", "bs"},
			// TODO: consider if the following match is a good one.
			// Due to new script first rule, which maybe should be an option.
			{"sr", "nl-Cyrl"},
		},
	},
	{
		"language-specific script fallbacks 2",
		"en, sh",
		[]struct{ match, desired string }{
			{"sh", "sr"},
			{"sh", "sr-Cyrl"},
			{"sh", "hr"},
		},
	},
	{
		"both deprecated and not",
		"fil, tl, iw, he",
		[]struct{ match, desired string }{
			{"he", "he-IT"},
			{"he", "he"},
			{"iw", "iw"},
			{"fil", "fil-IT"},
			{"fil", "fil"},
			{"tl", "tl"},
		},
	},
	{
		"nearby languages",
		"en, fil, ro, nn",
		[]struct{ match, desired string }{
			{"fil", "tl"},
			{"ro", "mo"},
			{"nn", "nb"},
			{"en", "ja"}, // make sure default works
		},
	},
	{
		"nearby languages: Nynorsk to Bokmål",
		"en, nb",
		[]struct{ match, desired string }{
			{"nb", "nn"},
		},
	},
	{
		"nearby languages: Danish does not match nn",
		"en, nn",
		[]struct{ match, desired string }{
			{"en", "da"},
		},
	},
	{
		"nearby languages: Danish matches no",
		"en, no",
		[]struct{ match, desired string }{
			{"no", "da"},
		},
	},
	{
		"nearby languages: Danish matches nb",
		"en, nb",
		[]struct{ match, desired string }{
			{"nb", "da"},
		},
	},
	{
		"prefer matching languages over language variants.",
		"nn, en-GB",
		[]struct{ match, desired string }{
			{"en-GB", "no, en-US"},
			{"en-GB", "nb, en-US"},
		},
	},
	{
		"deprecated version is closer than same language with other differences",
		"nl, he, en-GB",
		[]struct{ match, desired string }{
			{"he", "iw, en-US"},
		},
	},
	{
		"macro equivalent is closer than same language with other differences",
		"nl, zh, en-GB, no",
		[]struct{ match, desired string }{
			{"zh", "cmn, en-US"},
			{"no", "nb, en-US"},
		},
	},
	{
		"legacy equivalent is closer than same language with other differences",
		"nl, fil, en-GB",
		[]struct{ match, desired string }{
			{"fil", "tl, en-US"},
		},
	},
	{
		"exact over equivalent",
		"en, ro, mo, ro-MD",
		[]struct{ match, desired string }{
			{"ro", "ro"},
			{"mo", "mo"},
			{"ro-MD", "ro-MD"},
		},
	},
	{
		"maximization of legacy",
		"sr-Cyrl, sr-Latn, ro, ro-MD",
		[]struct{ match, desired string }{
			{"sr-Latn", "sh"},
			{"ro-MD", "mo"},
		},
	},
	{
		"empty",
		"",
		[]struct{ match, desired string }{
			{"und", "fr"},
			{"und", "en"},
		},
	},
	{
		"private use subtags",
		"fr, en-GB, x-bork, es-ES, es-419",
		[]struct{ match, desired string }{
			{"fr", "x-piglatin"},
			{"x-bork", "x-bork"},
		},
	},
	{
		"grandfathered codes",
		"fr, i-klingon, en-Latn-US",
		[]struct{ match, desired string }{
			{"en-Latn-US", "en-GB-oed"},
			{"tlh", "i-klingon"},
		},
	},
	{
		"exact match",
		"fr, en-GB, ja, es-ES, es-MX",
		[]struct{ match, desired string }{
			{"ja", "ja, de"},
		},
	},
	{
		"simple variant match",
		"fr, en-GB, ja, es-ES, es-MX",
		[]struct{ match, desired string }{
			// Intentionally avoiding a perfect-match or two candidates for variant matches.
			{"en-GB", "de, en-US"},
			// Fall back.
			{"fr", "de, zh"},
		},
	},
	{
		"best match for traditional Chinese",
		// Scenario: An application that only supports Simplified Chinese (and some
		// other languages), but does not support Traditional Chinese. zh-Hans-CN
		// could be replaced with zh-CN, zh, or zh-Hans, it wouldn't make much of
		// a difference.
		"fr, zh-Hans-CN, en-US",
		[]struct{ match, desired string }{
			{"zh-Hans-CN", "zh-TW"},
			{"zh-Hans-CN", "zh-Hant"},
			// One can avoid a zh-Hant to zh-Hans match by including a second language
			// preference which is a better match.
			{"en-US", "zh-TW, en"},
			{"en-US", "zh-Hant-CN, en"},
			{"zh-Hans-CN", "zh-Hans, en"},
		},
	},
	// More specific region and script tie-breakers.
	{
		"more specific script should win in case regions are identical",
		"af, af-Latn, af-Arab",
		[]struct{ match, desired string }{
			{"af", "af"},
			{"af", "af-ZA"},
			{"af-Latn", "af-Latn-ZA"},
			{"af-Latn", "af-Latn"},
		},
	},
	{
		"more specific region should win",
		"nl, nl-NL, nl-BE",
		[]struct{ match, desired string }{
			{"nl", "nl"},
			{"nl", "nl-Latn"},
			{"nl-NL", "nl-Latn-NL"},
			{"nl-NL", "nl-NL"},
		},
	},
	{
		"region may replace matched if matched is enclosing",
		"es-419,es",
		[]struct{ match, desired string }{
			{"es-MX", "es-MX"},
			{"es", "es-SG"},
		},
	},
	{
		"more specific region wins over more specific script",
		"nl, nl-Latn, nl-NL, nl-BE",
		[]struct{ match, desired string }{
			{"nl", "nl"},
			{"nl-Latn", "nl-Latn"},
			{"nl-NL", "nl-NL"},
			{"nl-NL", "nl-Latn-NL"},
		},
	},
	// Region distance tie-breakers.
	{
		"region distance Portuguese",
		"pt, pt-PT",
		[]struct{ match, desired string }{
			{"pt-PT", "pt-ES"},
		},
	},
	{
		"region distance French",
		"en, fr, fr-CA, fr-CH",
		[]struct{ match, desired string }{
			{"fr-CA", "fr-US"},
		},
	},
	{
		"region distance German",
		"de-AT, de-DE, de-CH",
		[]struct{ match, desired string }{
			{"de-DE", "de"},
		},
	},
	{
		"en-AU is closer to en-GB than to en (which is en-US)",
		"en, en-GB, es-ES, es-419",
		[]struct{ match, desired string }{
			{"en-GB", "en-AU"},
			{"es-MX", "es-MX"},
			{"es-ES", "es-PT"},
		},
	},
	// Test exceptions with "und".
	// When the undefined language doesn't match anything in the list, return the default, as usual.
	// max("und") = "en-Latn-US", and since matching is based on maximized tags, the undefined
	// language would normally match English.  But that would produce the counterintuitive results.
	// Matching "und" to "it,en" would be "en" matching "en" to "it,und" would be "und".
	// To avoid this max("und") is defined as "und"
	{
		"undefined",
		"it, fr",
		[]struct{ match, desired string }{
			{"it", "und"},
		},
	},
	{
		"und does not match en",
		"it, en",
		[]struct{ match, desired string }{
			{"it", "und"},
		},
	},
	{
		"undefined in priority list",
		"it, und",
		[]struct{ match, desired string }{
			{"und", "und"},
			{"it", "en"},
		},
	},
	// Undefined scripts and regions.
	{
		"undefined",
		"it, fr, zh",
		[]struct{ match, desired string }{
			{"fr", "und-FR"},
			{"zh", "und-CN"},
			{"zh", "und-Hans"},
			{"zh", "und-Hant"},
			{"it", "und-Latn"},
		},
	},
	// Early termination conditions: do not consider all desired strings if
	// a match is good enough.
	{
		"match on maximized tag",
		"fr, en-GB, ja, es-ES, es-MX",
		[]struct{ match, desired string }{
			// ja-JP matches ja on likely subtags, and it's listed first,
			// thus it wins over the second preference en-GB.
			{"ja", "ja-JP, en-GB"},
			{"ja", "ja-Jpan-JP, en-GB"},
		},
	},
	{
		"pick best maximized tag",
		"ja, ja-Jpan-US, ja-JP, en, ru",
		[]struct{ match, desired string }{
			{"ja", "ja-Jpan, ru"},
			{"ja-JP", "ja-JP, ru"},
			{"ja-Jpan-US", "ja-US, ru"},
		},
	},
	{
		"termination: pick best maximized match",
		"ja, ja-Jpan, ja-JP, en, ru",
		[]struct{ match, desired string }{
			{"ja-JP", "ja-Jpan-JP, ru"},
			{"ja-Jpan", "ja-Jpan, ru"},
		},
	},
	{
		"no match on maximized",
		"en, de, fr, ja",
		[]struct{ match, desired string }{
			// de maximizes to de-DE.
			// Pick the exact match for the secondary language instead.
			{"fr", "de-CH, fr"},
		},
	},

	// Test that the CLDR parent relations are correctly preserved by the matcher.
	// These matches may change for different CLDR versions.
	{
		"parent relation preserved",
		"en, en-US, en-GB, es, es-419, pt, pt-BR, pt-PT, zh,  zh-Hant, zh-Hant-HK",
		[]struct{ match, desired string }{
			{"en-GB", "en-150"},
			// {"en-GB", "en-001"}, // TODO: currently en, should probably be en-GB
			{"en-GB", "en-AU"},
			{"en-GB", "en-BE"},
			{"en-GB", "en-GG"},
			{"en-GB", "en-GI"},
			{"en-GB", "en-HK"},
			{"en-GB", "en-IE"},
			{"en-GB", "en-IM"},
			{"en-GB", "en-IN"},
			{"en-GB", "en-JE"},
			{"en-GB", "en-MT"},
			{"en-GB", "en-NZ"},
			{"en-GB", "en-PK"},
			{"en-GB", "en-SG"},
			{"en-GB", "en-DE"},
			{"en-GB", "en-MT"},
			{"es-AR", "es-AR"},
			{"es-BO", "es-BO"},
			{"es-CL", "es-CL"},
			{"es-CO", "es-CO"},
			{"es-CR", "es-CR"},
			{"es-CU", "es-CU"},
			{"es-DO", "es-DO"},
			{"es-EC", "es-EC"},
			{"es-GT", "es-GT"},
			{"es-HN", "es-HN"},
			{"es-MX", "es-MX"},
			{"es-NI", "es-NI"},
			{"es-PA", "es-PA"},
			{"es-PE", "es-PE"},
			{"es-PR", "es-PR"},
			{"es", "es-PT"},
			{"es-PY", "es-PY"},
			{"es-SV", "es-SV"},
			{"es-419", "es-US"}, // US is not in Latin America, so don't make more specific.
			{"es-UY", "es-UY"},
			{"es-VE", "es-VE"},
			{"pt-PT", "pt-AO"},
			{"pt-PT", "pt-CV"},
			{"pt-PT", "pt-GW"},
			{"pt-PT", "pt-MO"},
			{"pt-PT", "pt-MZ"},
			{"pt-PT", "pt-ST"},
			{"pt-PT", "pt-TL"},
		},
	},
	// Options and variants are inherited from user-defined settings.
	{
		"preserve Unicode extension",
		"en, de, sl-nedis",
		[]struct{ match, desired string }{
			{"de-u-co-phonebk", "de-FR-u-co-phonebk"},
			{"sl-nedis-u-cu-eur", "sl-nedis-u-cu-eur"},
			{"sl-nedis-u-cu-eur", "sl-u-cu-eur"},
			{"sl-nedis-u-cu-eur", "sl-HR-nedis-u-cu-eur"},
		},
	},
}
