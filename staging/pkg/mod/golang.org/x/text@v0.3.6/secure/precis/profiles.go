// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package precis

import (
	"unicode"

	"golang.org/x/text/runes"
	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"
)

var (
	// Implements the Nickname profile specified in RFC 8266.
	Nickname *Profile = nickname

	// Implements the UsernameCaseMapped profile specified in RFC 8265.
	UsernameCaseMapped *Profile = usernameCaseMap

	// Implements the UsernameCasePreserved profile specified in RFC 8265.
	UsernameCasePreserved *Profile = usernameNoCaseMap

	// Implements the OpaqueString profile defined in RFC 8265 for passwords and
	// other secure labels.
	OpaqueString *Profile = opaquestring
)

var (
	nickname = &Profile{
		options: getOpts(
			AdditionalMapping(func() transform.Transformer {
				return &nickAdditionalMapping{}
			}),
			IgnoreCase,
			Norm(norm.NFKC),
			DisallowEmpty,
			repeat,
		),
		class: freeform,
	}
	usernameCaseMap = &Profile{
		options: getOpts(
			FoldWidth,
			LowerCase(),
			Norm(norm.NFC),
			BidiRule,
		),
		class: identifier,
	}
	usernameNoCaseMap = &Profile{
		options: getOpts(
			FoldWidth,
			Norm(norm.NFC),
			BidiRule,
		),
		class: identifier,
	}
	opaquestring = &Profile{
		options: getOpts(
			AdditionalMapping(func() transform.Transformer {
				return mapSpaces
			}),
			Norm(norm.NFC),
			DisallowEmpty,
		),
		class: freeform,
	}
)

// mapSpaces is a shared value of a runes.Map transformer.
var mapSpaces transform.Transformer = runes.Map(func(r rune) rune {
	if unicode.Is(unicode.Zs, r) {
		return ' '
	}
	return r
})
