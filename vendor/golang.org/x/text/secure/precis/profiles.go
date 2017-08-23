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
	Nickname              *Profile = nickname          // Implements the Nickname profile specified in RFC 7700.
	UsernameCaseMapped    *Profile = usernameCaseMap   // Implements the UsernameCaseMapped profile specified in RFC 7613.
	UsernameCasePreserved *Profile = usernameNoCaseMap // Implements the UsernameCasePreserved profile specified in RFC 7613.
	OpaqueString          *Profile = opaquestring      // Implements the OpaqueString profile defined in RFC 7613 for passwords and other secure labels.
)

// TODO: mvl: "Ultimately, I would manually define the structs for the internal
// profiles. This avoid pulling in unneeded tables when they are not used."
var (
	nickname = NewFreeform(
		AdditionalMapping(func() transform.Transformer {
			return &nickAdditionalMapping{}
		}),
		IgnoreCase,
		Norm(norm.NFKC),
		DisallowEmpty,
	)
	usernameCaseMap = NewIdentifier(
		FoldWidth,
		FoldCase(),
		Norm(norm.NFC),
		BidiRule,
	)
	usernameNoCaseMap = NewIdentifier(
		FoldWidth,
		Norm(norm.NFC),
		BidiRule,
	)
	opaquestring = NewFreeform(
		AdditionalMapping(func() transform.Transformer {
			return runes.Map(func(r rune) rune {
				if unicode.Is(unicode.Zs, r) {
					return ' '
				}
				return r
			})
		}),
		Norm(norm.NFC),
		DisallowEmpty,
	)
)
