// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package precis

import (
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/transform"
)

type nickAdditionalMapping struct {
	// TODO: This transformer needs to be stateless somehow…
	notStart  bool
	prevSpace bool
}

func (t *nickAdditionalMapping) Reset() {
	t.prevSpace = false
	t.notStart = false
}

func (t *nickAdditionalMapping) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	// RFC 8266 §2.1.  Rules
	//
	// 2.  Additional Mapping Rule: The additional mapping rule consists of
	//     the following sub-rules.
	//
	//     a.  Map any instances of non-ASCII space to SPACE (U+0020); a
	//         non-ASCII space is any Unicode code point having a general
	//         category of "Zs", naturally with the exception of SPACE
	//         (U+0020).  (The inclusion of only ASCII space prevents
	//         confusion with various non-ASCII space code points, many of
	//         which are difficult to reproduce across different input
	//         methods.)
	//
	//     b.  Remove any instances of the ASCII space character at the
	//         beginning or end of a nickname (e.g., "stpeter " is mapped to
	//         "stpeter").
	//
	//     c.  Map interior sequences of more than one ASCII space character
	//         to a single ASCII space character (e.g., "St  Peter" is
	//         mapped to "St Peter").
	for nSrc < len(src) {
		r, size := utf8.DecodeRune(src[nSrc:])
		if size == 0 { // Incomplete UTF-8 encoding
			if !atEOF {
				return nDst, nSrc, transform.ErrShortSrc
			}
			size = 1
		}
		if unicode.Is(unicode.Zs, r) {
			t.prevSpace = true
		} else {
			if t.prevSpace && t.notStart {
				dst[nDst] = ' '
				nDst += 1
			}
			if size != copy(dst[nDst:], src[nSrc:nSrc+size]) {
				nDst += size
				return nDst, nSrc, transform.ErrShortDst
			}
			nDst += size
			t.prevSpace = false
			t.notStart = true
		}
		nSrc += size
	}
	return nDst, nSrc, nil
}
