// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ianaindex

import (
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/encoding"
	"golang.org/x/text/encoding/internal"
	"golang.org/x/text/encoding/internal/identifier"
	"golang.org/x/text/transform"
)

type asciiDecoder struct {
	transform.NopResetter
}

func (d asciiDecoder) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	for _, c := range src {
		if c > unicode.MaxASCII {
			r := unicode.ReplacementChar
			if nDst+utf8.RuneLen(r) > len(dst) {
				err = transform.ErrShortDst
				break
			}
			nDst += utf8.EncodeRune(dst[nDst:], r)
			nSrc++
			continue
		}

		if nDst >= len(dst) {
			err = transform.ErrShortDst
			break
		}
		dst[nDst] = c
		nDst++
		nSrc++
	}
	return nDst, nSrc, err
}

type asciiEncoder struct {
	transform.NopResetter
}

func (d asciiEncoder) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	for _, c := range src {
		if c > unicode.MaxASCII {
			err = internal.RepertoireError(encoding.ASCIISub)
			break
		}

		if nDst >= len(dst) {
			err = transform.ErrShortDst
			break
		}
		dst[nDst] = c
		nDst++
		nSrc++
	}
	return nDst, nSrc, err
}

var asciiEnc = &internal.Encoding{
	Encoding: &internal.SimpleEncoding{
		asciiDecoder{},
		asciiEncoder{},
	},
	Name: "US-ASCII",
	MIB:  identifier.ASCII,
}
