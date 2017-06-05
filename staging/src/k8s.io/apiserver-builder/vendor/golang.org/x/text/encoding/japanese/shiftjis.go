// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package japanese

import (
	"errors"
	"unicode/utf8"

	"golang.org/x/text/encoding"
	"golang.org/x/text/encoding/internal"
	"golang.org/x/text/encoding/internal/identifier"
	"golang.org/x/text/transform"
)

// ShiftJIS is the Shift JIS encoding, also known as Code Page 932 and
// Windows-31J.
var ShiftJIS encoding.Encoding = &shiftJIS

var shiftJIS = internal.Encoding{
	&internal.SimpleEncoding{shiftJISDecoder{}, shiftJISEncoder{}},
	"Shift JIS",
	identifier.ShiftJIS,
}

var errInvalidShiftJIS = errors.New("japanese: invalid Shift JIS encoding")

type shiftJISDecoder struct{ transform.NopResetter }

func (shiftJISDecoder) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	r, size := rune(0), 0
loop:
	for ; nSrc < len(src); nSrc += size {
		switch c0 := src[nSrc]; {
		case c0 < utf8.RuneSelf:
			r, size = rune(c0), 1

		case 0xa1 <= c0 && c0 < 0xe0:
			r, size = rune(c0)+(0xff61-0xa1), 1

		case (0x81 <= c0 && c0 < 0xa0) || (0xe0 <= c0 && c0 < 0xfd):
			if c0 <= 0x9f {
				c0 -= 0x70
			} else {
				c0 -= 0xb0
			}
			c0 = 2*c0 - 0x21

			if nSrc+1 >= len(src) {
				err = transform.ErrShortSrc
				break loop
			}
			c1 := src[nSrc+1]
			switch {
			case c1 < 0x40:
				err = errInvalidShiftJIS
				break loop
			case c1 < 0x7f:
				c0--
				c1 -= 0x40
			case c1 == 0x7f:
				err = errInvalidShiftJIS
				break loop
			case c1 < 0x9f:
				c0--
				c1 -= 0x41
			case c1 < 0xfd:
				c1 -= 0x9f
			default:
				err = errInvalidShiftJIS
				break loop
			}
			r, size = '\ufffd', 2
			if i := int(c0)*94 + int(c1); i < len(jis0208Decode) {
				r = rune(jis0208Decode[i])
				if r == 0 {
					r = '\ufffd'
				}
			}

		default:
			err = errInvalidShiftJIS
			break loop
		}

		if nDst+utf8.RuneLen(r) > len(dst) {
			err = transform.ErrShortDst
			break loop
		}
		nDst += utf8.EncodeRune(dst[nDst:], r)
	}
	if atEOF && err == transform.ErrShortSrc {
		err = errInvalidShiftJIS
	}
	return nDst, nSrc, err
}

type shiftJISEncoder struct{ transform.NopResetter }

func (shiftJISEncoder) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	r, size := rune(0), 0
loop:
	for ; nSrc < len(src); nSrc += size {
		r = rune(src[nSrc])

		// Decode a 1-byte rune.
		if r < utf8.RuneSelf {
			size = 1

		} else {
			// Decode a multi-byte rune.
			r, size = utf8.DecodeRune(src[nSrc:])
			if size == 1 {
				// All valid runes of size 1 (those below utf8.RuneSelf) were
				// handled above. We have invalid UTF-8 or we haven't seen the
				// full character yet.
				if !atEOF && !utf8.FullRune(src[nSrc:]) {
					err = transform.ErrShortSrc
					break loop
				}
			}

			// func init checks that the switch covers all tables.
			switch {
			case encode0Low <= r && r < encode0High:
				if r = rune(encode0[r-encode0Low]); r>>tableShift == jis0208 {
					goto write2
				}
			case encode1Low <= r && r < encode1High:
				if r = rune(encode1[r-encode1Low]); r>>tableShift == jis0208 {
					goto write2
				}
			case encode2Low <= r && r < encode2High:
				if r = rune(encode2[r-encode2Low]); r>>tableShift == jis0208 {
					goto write2
				}
			case encode3Low <= r && r < encode3High:
				if r = rune(encode3[r-encode3Low]); r>>tableShift == jis0208 {
					goto write2
				}
			case encode4Low <= r && r < encode4High:
				if r = rune(encode4[r-encode4Low]); r>>tableShift == jis0208 {
					goto write2
				}
			case encode5Low <= r && r < encode5High:
				if 0xff61 <= r && r < 0xffa0 {
					r -= 0xff61 - 0xa1
					goto write1
				}
				if r = rune(encode5[r-encode5Low]); r>>tableShift == jis0208 {
					goto write2
				}
			}
			err = internal.ErrASCIIReplacement
			break
		}

	write1:
		if nDst >= len(dst) {
			err = transform.ErrShortDst
			break
		}
		dst[nDst] = uint8(r)
		nDst++
		continue

	write2:
		j1 := uint8(r>>codeShift) & codeMask
		j2 := uint8(r) & codeMask
		if nDst+2 > len(dst) {
			err = transform.ErrShortDst
			break loop
		}
		if j1 <= 61 {
			dst[nDst+0] = 129 + j1/2
		} else {
			dst[nDst+0] = 193 + j1/2
		}
		if j1&1 == 0 {
			dst[nDst+1] = j2 + j2/63 + 64
		} else {
			dst[nDst+1] = j2 + 159
		}
		nDst += 2
		continue
	}
	return nDst, nSrc, err
}
