// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package simplifiedchinese

import (
	"errors"
	"unicode/utf8"

	"golang.org/x/text/encoding"
	"golang.org/x/text/encoding/internal"
	"golang.org/x/text/encoding/internal/identifier"
	"golang.org/x/text/transform"
)

// HZGB2312 is the HZ-GB2312 encoding.
var HZGB2312 encoding.Encoding = &hzGB2312

var hzGB2312 = internal.Encoding{
	internal.FuncEncoding{hzGB2312NewDecoder, hzGB2312NewEncoder},
	"HZ-GB2312",
	identifier.HZGB2312,
}

func hzGB2312NewDecoder() transform.Transformer {
	return new(hzGB2312Decoder)
}

func hzGB2312NewEncoder() transform.Transformer {
	return new(hzGB2312Encoder)
}

var errInvalidHZGB2312 = errors.New("simplifiedchinese: invalid HZ-GB2312 encoding")

const (
	asciiState = iota
	gbState
)

type hzGB2312Decoder int

func (d *hzGB2312Decoder) Reset() {
	*d = asciiState
}

func (d *hzGB2312Decoder) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	r, size := rune(0), 0
loop:
	for ; nSrc < len(src); nSrc += size {
		c0 := src[nSrc]
		if c0 >= utf8.RuneSelf {
			err = errInvalidHZGB2312
			break loop
		}

		if c0 == '~' {
			if nSrc+1 >= len(src) {
				err = transform.ErrShortSrc
				break loop
			}
			size = 2
			switch src[nSrc+1] {
			case '{':
				*d = gbState
				continue
			case '}':
				*d = asciiState
				continue
			case '~':
				if nDst >= len(dst) {
					err = transform.ErrShortDst
					break loop
				}
				dst[nDst] = '~'
				nDst++
				continue
			case '\n':
				continue
			default:
				err = errInvalidHZGB2312
				break loop
			}
		}

		if *d == asciiState {
			r, size = rune(c0), 1
		} else {
			if nSrc+1 >= len(src) {
				err = transform.ErrShortSrc
				break loop
			}
			c1 := src[nSrc+1]
			if c0 < 0x21 || 0x7e <= c0 || c1 < 0x21 || 0x7f <= c1 {
				err = errInvalidHZGB2312
				break loop
			}

			r, size = '\ufffd', 2
			if i := int(c0-0x01)*190 + int(c1+0x3f); i < len(decode) {
				r = rune(decode[i])
				if r == 0 {
					r = '\ufffd'
				}
			}
		}

		if nDst+utf8.RuneLen(r) > len(dst) {
			err = transform.ErrShortDst
			break loop
		}
		nDst += utf8.EncodeRune(dst[nDst:], r)
	}
	if atEOF && err == transform.ErrShortSrc {
		err = errInvalidHZGB2312
	}
	return nDst, nSrc, err
}

type hzGB2312Encoder int

func (d *hzGB2312Encoder) Reset() {
	*d = asciiState
}

func (e *hzGB2312Encoder) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	r, size := rune(0), 0
	for ; nSrc < len(src); nSrc += size {
		r = rune(src[nSrc])

		// Decode a 1-byte rune.
		if r < utf8.RuneSelf {
			size = 1
			if r == '~' {
				if nDst+2 > len(dst) {
					err = transform.ErrShortDst
					break
				}
				dst[nDst+0] = '~'
				dst[nDst+1] = '~'
				nDst += 2
				continue
			} else if *e != asciiState {
				if nDst+3 > len(dst) {
					err = transform.ErrShortDst
					break
				}
				*e = asciiState
				dst[nDst+0] = '~'
				dst[nDst+1] = '}'
				nDst += 2
			} else if nDst >= len(dst) {
				err = transform.ErrShortDst
				break
			}
			dst[nDst] = uint8(r)
			nDst += 1
			continue

		}

		// Decode a multi-byte rune.
		r, size = utf8.DecodeRune(src[nSrc:])
		if size == 1 {
			// All valid runes of size 1 (those below utf8.RuneSelf) were
			// handled above. We have invalid UTF-8 or we haven't seen the
			// full character yet.
			if !atEOF && !utf8.FullRune(src[nSrc:]) {
				err = transform.ErrShortSrc
				break
			}
		}

		// func init checks that the switch covers all tables.
		switch {
		case encode0Low <= r && r < encode0High:
			if r = rune(encode0[r-encode0Low]); r != 0 {
				goto writeGB
			}
		case encode1Low <= r && r < encode1High:
			if r = rune(encode1[r-encode1Low]); r != 0 {
				goto writeGB
			}
		case encode2Low <= r && r < encode2High:
			if r = rune(encode2[r-encode2Low]); r != 0 {
				goto writeGB
			}
		case encode3Low <= r && r < encode3High:
			if r = rune(encode3[r-encode3Low]); r != 0 {
				goto writeGB
			}
		case encode4Low <= r && r < encode4High:
			if r = rune(encode4[r-encode4Low]); r != 0 {
				goto writeGB
			}
		}

	terminateInASCIIState:
		// Switch back to ASCII state in case of error so that an ASCII
		// replacement character can be written in the correct state.
		if *e != asciiState {
			if nDst+2 > len(dst) {
				err = transform.ErrShortDst
				break
			}
			dst[nDst+0] = '~'
			dst[nDst+1] = '}'
			nDst += 2
		}
		err = internal.ErrASCIIReplacement
		break

	writeGB:
		c0 := uint8(r>>8) - 0x80
		c1 := uint8(r) - 0x80
		if c0 < 0x21 || 0x7e <= c0 || c1 < 0x21 || 0x7f <= c1 {
			goto terminateInASCIIState
		}
		if *e == asciiState {
			if nDst+4 > len(dst) {
				err = transform.ErrShortDst
				break
			}
			*e = gbState
			dst[nDst+0] = '~'
			dst[nDst+1] = '{'
			nDst += 2
		} else if nDst+2 > len(dst) {
			err = transform.ErrShortDst
			break
		}
		dst[nDst+0] = c0
		dst[nDst+1] = c1
		nDst += 2
		continue
	}
	// TODO: should one always terminate in ASCII state to make it safe to
	// concatenate two HZ-GB2312-encoded strings?
	return nDst, nSrc, err
}
