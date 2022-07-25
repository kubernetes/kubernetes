// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package simplifiedchinese

import (
	"unicode/utf8"

	"golang.org/x/text/encoding"
	"golang.org/x/text/encoding/internal"
	"golang.org/x/text/encoding/internal/identifier"
	"golang.org/x/text/transform"
)

var (
	// GB18030 is the GB18030 encoding.
	GB18030 encoding.Encoding = &gbk18030
	// GBK is the GBK encoding. It encodes an extension of the GB2312 character set
	// and is also known as Code Page 936.
	GBK encoding.Encoding = &gbk
)

var gbk = internal.Encoding{
	&internal.SimpleEncoding{
		gbkDecoder{gb18030: false},
		gbkEncoder{gb18030: false},
	},
	"GBK",
	identifier.GBK,
}

var gbk18030 = internal.Encoding{
	&internal.SimpleEncoding{
		gbkDecoder{gb18030: true},
		gbkEncoder{gb18030: true},
	},
	"GB18030",
	identifier.GB18030,
}

type gbkDecoder struct {
	transform.NopResetter
	gb18030 bool
}

func (d gbkDecoder) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	r, size := rune(0), 0
loop:
	for ; nSrc < len(src); nSrc += size {
		switch c0 := src[nSrc]; {
		case c0 < utf8.RuneSelf:
			r, size = rune(c0), 1

		// Microsoft's Code Page 936 extends GBK 1.0 to encode the euro sign U+20AC
		// as 0x80. The HTML5 specification at http://encoding.spec.whatwg.org/#gbk
		// says to treat "gbk" as Code Page 936.
		case c0 == 0x80:
			r, size = '€', 1

		case c0 < 0xff:
			if nSrc+1 >= len(src) {
				if !atEOF {
					err = transform.ErrShortSrc
					break loop
				}
				r, size = utf8.RuneError, 1
				goto write
			}
			c1 := src[nSrc+1]
			switch {
			case 0x40 <= c1 && c1 < 0x7f:
				c1 -= 0x40
			case 0x80 <= c1 && c1 < 0xff:
				c1 -= 0x41
			case d.gb18030 && 0x30 <= c1 && c1 < 0x40:
				if nSrc+3 >= len(src) {
					if !atEOF {
						err = transform.ErrShortSrc
						break loop
					}
					// The second byte here is always ASCII, so we can set size
					// to 1 in all cases.
					r, size = utf8.RuneError, 1
					goto write
				}
				c2 := src[nSrc+2]
				if c2 < 0x81 || 0xff <= c2 {
					r, size = utf8.RuneError, 1
					goto write
				}
				c3 := src[nSrc+3]
				if c3 < 0x30 || 0x3a <= c3 {
					r, size = utf8.RuneError, 1
					goto write
				}
				size = 4
				r = ((rune(c0-0x81)*10+rune(c1-0x30))*126+rune(c2-0x81))*10 + rune(c3-0x30)
				if r < 39420 {
					i, j := 0, len(gb18030)
					for i < j {
						h := i + (j-i)/2
						if r >= rune(gb18030[h][0]) {
							i = h + 1
						} else {
							j = h
						}
					}
					dec := &gb18030[i-1]
					r += rune(dec[1]) - rune(dec[0])
					goto write
				}
				r -= 189000
				if 0 <= r && r < 0x100000 {
					r += 0x10000
				} else {
					r, size = utf8.RuneError, 1
				}
				goto write
			default:
				r, size = utf8.RuneError, 1
				goto write
			}
			r, size = '\ufffd', 2
			if i := int(c0-0x81)*190 + int(c1); i < len(decode) {
				r = rune(decode[i])
				if r == 0 {
					r = '\ufffd'
				}
			}

		default:
			r, size = utf8.RuneError, 1
		}

	write:
		if nDst+utf8.RuneLen(r) > len(dst) {
			err = transform.ErrShortDst
			break loop
		}
		nDst += utf8.EncodeRune(dst[nDst:], r)
	}
	return nDst, nSrc, err
}

type gbkEncoder struct {
	transform.NopResetter
	gb18030 bool
}

func (e gbkEncoder) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	r, r2, size := rune(0), rune(0), 0
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
					break
				}
			}

			// func init checks that the switch covers all tables.
			switch {
			case encode0Low <= r && r < encode0High:
				if r2 = rune(encode0[r-encode0Low]); r2 != 0 {
					goto write2
				}
			case encode1Low <= r && r < encode1High:
				// Microsoft's Code Page 936 extends GBK 1.0 to encode the euro sign U+20AC
				// as 0x80. The HTML5 specification at http://encoding.spec.whatwg.org/#gbk
				// says to treat "gbk" as Code Page 936.
				if r == '€' {
					r = 0x80
					goto write1
				}
				if r2 = rune(encode1[r-encode1Low]); r2 != 0 {
					goto write2
				}
			case encode2Low <= r && r < encode2High:
				if r2 = rune(encode2[r-encode2Low]); r2 != 0 {
					goto write2
				}
			case encode3Low <= r && r < encode3High:
				if r2 = rune(encode3[r-encode3Low]); r2 != 0 {
					goto write2
				}
			case encode4Low <= r && r < encode4High:
				if r2 = rune(encode4[r-encode4Low]); r2 != 0 {
					goto write2
				}
			}

			if e.gb18030 {
				if r < 0x10000 {
					i, j := 0, len(gb18030)
					for i < j {
						h := i + (j-i)/2
						if r >= rune(gb18030[h][1]) {
							i = h + 1
						} else {
							j = h
						}
					}
					dec := &gb18030[i-1]
					r += rune(dec[0]) - rune(dec[1])
					goto write4
				} else if r < 0x110000 {
					r += 189000 - 0x10000
					goto write4
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
		if nDst+2 > len(dst) {
			err = transform.ErrShortDst
			break
		}
		dst[nDst+0] = uint8(r2 >> 8)
		dst[nDst+1] = uint8(r2)
		nDst += 2
		continue

	write4:
		if nDst+4 > len(dst) {
			err = transform.ErrShortDst
			break
		}
		dst[nDst+3] = uint8(r%10 + 0x30)
		r /= 10
		dst[nDst+2] = uint8(r%126 + 0x81)
		r /= 126
		dst[nDst+1] = uint8(r%10 + 0x30)
		r /= 10
		dst[nDst+0] = uint8(r + 0x81)
		nDst += 4
		continue
	}
	return nDst, nSrc, err
}

func init() {
	// Check that the hard-coded encode switch covers all tables.
	if numEncodeTables != 5 {
		panic("bad numEncodeTables")
	}
}
