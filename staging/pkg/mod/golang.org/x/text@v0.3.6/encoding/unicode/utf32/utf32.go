// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package utf32 provides the UTF-32 Unicode encoding.
//
// Please note that support for UTF-32 is discouraged as it is a rare and
// inefficient encoding, unfit for use as an interchange format. For use
// on the web, the W3C strongly discourages its use
// (https://www.w3.org/TR/html5/document-metadata.html#charset)
// while WHATWG directly prohibits supporting it
// (https://html.spec.whatwg.org/multipage/syntax.html#character-encodings).
package utf32 // import "golang.org/x/text/encoding/unicode/utf32"

import (
	"errors"
	"unicode/utf8"

	"golang.org/x/text/encoding"
	"golang.org/x/text/encoding/internal/identifier"
	"golang.org/x/text/transform"
)

// All lists a configuration for each IANA-defined UTF-32 variant.
var All = []encoding.Encoding{
	UTF32(BigEndian, UseBOM),
	UTF32(BigEndian, IgnoreBOM),
	UTF32(LittleEndian, IgnoreBOM),
}

// ErrMissingBOM means that decoding UTF-32 input with ExpectBOM did not
// find a starting byte order mark.
var ErrMissingBOM = errors.New("encoding: missing byte order mark")

// UTF32 returns a UTF-32 Encoding for the given default endianness and
// byte order mark (BOM) policy.
//
// When decoding from UTF-32 to UTF-8, if the BOMPolicy is IgnoreBOM then
// neither BOMs U+FEFF nor ill-formed code units 0xFFFE0000 in the input
// stream will affect the endianness used for decoding. Instead BOMs will
// be output as their standard UTF-8 encoding "\xef\xbb\xbf" while
// 0xFFFE0000 code units will be output as "\xef\xbf\xbd", the standard
// UTF-8 encoding for the Unicode replacement character. If the BOMPolicy
// is UseBOM or ExpectBOM a starting BOM is not written to the UTF-8
// output. Instead, it overrides the default endianness e for the remainder
// of the transformation. Any subsequent BOMs U+FEFF or ill-formed code
// units 0xFFFE0000 will not affect the endianness used, and will instead
// be output as their standard UTF-8 (replacement) encodings. For UseBOM,
// if there is no starting BOM, it will proceed with the default
// Endianness. For ExpectBOM, in that case, the transformation will return
// early with an ErrMissingBOM error.
//
// When encoding from UTF-8 to UTF-32, a BOM will be inserted at the start
// of the output if the BOMPolicy is UseBOM or ExpectBOM. Otherwise, a BOM
// will not be inserted. The UTF-8 input does not need to contain a BOM.
//
// There is no concept of a 'native' endianness. If the UTF-32 data is
// produced and consumed in a greater context that implies a certain
// endianness, use IgnoreBOM. Otherwise, use ExpectBOM and always produce
// and consume a BOM.
//
// In the language of https://www.unicode.org/faq/utf_bom.html#bom10,
// IgnoreBOM corresponds to "Where the precise type of the data stream is
// known... the BOM should not be used" and ExpectBOM corresponds to "A
// particular protocol... may require use of the BOM".
func UTF32(e Endianness, b BOMPolicy) encoding.Encoding {
	return utf32Encoding{config{e, b}, mibValue[e][b&bomMask]}
}

// mibValue maps Endianness and BOMPolicy settings to MIB constants for UTF-32.
// Note that some configurations map to the same MIB identifier.
var mibValue = map[Endianness][numBOMValues]identifier.MIB{
	BigEndian: [numBOMValues]identifier.MIB{
		IgnoreBOM: identifier.UTF32BE,
		UseBOM:    identifier.UTF32,
	},
	LittleEndian: [numBOMValues]identifier.MIB{
		IgnoreBOM: identifier.UTF32LE,
		UseBOM:    identifier.UTF32,
	},
	// ExpectBOM is not widely used and has no valid MIB identifier.
}

// BOMPolicy is a UTF-32 encodings's byte order mark policy.
type BOMPolicy uint8

const (
	writeBOM   BOMPolicy = 0x01
	acceptBOM  BOMPolicy = 0x02
	requireBOM BOMPolicy = 0x04
	bomMask    BOMPolicy = 0x07

	// HACK: numBOMValues == 8 triggers a bug in the 1.4 compiler (cannot have a
	// map of an array of length 8 of a type that is also used as a key or value
	// in another map). See golang.org/issue/11354.
	// TODO: consider changing this value back to 8 if the use of 1.4.* has
	// been minimized.
	numBOMValues = 8 + 1

	// IgnoreBOM means to ignore any byte order marks.
	IgnoreBOM BOMPolicy = 0
	// Unicode-compliant interpretation for UTF-32BE/LE.

	// UseBOM means that the UTF-32 form may start with a byte order mark,
	// which will be used to override the default encoding.
	UseBOM BOMPolicy = writeBOM | acceptBOM
	// Unicode-compliant interpretation for UTF-32.

	// ExpectBOM means that the UTF-32 form must start with a byte order mark,
	// which will be used to override the default encoding.
	ExpectBOM BOMPolicy = writeBOM | acceptBOM | requireBOM
	// Consistent with BOMPolicy definition in golang.org/x/text/encoding/unicode
)

// Endianness is a UTF-32 encoding's default endianness.
type Endianness bool

const (
	// BigEndian is UTF-32BE.
	BigEndian Endianness = false
	// LittleEndian is UTF-32LE.
	LittleEndian Endianness = true
)

type config struct {
	endianness Endianness
	bomPolicy  BOMPolicy
}

type utf32Encoding struct {
	config
	mib identifier.MIB
}

func (u utf32Encoding) NewDecoder() *encoding.Decoder {
	return &encoding.Decoder{Transformer: &utf32Decoder{
		initial: u.config,
		current: u.config,
	}}
}

func (u utf32Encoding) NewEncoder() *encoding.Encoder {
	return &encoding.Encoder{Transformer: &utf32Encoder{
		endianness:       u.endianness,
		initialBOMPolicy: u.bomPolicy,
		currentBOMPolicy: u.bomPolicy,
	}}
}

func (u utf32Encoding) ID() (mib identifier.MIB, other string) {
	return u.mib, ""
}

func (u utf32Encoding) String() string {
	e, b := "B", ""
	if u.endianness == LittleEndian {
		e = "L"
	}
	switch u.bomPolicy {
	case ExpectBOM:
		b = "Expect"
	case UseBOM:
		b = "Use"
	case IgnoreBOM:
		b = "Ignore"
	}
	return "UTF-32" + e + "E (" + b + " BOM)"
}

type utf32Decoder struct {
	initial config
	current config
}

func (u *utf32Decoder) Reset() {
	u.current = u.initial
}

func (u *utf32Decoder) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	if len(src) == 0 {
		if atEOF && u.current.bomPolicy&requireBOM != 0 {
			return 0, 0, ErrMissingBOM
		}
		return 0, 0, nil
	}
	if u.current.bomPolicy&acceptBOM != 0 {
		if len(src) < 4 {
			return 0, 0, transform.ErrShortSrc
		}
		switch {
		case src[0] == 0x00 && src[1] == 0x00 && src[2] == 0xfe && src[3] == 0xff:
			u.current.endianness = BigEndian
			nSrc = 4
		case src[0] == 0xff && src[1] == 0xfe && src[2] == 0x00 && src[3] == 0x00:
			u.current.endianness = LittleEndian
			nSrc = 4
		default:
			if u.current.bomPolicy&requireBOM != 0 {
				return 0, 0, ErrMissingBOM
			}
		}
		u.current.bomPolicy = IgnoreBOM
	}

	var r rune
	var dSize, sSize int
	for nSrc < len(src) {
		if nSrc+3 < len(src) {
			x := uint32(src[nSrc+0])<<24 | uint32(src[nSrc+1])<<16 |
				uint32(src[nSrc+2])<<8 | uint32(src[nSrc+3])
			if u.current.endianness == LittleEndian {
				x = x>>24 | (x >> 8 & 0x0000FF00) | (x << 8 & 0x00FF0000) | x<<24
			}
			r, sSize = rune(x), 4
			if dSize = utf8.RuneLen(r); dSize < 0 {
				r, dSize = utf8.RuneError, 3
			}
		} else if atEOF {
			// 1..3 trailing bytes.
			r, dSize, sSize = utf8.RuneError, 3, len(src)-nSrc
		} else {
			err = transform.ErrShortSrc
			break
		}
		if nDst+dSize > len(dst) {
			err = transform.ErrShortDst
			break
		}
		nDst += utf8.EncodeRune(dst[nDst:], r)
		nSrc += sSize
	}
	return nDst, nSrc, err
}

type utf32Encoder struct {
	endianness       Endianness
	initialBOMPolicy BOMPolicy
	currentBOMPolicy BOMPolicy
}

func (u *utf32Encoder) Reset() {
	u.currentBOMPolicy = u.initialBOMPolicy
}

func (u *utf32Encoder) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	if u.currentBOMPolicy&writeBOM != 0 {
		if len(dst) < 4 {
			return 0, 0, transform.ErrShortDst
		}
		dst[0], dst[1], dst[2], dst[3] = 0x00, 0x00, 0xfe, 0xff
		u.currentBOMPolicy = IgnoreBOM
		nDst = 4
	}

	r, size := rune(0), 0
	for nSrc < len(src) {
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
		}

		if nDst+4 > len(dst) {
			err = transform.ErrShortDst
			break
		}

		dst[nDst+0] = uint8(r >> 24)
		dst[nDst+1] = uint8(r >> 16)
		dst[nDst+2] = uint8(r >> 8)
		dst[nDst+3] = uint8(r)
		nDst += 4
		nSrc += size
	}

	if u.endianness == LittleEndian {
		for i := 0; i < nDst; i += 4 {
			dst[i], dst[i+1], dst[i+2], dst[i+3] = dst[i+3], dst[i+2], dst[i+1], dst[i]
		}
	}
	return nDst, nSrc, err
}
