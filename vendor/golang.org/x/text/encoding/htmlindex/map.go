// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package htmlindex

import (
	"golang.org/x/text/encoding"
	"golang.org/x/text/encoding/charmap"
	"golang.org/x/text/encoding/internal/identifier"
	"golang.org/x/text/encoding/japanese"
	"golang.org/x/text/encoding/korean"
	"golang.org/x/text/encoding/simplifiedchinese"
	"golang.org/x/text/encoding/traditionalchinese"
	"golang.org/x/text/encoding/unicode"
)

// mibMap maps a MIB identifier to an htmlEncoding index.
var mibMap = map[identifier.MIB]htmlEncoding{
	identifier.UTF8:              utf8,
	identifier.UTF16BE:           utf16be,
	identifier.UTF16LE:           utf16le,
	identifier.IBM866:            ibm866,
	identifier.ISOLatin2:         iso8859_2,
	identifier.ISOLatin3:         iso8859_3,
	identifier.ISOLatin4:         iso8859_4,
	identifier.ISOLatinCyrillic:  iso8859_5,
	identifier.ISOLatinArabic:    iso8859_6,
	identifier.ISOLatinGreek:     iso8859_7,
	identifier.ISOLatinHebrew:    iso8859_8,
	identifier.ISO88598I:         iso8859_8I,
	identifier.ISOLatin6:         iso8859_10,
	identifier.ISO885913:         iso8859_13,
	identifier.ISO885914:         iso8859_14,
	identifier.ISO885915:         iso8859_15,
	identifier.ISO885916:         iso8859_16,
	identifier.KOI8R:             koi8r,
	identifier.KOI8U:             koi8u,
	identifier.Macintosh:         macintosh,
	identifier.MacintoshCyrillic: macintoshCyrillic,
	identifier.Windows874:        windows874,
	identifier.Windows1250:       windows1250,
	identifier.Windows1251:       windows1251,
	identifier.Windows1252:       windows1252,
	identifier.Windows1253:       windows1253,
	identifier.Windows1254:       windows1254,
	identifier.Windows1255:       windows1255,
	identifier.Windows1256:       windows1256,
	identifier.Windows1257:       windows1257,
	identifier.Windows1258:       windows1258,
	identifier.XUserDefined:      xUserDefined,
	identifier.GBK:               gbk,
	identifier.GB18030:           gb18030,
	identifier.Big5:              big5,
	identifier.EUCPkdFmtJapanese: eucjp,
	identifier.ISO2022JP:         iso2022jp,
	identifier.ShiftJIS:          shiftJIS,
	identifier.EUCKR:             euckr,
	identifier.Replacement:       replacement,
}

// encodings maps the internal htmlEncoding to an Encoding.
// TODO: consider using a reusable index in encoding/internal.
var encodings = [numEncodings]encoding.Encoding{
	utf8:              unicode.UTF8,
	ibm866:            charmap.CodePage866,
	iso8859_2:         charmap.ISO8859_2,
	iso8859_3:         charmap.ISO8859_3,
	iso8859_4:         charmap.ISO8859_4,
	iso8859_5:         charmap.ISO8859_5,
	iso8859_6:         charmap.ISO8859_6,
	iso8859_7:         charmap.ISO8859_7,
	iso8859_8:         charmap.ISO8859_8,
	iso8859_8I:        charmap.ISO8859_8I,
	iso8859_10:        charmap.ISO8859_10,
	iso8859_13:        charmap.ISO8859_13,
	iso8859_14:        charmap.ISO8859_14,
	iso8859_15:        charmap.ISO8859_15,
	iso8859_16:        charmap.ISO8859_16,
	koi8r:             charmap.KOI8R,
	koi8u:             charmap.KOI8U,
	macintosh:         charmap.Macintosh,
	windows874:        charmap.Windows874,
	windows1250:       charmap.Windows1250,
	windows1251:       charmap.Windows1251,
	windows1252:       charmap.Windows1252,
	windows1253:       charmap.Windows1253,
	windows1254:       charmap.Windows1254,
	windows1255:       charmap.Windows1255,
	windows1256:       charmap.Windows1256,
	windows1257:       charmap.Windows1257,
	windows1258:       charmap.Windows1258,
	macintoshCyrillic: charmap.MacintoshCyrillic,
	gbk:               simplifiedchinese.GBK,
	gb18030:           simplifiedchinese.GB18030,
	big5:              traditionalchinese.Big5,
	eucjp:             japanese.EUCJP,
	iso2022jp:         japanese.ISO2022JP,
	shiftJIS:          japanese.ShiftJIS,
	euckr:             korean.EUCKR,
	replacement:       encoding.Replacement,
	utf16be:           unicode.UTF16(unicode.BigEndian, unicode.IgnoreBOM),
	utf16le:           unicode.UTF16(unicode.LittleEndian, unicode.IgnoreBOM),
	xUserDefined:      charmap.XUserDefined,
}
