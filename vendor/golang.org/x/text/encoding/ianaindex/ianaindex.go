// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run gen.go

// Package ianaindex maps names to Encodings as specified by the IANA registry.
// This includes both the MIME and IANA names.
//
// See http://www.iana.org/assignments/character-sets/character-sets.xhtml for
// more details.
package ianaindex

import (
	"errors"
	"sort"
	"strings"

	"golang.org/x/text/encoding"
	"golang.org/x/text/encoding/charmap"
	"golang.org/x/text/encoding/internal/identifier"
	"golang.org/x/text/encoding/japanese"
	"golang.org/x/text/encoding/korean"
	"golang.org/x/text/encoding/simplifiedchinese"
	"golang.org/x/text/encoding/traditionalchinese"
	"golang.org/x/text/encoding/unicode"
)

// TODO: remove the "Status... incomplete" in the package doc comment.
// TODO: allow users to specify their own aliases?
// TODO: allow users to specify their own indexes?
// TODO: allow canonicalizing names

// NOTE: only use these top-level variables if we can get the linker to drop
// the indexes when they are not used. Make them a function or perhaps only
// support MIME otherwise.

var (
	// MIME is an index to map MIME names.
	MIME *Index = mime

	// IANA is an index that supports all names and aliases using IANA names as
	// the canonical identifier.
	IANA *Index = iana

	// MIB is an index that associates the MIB display name with an Encoding.
	MIB *Index = mib

	mime = &Index{mimeName, ianaToMIB, ianaAliases, encodings[:]}
	iana = &Index{ianaName, ianaToMIB, ianaAliases, encodings[:]}
	mib  = &Index{mibName, ianaToMIB, ianaAliases, encodings[:]}
)

// Index maps names registered by IANA to Encodings.
// Currently different Indexes only differ in the names they return for
// encodings. In the future they may also differ in supported aliases.
type Index struct {
	names func(i int) string
	toMIB []identifier.MIB // Sorted slice of supported MIBs
	alias map[string]int
	enc   []encoding.Encoding
}

var (
	errInvalidName = errors.New("ianaindex: invalid encoding name")
	errUnknown     = errors.New("ianaindex: unknown Encoding")
	errUnsupported = errors.New("ianaindex: unsupported Encoding")
)

// Encoding returns an Encoding for IANA-registered names. Matching is
// case-insensitive.
func (x *Index) Encoding(name string) (encoding.Encoding, error) {
	name = strings.TrimSpace(name)
	// First try without lowercasing (possibly creating an allocation).
	i, ok := x.alias[name]
	if !ok {
		i, ok = x.alias[strings.ToLower(name)]
		if !ok {
			return nil, errInvalidName
		}
	}
	return x.enc[i], nil
}

// Name reports the canonical name of the given Encoding. It will return an
// error if the e is not associated with a known encoding scheme.
func (x *Index) Name(e encoding.Encoding) (string, error) {
	id, ok := e.(identifier.Interface)
	if !ok {
		return "", errUnknown
	}
	mib, _ := id.ID()
	if mib == 0 {
		return "", errUnknown
	}
	v := findMIB(x.toMIB, mib)
	if v == -1 {
		return "", errUnsupported
	}
	return x.names(v), nil
}

// TODO: the coverage of this index is rather spotty. Allowing users to set
// encodings would allow:
// - users to increase coverage
// - allow a partially loaded set of encodings in case the user doesn't need to
//   them all.
// - write an OS-specific wrapper for supported encodings and set them.
// The exact definition of Set depends a bit on if and how we want to let users
// write their own Encoding implementations. Also, it is not possible yet to
// only partially load the encodings without doing some refactoring. Until this
// is solved, we might as well not support Set.
// // Set sets the e to be used for the encoding scheme identified by name. Only
// // canonical names may be used. An empty name assigns e to its internally
// // associated encoding scheme.
// func (x *Index) Set(name string, e encoding.Encoding) error {
// 	panic("TODO: implement")
// }

func findMIB(x []identifier.MIB, mib identifier.MIB) int {
	i := sort.Search(len(x), func(i int) bool { return x[i] >= mib })
	if i < len(x) && x[i] == mib {
		return i
	}
	return -1
}

const maxMIMENameLen = '0' - 1 // officially 40, but we leave some buffer.

func mimeName(x int) string {
	n := ianaNames[x]
	// See gen.go for a description of the encoding.
	if n[0] <= maxMIMENameLen {
		return n[1:n[0]]
	}
	return n
}

func ianaName(x int) string {
	n := ianaNames[x]
	// See gen.go for a description of the encoding.
	if n[0] <= maxMIMENameLen {
		return n[n[0]:]
	}
	return n
}

func mibName(x int) string {
	return mibNames[x]
}

var encodings = [numIANA]encoding.Encoding{
	enc106:  unicode.UTF8,
	enc1015: unicode.UTF16(unicode.BigEndian, unicode.UseBOM),
	enc1013: unicode.UTF16(unicode.BigEndian, unicode.IgnoreBOM),
	enc1014: unicode.UTF16(unicode.LittleEndian, unicode.IgnoreBOM),
	enc2028: charmap.CodePage037,
	enc2011: charmap.CodePage437,
	enc2009: charmap.CodePage850,
	enc2010: charmap.CodePage852,
	enc2046: charmap.CodePage855,
	enc2089: charmap.CodePage858,
	enc2048: charmap.CodePage860,
	enc2013: charmap.CodePage862,
	enc2050: charmap.CodePage863,
	enc2052: charmap.CodePage865,
	enc2086: charmap.CodePage866,
	enc2102: charmap.CodePage1047,
	enc2091: charmap.CodePage1140,
	enc4:    charmap.ISO8859_1,
	enc5:    charmap.ISO8859_2,
	enc6:    charmap.ISO8859_3,
	enc7:    charmap.ISO8859_4,
	enc8:    charmap.ISO8859_5,
	enc9:    charmap.ISO8859_6,
	enc81:   charmap.ISO8859_6E,
	enc82:   charmap.ISO8859_6I,
	enc10:   charmap.ISO8859_7,
	enc11:   charmap.ISO8859_8,
	enc84:   charmap.ISO8859_8E,
	enc85:   charmap.ISO8859_8I,
	enc12:   charmap.ISO8859_9,
	enc13:   charmap.ISO8859_10,
	enc109:  charmap.ISO8859_13,
	enc110:  charmap.ISO8859_14,
	enc111:  charmap.ISO8859_15,
	enc112:  charmap.ISO8859_16,
	enc2084: charmap.KOI8R,
	enc2088: charmap.KOI8U,
	enc2027: charmap.Macintosh,
	enc2109: charmap.Windows874,
	enc2250: charmap.Windows1250,
	enc2251: charmap.Windows1251,
	enc2252: charmap.Windows1252,
	enc2253: charmap.Windows1253,
	enc2254: charmap.Windows1254,
	enc2255: charmap.Windows1255,
	enc2256: charmap.Windows1256,
	enc2257: charmap.Windows1257,
	enc2258: charmap.Windows1258,
	enc18:   japanese.EUCJP,
	enc39:   japanese.ISO2022JP,
	enc17:   japanese.ShiftJIS,
	enc38:   korean.EUCKR,
	enc114:  simplifiedchinese.GB18030,
	enc113:  simplifiedchinese.GBK,
	enc2085: simplifiedchinese.HZGB2312,
	enc2026: traditionalchinese.Big5,
}
