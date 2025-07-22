// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run gen.go

// Package identifier defines the contract between implementations of Encoding
// and Index by defining identifiers that uniquely identify standardized coded
// character sets (CCS) and character encoding schemes (CES), which we will
// together refer to as encodings, for which Encoding implementations provide
// converters to and from UTF-8. This package is typically only of concern to
// implementers of Indexes and Encodings.
//
// One part of the identifier is the MIB code, which is defined by IANA and
// uniquely identifies a CCS or CES. Each code is associated with data that
// references authorities, official documentation as well as aliases and MIME
// names.
//
// Not all CESs are covered by the IANA registry. The "other" string that is
// returned by ID can be used to identify other character sets or versions of
// existing ones.
//
// It is recommended that each package that provides a set of Encodings provide
// the All and Common variables to reference all supported encodings and
// commonly used subset. This allows Index implementations to include all
// available encodings without explicitly referencing or knowing about them.
package identifier

// Note: this package is internal, but could be made public if there is a need
// for writing third-party Indexes and Encodings.

// References:
// - http://source.icu-project.org/repos/icu/icu/trunk/source/data/mappings/convrtrs.txt
// - http://www.iana.org/assignments/character-sets/character-sets.xhtml
// - http://www.iana.org/assignments/ianacharset-mib/ianacharset-mib
// - http://www.ietf.org/rfc/rfc2978.txt
// - https://www.unicode.org/reports/tr22/
// - http://www.w3.org/TR/encoding/
// - https://encoding.spec.whatwg.org/
// - https://encoding.spec.whatwg.org/encodings.json
// - https://tools.ietf.org/html/rfc6657#section-5

// Interface can be implemented by Encodings to define the CCS or CES for which
// it implements conversions.
type Interface interface {
	// ID returns an encoding identifier. Exactly one of the mib and other
	// values should be non-zero.
	//
	// In the usual case it is only necessary to indicate the MIB code. The
	// other string can be used to specify encodings for which there is no MIB,
	// such as "x-mac-dingbat".
	//
	// The other string may only contain the characters a-z, A-Z, 0-9, - and _.
	ID() (mib MIB, other string)

	// NOTE: the restrictions on the encoding are to allow extending the syntax
	// with additional information such as versions, vendors and other variants.
}

// A MIB identifies an encoding. It is derived from the IANA MIB codes and adds
// some identifiers for some encodings that are not covered by the IANA
// standard.
//
// See http://www.iana.org/assignments/ianacharset-mib.
type MIB uint16

// These additional MIB types are not defined in IANA. They are added because
// they are common and defined within the text repo.
const (
	// Unofficial marks the start of encodings not registered by IANA.
	Unofficial MIB = 10000 + iota

	// Replacement is the WhatWG replacement encoding.
	Replacement

	// XUserDefined is the code for x-user-defined.
	XUserDefined

	// MacintoshCyrillic is the code for x-mac-cyrillic.
	MacintoshCyrillic
)
