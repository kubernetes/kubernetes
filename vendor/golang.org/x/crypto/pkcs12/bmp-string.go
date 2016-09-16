// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkcs12

import (
	"errors"
	"unicode/utf16"
)

// bmpString returns s encoded in UCS-2 with a zero terminator.
func bmpString(s string) ([]byte, error) {
	// References:
	// https://tools.ietf.org/html/rfc7292#appendix-B.1
	// http://en.wikipedia.org/wiki/Plane_(Unicode)#Basic_Multilingual_Plane
	//  - non-BMP characters are encoded in UTF 16 by using a surrogate pair of 16-bit codes
	//	  EncodeRune returns 0xfffd if the rune does not need special encoding
	//  - the above RFC provides the info that BMPStrings are NULL terminated.

	ret := make([]byte, 0, 2*len(s)+2)

	for _, r := range s {
		if t, _ := utf16.EncodeRune(r); t != 0xfffd {
			return nil, errors.New("pkcs12: string contains characters that cannot be encoded in UCS-2")
		}
		ret = append(ret, byte(r/256), byte(r%256))
	}

	return append(ret, 0, 0), nil
}

func decodeBMPString(bmpString []byte) (string, error) {
	if len(bmpString)%2 != 0 {
		return "", errors.New("pkcs12: odd-length BMP string")
	}

	// strip terminator if present
	if l := len(bmpString); l >= 2 && bmpString[l-1] == 0 && bmpString[l-2] == 0 {
		bmpString = bmpString[:l-2]
	}

	s := make([]uint16, 0, len(bmpString)/2)
	for len(bmpString) > 0 {
		s = append(s, uint16(bmpString[0])<<8+uint16(bmpString[1]))
		bmpString = bmpString[2:]
	}

	return string(utf16.Decode(s)), nil
}
