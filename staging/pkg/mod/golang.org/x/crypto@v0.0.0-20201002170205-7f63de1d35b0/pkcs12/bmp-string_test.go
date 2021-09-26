// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkcs12

import (
	"bytes"
	"encoding/hex"
	"testing"
)

var bmpStringTests = []struct {
	in          string
	expectedHex string
	shouldFail  bool
}{
	{"", "0000", false},
	// Example from https://tools.ietf.org/html/rfc7292#appendix-B.
	{"Beavis", "0042006500610076006900730000", false},
	// Some characters from the "Letterlike Symbols Unicode block".
	{"\u2115 - Double-struck N", "21150020002d00200044006f00750062006c0065002d00730074007200750063006b0020004e0000", false},
	// any character outside the BMP should trigger an error.
	{"\U0001f000 East wind (Mahjong)", "", true},
}

func TestBMPString(t *testing.T) {
	for i, test := range bmpStringTests {
		expected, err := hex.DecodeString(test.expectedHex)
		if err != nil {
			t.Fatalf("#%d: failed to decode expectation", i)
		}

		out, err := bmpString(test.in)
		if err == nil && test.shouldFail {
			t.Errorf("#%d: expected to fail, but produced %x", i, out)
			continue
		}

		if err != nil && !test.shouldFail {
			t.Errorf("#%d: failed unexpectedly: %s", i, err)
			continue
		}

		if !test.shouldFail {
			if !bytes.Equal(out, expected) {
				t.Errorf("#%d: expected %s, got %x", i, test.expectedHex, out)
				continue
			}

			roundTrip, err := decodeBMPString(out)
			if err != nil {
				t.Errorf("#%d: decoding output gave an error: %s", i, err)
				continue
			}

			if roundTrip != test.in {
				t.Errorf("#%d: decoding output resulted in %q, but it should have been %q", i, roundTrip, test.in)
				continue
			}
		}
	}
}
