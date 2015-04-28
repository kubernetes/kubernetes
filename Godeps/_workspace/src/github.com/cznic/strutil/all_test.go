// Copyright (c) 2014 The sortutil Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strutil

import (
	"bytes"
	"github.com/cznic/mathutil"
	"math"
	"strings"
	"testing"
)

func TestBase64(t *testing.T) {
	const max = 768
	r, err := mathutil.NewFC32(math.MinInt32, math.MaxInt32, true)
	if err != nil {
		t.Fatal(err)
	}

	bin := []byte{}
	for i := 0; i < max; i++ {
		bin = append(bin, byte(r.Next()))
		cmp, err := Base64Decode(Base64Encode(bin))
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.Equal(bin, cmp) {
			t.Fatalf("a: % x\nb: % x", bin, cmp)
		}
	}
}

func TestBase32Ext(t *testing.T) {
	const max = 640
	r, err := mathutil.NewFC32(math.MinInt32, math.MaxInt32, true)
	if err != nil {
		t.Fatal(err)
	}

	bin := []byte{}
	for i := 0; i < max; i++ {
		bin = append(bin, byte(r.Next()))
		cmp, err := Base32ExtDecode(Base32ExtEncode(bin))
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.Equal(bin, cmp) {
			t.Fatalf("a: % x\nb: % x", bin, cmp)
		}
	}
}

func TestFields(t *testing.T) {
	p := []string{"", "\\", "|", "0", "1", "2"}
	one := func(n int) string {
		s := ""
		for i := 0; i < 3; i++ {
			s += p[n%len(p)]
			n /= len(p)
		}
		return s
	}
	max := len(p) * len(p) * len(p)
	var a [3]string
	for x := 0; x < max; x++ {
		a[0] = one(x)
		for x := 0; x < max; x++ {
			a[1] = one(x)
			for x := 0; x < len(p)*len(p); x++ {
				a[2] = one(x)
				enc := JoinFields(a[:], "|")
				dec := SplitFields(enc, "|")
				if g, e := strings.Join(dec, ","), strings.Join(a[:], ","); g != e {
					t.Fatal(g, e)
				}
			}
		}
	}
}
