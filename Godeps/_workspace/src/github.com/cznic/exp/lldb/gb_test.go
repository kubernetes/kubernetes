// Copyright 2014 The lldb Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Utilities to encode/decode and collate Go predeclared scalar types.  The
// encoding format reused the one used by the "encoding/gob" package.

package lldb

import (
	"bytes"
	"math"
	"testing"
)

const s256 = "" +
	"0123456789abcdef" +
	"0123456789abcdef" +
	"0123456789abcdef" +
	"0123456789abcdef" +
	"0123456789abcdef" +
	"0123456789abcdef" +
	"0123456789abcdef" +
	"0123456789abcdef" +
	"0123456789abcdef" +
	"0123456789abcdef" +
	"0123456789abcdef" +
	"0123456789abcdef" +
	"0123456789abcdef" +
	"0123456789abcdef" +
	"0123456789abcdef" +
	"0123456789abcdef"

func TestEncodeDecodeScalars(t *testing.T) {
	table := []struct{ v, exp interface{} }{
		{nil, "00"},
		{false, "01"},
		{true, "02"},
		{math.Float64frombits(0), []byte{gbFloat0}},
		{17., []byte{gbFloat2, 0x31, 0x40}},
		{math.Float64frombits(0x4031320000000000), []byte{gbFloat3, 0x32, 0x31, 0x40}},
		{math.Float64frombits(0x4031323300000000), []byte{gbFloat4, 0x33, 0x32, 0x31, 0x40}},
		{math.Float64frombits(0x4031323334000000), []byte{gbFloat5, 0x34, 0x33, 0x32, 0x31, 0x40}},
		{math.Float64frombits(0x4031323334350000), []byte{gbFloat6, 0x35, 0x34, 0x33, 0x32, 0x31, 0x40}},
		{math.Float64frombits(0x4031323334353600), []byte{gbFloat7, 0x36, 0x35, 0x34, 0x33, 0x32, 0x31, 0x40}},
		{math.Float64frombits(0x4031323334353637), []byte{gbFloat8, 0x37, 0x36, 0x35, 0x34, 0x33, 0x32, 0x31, 0x40}},
		{0 + 0i, []byte{gbComplex0, gbComplex0}},
		{17 + 17i, []byte{gbComplex2, 0x31, 0x40, gbComplex2, 0x31, 0x40}},
		{complex(math.Float64frombits(0x4041420000000000), math.Float64frombits(0x4031320000000000)), []byte{gbComplex3, 0x42, 0x41, 0x40, gbComplex3, 0x32, 0x31, 0x40}},
		{complex(math.Float64frombits(0x4041424300000000), math.Float64frombits(0x4031323300000000)), []byte{gbComplex4, 0x43, 0x42, 0x41, 0x40, gbComplex4, 0x33, 0x32, 0x31, 0x40}},
		{complex(math.Float64frombits(0x4041424344000000), math.Float64frombits(0x4031323334000000)), []byte{gbComplex5, 0x44, 0x43, 0x42, 0x41, 0x40, gbComplex5, 0x34, 0x33, 0x32, 0x31, 0x40}},
		{complex(math.Float64frombits(0x4041424344450000), math.Float64frombits(0x4031323334350000)), []byte{gbComplex6, 0x45, 0x44, 0x43, 0x42, 0x41, 0x40, gbComplex6, 0x35, 0x34, 0x33, 0x32, 0x31, 0x40}},
		{complex(math.Float64frombits(0x4041424344454600), math.Float64frombits(0x4031323334353600)), []byte{gbComplex7, 0x46, 0x45, 0x44, 0x43, 0x42, 0x41, 0x40, gbComplex7, 0x36, 0x35, 0x34, 0x33, 0x32, 0x31, 0x40}},
		{complex(math.Float64frombits(0x4041424344454647), math.Float64frombits(0x4031323334353637)), []byte{gbComplex8, 0x47, 0x46, 0x45, 0x44, 0x43, 0x42, 0x41, 0x40, gbComplex8, 0x37, 0x36, 0x35, 0x34, 0x33, 0x32, 0x31, 0x40}},
		{[]byte(""), []byte{gbBytes00}},
		{[]byte("f"), []byte{gbBytes01, 'f'}},
		{[]byte("fo"), []byte{gbBytes02, 'f', 'o'}},
		{[]byte("0123456789abcdefx"), []byte{gbBytes17, '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'x'}},
		{[]byte("0123456789abcdefxy"), []byte{gbBytes1, 18, '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'x', 'y'}},
		{[]byte(s256[:255]), append([]byte{gbBytes1, 0xff}, []byte(s256[:255])...)},
		{[]byte(s256), append([]byte{gbBytes2, 0x00, 0xff}, []byte(s256)...)},
		{"", []byte{gbString00}},
		{"f", []byte{gbString01, 'f'}},
		{"fo", []byte{gbString02, 'f', 'o'}},
		{"0123456789abcdefx", []byte{gbString17, '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'x'}},
		{"0123456789abcdefxy", []byte{gbString1, 18, '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'x', 'y'}},
		{s256[:255], append([]byte{gbString1, 0xff}, []byte(s256[:255])...)},
		{s256, append([]byte{gbString2, 0x01, 0x00}, []byte(s256)...)},
		{uint64(0xff), []byte{gbUintP1, 255}},
		{uint64(0xffff), []byte{gbUintP2, 255, 255}},
		{uint64(0xffffff), []byte{gbUintP3, 255, 255, 255}},
		{uint64(0xffffffff), []byte{gbUintP4, 255, 255, 255, 255}},
		{uint64(0xffffffffff), []byte{gbUintP5, 255, 255, 255, 255, 255}},
		{uint64(0xffffffffffff), []byte{gbUintP6, 255, 255, 255, 255, 255, 255}},
		{uint64(0xffffffffffffff), []byte{gbUintP7, 255, 255, 255, 255, 255, 255, 255}},
		{uint64(0xffffffffffffffff), []byte{gbUintP8, 255, 255, 255, 255, 255, 255, 255, 255}},
		{int64(math.MinInt64), []byte{gbIntM8, 128, 0, 0, 0, 0, 0, 0, 0}},
		{-int64(0x100000000000000), []byte{gbIntM7, 0, 0, 0, 0, 0, 0, 0}},
		{-int64(0x1000000000000), []byte{gbIntM6, 0, 0, 0, 0, 0, 0}},
		{-int64(0x10000000000), []byte{gbIntM5, 0, 0, 0, 0, 0}},
		{-int64(0x100000000), []byte{gbIntM4, 0, 0, 0, 0}},
		{-int64(0x1000000), []byte{gbIntM3, 0, 0, 0}},
		{-int64(0x10000), []byte{gbIntM2, 0, 0}},
		{-int64(0x100), []byte{gbIntM1, 0}},
		{-int64(0xff), []byte{gbIntM1, 1}},
		{-int64(1), []byte{gbIntM1, 255}},
		{int64(gbIntMax + 1), []byte{gbIntP1, gbIntMax + 1}},
		{int64(0xff), []byte{gbIntP1, 255}},
		{int64(0xffff), []byte{gbIntP2, 255, 255}},
		{int64(0xffffff), []byte{gbIntP3, 255, 255, 255}},
		{int64(0xffffffff), []byte{gbIntP4, 255, 255, 255, 255}},
		{int64(0xffffffffff), []byte{gbIntP5, 255, 255, 255, 255, 255}},
		{int64(0xffffffffffff), []byte{gbIntP6, 255, 255, 255, 255, 255, 255}},
		{int64(0xffffffffffffff), []byte{gbIntP7, 255, 255, 255, 255, 255, 255, 255}},
		{int64(0x7fffffffffffffff), []byte{gbIntP8, 127, 255, 255, 255, 255, 255, 255, 255}},
		{int64(0), []byte{0 + gbInt0}},
		{int64(1), []byte{1 + gbInt0}},
		{int64(2), []byte{2 + gbInt0}},
		{int64(gbIntMax - 2), "fd"},
		{int64(gbIntMax - 1), "fe"},
		{int64(gbIntMax), "ff"},
	}

	for i, v := range table {
		g, err := EncodeScalars(v.v)
		if err != nil {
			t.Fatal(i, err)
		}

		var e []byte
		switch x := v.exp.(type) {
		case string:
			e = s2b(x)
		case []byte:
			e = x
		}

		if !bytes.Equal(g, e) {
			t.Fatalf("%d %v\n|% 02x|\n|% 02x|", i, v.v, g, e)
		}

		t.Logf("%#v |% 02x|", v.v, g)

		dec, err := DecodeScalars(g)
		if err != nil {
			t.Fatal(err)
		}

		if g, e := len(dec), 1; g != e {
			t.Fatalf("%d %d %#v", g, e, dec)
		}

		if g, ok := dec[0].([]byte); ok {
			if e := v.v.([]byte); !bytes.Equal(g, e) {
				t.Fatal(g, e)
			}

			continue
		}

		if g, e := dec[0], v.v; g != e {
			t.Fatal(g, e)
		}
	}
}

func strcmp(a, b string) (r int) {
	if a < b {
		return -1
	}

	if a == b {
		return 0
	}

	return 1
}

func TestCollateScalars(t *testing.T) {
	// all cases must return -1
	table := []struct{ x, y []interface{} }{
		{[]interface{}{}, []interface{}{1}},
		{[]interface{}{1}, []interface{}{2}},
		{[]interface{}{1, 2}, []interface{}{2, 3}},

		{[]interface{}{nil}, []interface{}{nil, true}},
		{[]interface{}{nil}, []interface{}{false}},
		{[]interface{}{nil}, []interface{}{nil, 1}},
		{[]interface{}{nil}, []interface{}{1}},
		{[]interface{}{nil}, []interface{}{nil, uint(1)}},
		{[]interface{}{nil}, []interface{}{uint(1)}},
		{[]interface{}{nil}, []interface{}{nil, 3.14}},
		{[]interface{}{nil}, []interface{}{3.14}},
		{[]interface{}{nil}, []interface{}{nil, 3.14 + 1i}},
		{[]interface{}{nil}, []interface{}{3.14 + 1i}},
		{[]interface{}{nil}, []interface{}{nil, []byte("foo")}},
		{[]interface{}{nil}, []interface{}{[]byte("foo")}},
		{[]interface{}{nil}, []interface{}{nil, "foo"}},
		{[]interface{}{nil}, []interface{}{"foo"}},

		{[]interface{}{false}, []interface{}{false, false}},
		{[]interface{}{false}, []interface{}{false, true}},
		{[]interface{}{false}, []interface{}{true}},
		{[]interface{}{false}, []interface{}{false, 1}},
		{[]interface{}{false}, []interface{}{1}},
		{[]interface{}{false}, []interface{}{false, uint(1)}},
		{[]interface{}{false}, []interface{}{uint(1)}},
		{[]interface{}{false}, []interface{}{false, 1.5}},
		{[]interface{}{false}, []interface{}{1.5}},
		{[]interface{}{false}, []interface{}{false, 1.5 + 3i}},
		{[]interface{}{false}, []interface{}{1.5 + 3i}},
		{[]interface{}{false}, []interface{}{false, []byte("foo")}},
		{[]interface{}{false}, []interface{}{[]byte("foo")}},
		{[]interface{}{false}, []interface{}{false, "foo"}},
		{[]interface{}{false}, []interface{}{"foo"}},

		{[]interface{}{1}, []interface{}{1, 2}},
		{[]interface{}{1}, []interface{}{1, 1}},
		{[]interface{}{1}, []interface{}{1, uint(2)}},
		{[]interface{}{1}, []interface{}{uint(2)}},
		{[]interface{}{1}, []interface{}{1, 1.1}},
		{[]interface{}{1}, []interface{}{1.1}},
		{[]interface{}{1}, []interface{}{1, 1.1 + 2i}},
		{[]interface{}{1}, []interface{}{1.1 + 2i}},
		{[]interface{}{1}, []interface{}{1, []byte("foo")}},
		{[]interface{}{1}, []interface{}{[]byte("foo")}},
		{[]interface{}{1}, []interface{}{1, "foo"}},
		{[]interface{}{1}, []interface{}{"foo"}},

		{[]interface{}{uint(1)}, []interface{}{uint(1), uint(1)}},
		{[]interface{}{uint(1)}, []interface{}{uint(2)}},
		{[]interface{}{uint(1)}, []interface{}{uint(1), 2.}},
		{[]interface{}{uint(1)}, []interface{}{2.}},
		{[]interface{}{uint(1)}, []interface{}{uint(1), 2. + 0i}},
		{[]interface{}{uint(1)}, []interface{}{2. + 0i}},
		{[]interface{}{uint(1)}, []interface{}{uint(1), []byte("foo")}},
		{[]interface{}{uint(1)}, []interface{}{[]byte("foo")}},
		{[]interface{}{uint(1)}, []interface{}{uint(1), "foo"}},
		{[]interface{}{uint(1)}, []interface{}{"foo"}},

		{[]interface{}{1.}, []interface{}{1., 1}},
		{[]interface{}{1.}, []interface{}{2}},
		{[]interface{}{1.}, []interface{}{1., uint(1)}},
		{[]interface{}{1.}, []interface{}{uint(2)}},
		{[]interface{}{1.}, []interface{}{1., 1.}},
		{[]interface{}{1.}, []interface{}{1.1}},
		{[]interface{}{1.}, []interface{}{1., []byte("foo")}},
		{[]interface{}{1.}, []interface{}{[]byte("foo")}},
		{[]interface{}{1.}, []interface{}{1., "foo"}},
		{[]interface{}{1.}, []interface{}{"foo"}},

		{[]interface{}{1 + 2i}, []interface{}{1 + 2i, 1}},
		{[]interface{}{1 + 2i}, []interface{}{2}},
		{[]interface{}{1 + 2i}, []interface{}{1 + 2i, uint(1)}},
		{[]interface{}{1 + 2i}, []interface{}{uint(2)}},
		{[]interface{}{1 + 2i}, []interface{}{1 + 2i, 1.1}},
		{[]interface{}{1 + 2i}, []interface{}{1.1}},
		{[]interface{}{1 + 2i}, []interface{}{1 + 2i, []byte("foo")}},
		{[]interface{}{1 + 2i}, []interface{}{[]byte("foo")}},
		{[]interface{}{1 + 2i}, []interface{}{1 + 2i, "foo"}},
		{[]interface{}{1 + 2i}, []interface{}{"foo"}},

		{[]interface{}{[]byte("bar")}, []interface{}{[]byte("bar"), []byte("bar")}},
		{[]interface{}{[]byte("bar")}, []interface{}{[]byte("foo")}},
		{[]interface{}{[]byte("bar")}, []interface{}{[]byte("c")}},
		{[]interface{}{[]byte("bar")}, []interface{}{[]byte("bas")}},
		{[]interface{}{[]byte("bar")}, []interface{}{[]byte("bara")}},

		{[]interface{}{[]byte("bar")}, []interface{}{"bap"}},
		{[]interface{}{[]byte("bar")}, []interface{}{"bar"}},
		{[]interface{}{[]byte("bar")}, []interface{}{"bas"}},

		{[]interface{}{"bar"}, []interface{}{"bar", "bar"}},
		{[]interface{}{"bar"}, []interface{}{"foo"}},
		{[]interface{}{"bar"}, []interface{}{"c"}},
		{[]interface{}{"bar"}, []interface{}{"bas"}},
		{[]interface{}{"bar"}, []interface{}{"bara"}},

		{[]interface{}{1 + 2i}, []interface{}{1 + 3i}},
		{[]interface{}{int64(math.MaxInt64)}, []interface{}{uint64(math.MaxInt64 + 1)}},
		{[]interface{}{int8(1)}, []interface{}{int16(2)}},
		{[]interface{}{int32(1)}, []interface{}{uint8(2)}},
		{[]interface{}{uint16(1)}, []interface{}{uint32(2)}},
		{[]interface{}{float32(1)}, []interface{}{complex(float32(2), 0)}},

		// resolved bugs
		{[]interface{}{"Customer"}, []interface{}{"Date"}},
		{[]interface{}{"Customer"}, []interface{}{"Items", 1, "Quantity"}},
	}

	more := []interface{}{42, nil, 1, uint(2), 3.0, 4 + 5i, "..."}

	collate := func(x, y []interface{}, strCollate func(string, string) int) (r int) {
		var err error
		r, err = Collate(x, y, strCollate)
		if err != nil {
			t.Fatal(err)
		}

		return
	}

	for _, scf := range []func(string, string) int{nil, strcmp} {
		for _, prefix := range more {
			for i, test := range table {
				var x, y []interface{}
				if prefix != 42 {
					x = append(x, prefix)
					y = append(y, prefix)
				}
				x = append(x, test.x...)
				y = append(y, test.y...)

				// cmp(x, y) == -1
				if g, e := collate(x, y, scf), -1; g != e {
					t.Fatal(i, g, e, x, y)
				}

				// cmp(y, x) == 1
				if g, e := collate(y, x, scf), 1; g != e {
					t.Fatal(i, g, e, y, x)
				}

				src := x
				for ix := len(src) - 1; ix > 0; ix-- {
					if g, e := collate(src[:ix], src[:ix], scf), 0; g != e {
						t.Fatal(ix, g, e)
					}

					if g, e := collate(src[:ix], src, scf), -1; g != e {
						t.Fatal(ix, g, e)
					}

				}

				src = y
				for ix := len(src) - 1; ix > 0; ix-- {
					if g, e := collate(src[:ix], src[:ix], scf), 0; g != e {
						t.Fatal(ix, g, e)
					}

					if g, e := collate(src[:ix], src, scf), -1; g != e {
						t.Fatal(ix, g, e)
					}

				}
			}
		}
	}
}

func TestEncodingBug(t *testing.T) {
	bits := uint64(0)
	for i := 0; i <= 64; i++ {
		encoded, err := EncodeScalars(math.Float64frombits(bits))
		if err != nil {
			t.Fatal(err)
		}

		t.Logf("bits %016x, enc |% x|", bits, encoded)
		decoded, err := DecodeScalars(encoded)
		if err != nil {
			t.Fatal(err)
		}

		if g, e := len(decoded), 1; g != e {
			t.Fatal(g, e)
		}

		f, ok := decoded[0].(float64)
		if !ok {
			t.Fatal(err)
		}

		if g, e := math.Float64bits(f), bits; g != e {
			t.Fatal(err)
		}

		t.Log(f)

		bits >>= 1
		bits |= 1 << 63
	}
}
