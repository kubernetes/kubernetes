// Copyright (c) 2014 The mersenne Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mersenne

import (
	"math"
	"math/big"
	"math/rand"
	"runtime"
	"sync"
	"testing"

	"github.com/cznic/mathutil"
)

func r32() *mathutil.FC32 {
	r, err := mathutil.NewFC32(math.MinInt32, math.MaxInt32, true)
	if err != nil {
		panic(err)
	}

	return r
}

var (
	r64lo = big.NewInt(math.MinInt64)
	r64hi = big.NewInt(math.MaxInt64)
)

func r64() *mathutil.FCBig {
	r, err := mathutil.NewFCBig(r64lo, r64hi, true)
	if err != nil {
		panic(err)
	}

	return r
}

func TestNew(t *testing.T) {
	const N = 1e4
	data := []struct{ n, m uint32 }{
		{0, 0},
		{1, 1},
		{2, 3},
		{3, 7},
		{4, 15},
		{5, 31},
		{6, 63},
		{7, 127},
		{8, 255},
		{9, 511},
		{10, 1023},
		{11, 2047},
		{12, 4095},
		{13, 8191},
		{14, 16383},
		{15, 32767},
		{16, 65535},
		{17, 131071},
	}

	e := big.NewInt(0)
	for _, v := range data {
		g := New(v.n)
		e.SetInt64(int64(v.m))
		if g.Cmp(e) != 0 {
			t.Errorf("%d: got %s, exp %s", v.n, g, e)
		}
	}

	r := r32()
	for i := 0; i < N; i++ {
		exp := uint32(r.Next()) % 1e6
		g := New(exp)
		b0 := g.BitLen()
		g.Add(g, _1)
		b1 := g.BitLen()
		if b1-b0 != 1 {
			t.Fatal(i, exp, b1, b0)
		}
	}
}

func benchmarkNew(b *testing.B, max uint32) {
	const N = 1 << 16
	b.StopTimer()
	a := make([]uint32, N)
	r := r32()
	for i := range a {
		a[i] = uint32(r.Next()) % max
	}
	runtime.GC()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		New(a[i&(N-1)])
	}
}

func BenchmarkNew_1e1(b *testing.B) {
	benchmarkNew(b, 1e1)
}

func BenchmarkNew_1e2(b *testing.B) {
	benchmarkNew(b, 1e2)
}

func BenchmarkNew_1e3(b *testing.B) {
	benchmarkNew(b, 1e3)
}

func BenchmarkNew_1e4(b *testing.B) {
	benchmarkNew(b, 1e4)
}

func BenchmarkNew_1e5(b *testing.B) {
	benchmarkNew(b, 1e5)
}

func BenchmarkNew_1e6(b *testing.B) {
	benchmarkNew(b, 1e6)
}

func BenchmarkNew_1e7(b *testing.B) {
	benchmarkNew(b, 1e7)
}

func BenchmarkNew_1e8(b *testing.B) {
	benchmarkNew(b, 1e8)
}

func TestHasFactorUint32(t *testing.T) {
	data := []struct {
		d, e uint32
		r    bool
	}{
		{0, 42, false},
		{1, 24, true},
		{2, 22, false},
		{3, 2, true},
		{3, 3, false},
		{3, 4, true},
		{3, 5, false},
		{3, 6, true},
		{5, 4, true},
		{5, 5, false},
		{5, 6, false},
		{5, 7, false},
		{5, 8, true},
		{5, 9, false},
		{5, 10, false},
		{5, 11, false},
		{5, 12, true},
		{7, 3, true},
		{7, 6, true},
		{7, 9, true},
		{9, 6, true},
		{9, 12, true},
		{9, 18, true},
		{11, 10, true},
		{23, 11, true},
		{89, 11, true},
		{47, 23, true},
		{193707721, 67, true},
		{13007, 929, true},
		{264248689, 500471, true},
		{112027889, 1000249, true},
		{252079759, 2000633, true},
		{222054983, 3000743, true},
		{1920355681, 4000741, true},
		{330036367, 5000551, true},
		{1020081431, 6000479, true},
		{840074281, 7000619, true},
		{624031279, 8000401, true},
		{378031207, 9000743, true},
		{380036519, 10000961, true},
		{40001447, 20000723, true},
	}

	for _, v := range data {
		if g, e := HasFactorUint32(v.d, v.e), v.r; g != e {
			t.Errorf("d %d e %d: got %t, exp %t", v.d, v.e, g, e)
		}
	}
}

func TestHasFactorUint64(t *testing.T) {
	data := []struct {
		d uint64
		e uint32
		r bool
	}{
		{0, 42, false},
		{1, 24, true},
		{2, 22, false},
		{3, 2, true},
		{3, 3, false},
		{3, 4, true},
		{3, 5, false},
		{3, 6, true},
		{5, 4, true},
		{5, 5, false},
		{5, 6, false},
		{5, 7, false},
		{5, 8, true},
		{5, 9, false},
		{5, 10, false},
		{5, 11, false},
		{5, 12, true},
		{7, 3, true},
		{7, 6, true},
		{7, 9, true},
		{9, 6, true},
		{9, 12, true},
		{9, 18, true},
		{11, 10, true},
		{23, 11, true},
		{89, 11, true},
		{47, 23, true},
		{193707721, 67, true},
		{13007, 929, true},
		{264248689, 500471, true},
		{112027889, 1000249, true},
		{252079759, 2000633, true},
		{222054983, 3000743, true},
		{1920355681, 4000741, true},
		{330036367, 5000551, true},
		{1020081431, 6000479, true},
		{840074281, 7000619, true},
		{624031279, 8000401, true},
		{378031207, 9000743, true},
		{380036519, 10000961, true},
		{40001447, 20000723, true},
		{1872347344039, 1000099, true},
	}

	for _, v := range data {
		if g, e := HasFactorUint64(v.d, v.e), v.r; g != e {
			t.Errorf("d %d e %d: got %t, exp %t", v.d, v.e, g, e)
		}
	}
}

func TestHasFactorBigInt(t *testing.T) {
	data := []struct {
		d interface{}
		e uint32
		r bool
	}{
		{0, 42, false},
		{1, 24, true},
		{2, 22, false},
		{3, 2, true},
		{3, 3, false},
		{3, 4, true},
		{3, 5, false},
		{3, 6, true},
		{5, 4, true},
		{5, 5, false},
		{5, 6, false},
		{5, 7, false},
		{5, 8, true},
		{5, 9, false},
		{5, 10, false},
		{5, 11, false},
		{5, 12, true},
		{7, 3, true},
		{7, 6, true},
		{7, 9, true},
		{9, 6, true},
		{9, 12, true},
		{9, 18, true},
		{11, 10, true},
		{23, 11, true},
		{89, 11, true},
		{47, 23, true},
		{193707721, 67, true},
		{13007, 929, true},
		{264248689, 500471, true},
		{112027889, 1000249, true},
		{252079759, 2000633, true},
		{222054983, 3000743, true},
		{1920355681, 4000741, true},
		{330036367, 5000551, true},
		{1020081431, 6000479, true},
		{840074281, 7000619, true},
		{624031279, 8000401, true},
		{378031207, 9000743, true},
		{380036519, 10000961, true},
		{40001447, 20000723, true},
		{"1872347344039", 1000099, true},
		{"11502865265922183403581252152383", 100279, true},
		{"533975545077050000610542659519277030089249998649", 7293457, true},
	}

	var d big.Int
	for _, v := range data {
		bigInt(&d, v.d)
		if g, e := HasFactorBigInt(&d, v.e), v.r; g != e {
			t.Errorf("d %s e %d: got %t, exp %t", &d, v.e, g, e)
		}

		if g, e := HasFactorBigInt2(&d, big.NewInt(int64(v.e))), v.r; g != e {
			t.Errorf("d %s e %d: got %t, exp %t", &d, v.e, g, e)
		}
	}
}

var once309 sync.Once

func BenchmarkHasFactorUint32Rnd(b *testing.B) {
	const N = 1 << 16
	b.StopTimer()
	type t struct{ d, e uint32 }
	a := make([]t, N)
	r := r32()
	for i := range a {
		a[i] = t{
			uint32(r.Next()) | 1,
			uint32(r.Next()),
		}
	}
	once309.Do(func() { b.Log("Random 32 bit factor, random 32 bit exponent\n") })
	runtime.GC()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		v := a[i&(N-1)]
		HasFactorUint32(v.d, v.e)
	}
}

var once332 sync.Once

func BenchmarkHasFactorUint64Rnd(b *testing.B) {
	const N = 1 << 16
	b.StopTimer()
	type t struct {
		d uint64
		e uint32
	}
	a := make([]t, N)
	r := r64()
	for i := range a {
		a[i] = t{
			uint64(r.Next().Int64()) | 1,
			uint32(r.Next().Int64()),
		}
	}
	once332.Do(func() { b.Log("Random 64 bit factor, random 32 bit exponent\n") })
	runtime.GC()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		v := a[i&(N-1)]
		HasFactorUint64(v.d, v.e)
	}
}

var once358 sync.Once

func BenchmarkHasFactorBigIntRnd_128b(b *testing.B) {
	const N = 1 << 16
	b.StopTimer()
	type t struct {
		d *big.Int
		e uint32
	}
	a := make([]t, N)
	r, err := mathutil.NewFCBig(_1, New(128), true)
	if err != nil {
		b.Fatal(err)
	}
	r2 := r32()
	for i := range a {
		dd := r.Next()
		a[i] = t{
			dd.SetBit(dd, 0, 1),
			uint32(r2.Next()),
		}
	}
	once358.Do(func() { b.Log("Random 128 bit factor, random 32 bit exponent\n") })
	runtime.GC()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		v := a[i&(N-1)]
		HasFactorBigInt(v.d, v.e)
	}
}

var (
	f104b, _ = big.NewInt(0).SetString( // 104 bit factor of M100279
		"11502865265922183403581252152383",
		10,
	)
	f137b, _ = big.NewInt(0).SetString( // 137 bit factor of M7293457
		"533975545077050000610542659519277030089249998649",
		10,
	)
)

var once396 sync.Once

func BenchmarkHasFactorBigInt_104b(b *testing.B) {
	b.StopTimer()
	once396.Do(func() { b.Log("Verify a 104 bit factor of M100279 (16.6 bit exponent)\n") })
	runtime.GC()
	var r bool
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		r = HasFactorBigInt(f104b, 100279)
	}
	if !r {
		b.Fatal(r)
	}
}

var once412 sync.Once

func BenchmarkHasFactorBigIntMod104b(b *testing.B) {
	b.StopTimer()
	once412.Do(func() { b.Log("Verify a 104 bit factor of M100279 (16.6 bit exponent) using big.Int.Mod\n") })
	runtime.GC()
	m := New(100279)
	var x big.Int
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		x.Mod(m, f104b)
	}
	if x.Cmp(_0) != 0 {
		b.Fatal(x)
	}
}

var once429 sync.Once

func BenchmarkHasFactorBigInt_137b(b *testing.B) {
	b.StopTimer()
	once429.Do(func() { b.Log("Verify a 137 bit factor of M7293457 (22.8 bit exponent)\n") })
	runtime.GC()
	var r bool
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		r = HasFactorBigInt(f137b, 7293457)
	}
	if !r {
		b.Fatal(r)
	}
}

var once445 sync.Once

func BenchmarkHasFactorBigIntMod137b(b *testing.B) {
	b.StopTimer()
	once445.Do(func() { b.Log("Verify a 137 bit factor of M7293457 (22.8 bit exponent) using big.Int.Mod\n") })
	runtime.GC()
	m := New(7293457)
	var x big.Int
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		x.Mod(m, f137b)
	}
	if x.Cmp(_0) != 0 {
		b.Fatal(x)
	}
}

func bigInt(b *big.Int, v interface{}) {
	switch v := v.(type) {
	case int:
		b.SetInt64(int64(v))
	case string:
		if _, ok := b.SetString(v, 10); !ok {
			panic("bigInt: bad decimal string")
		}
	default:
		panic("bigInt: bad v.(type)")
	}
}

func TestFromFactorBigInt(t *testing.T) {
	data := []struct {
		d interface{}
		n uint32
	}{
		{0, 0},
		{1, 1},
		{2, 0},
		{3, 2},
		{4, 0},
		{5, 4},
		{7, 3},
		{9, 6},
		{11, 10},
		{23, 11},
		{89, 11},
		{"7432339208719", 101},
		{"198582684439", 1009},
		{"20649907789079", 1009},
		{"21624641697047", 1009},
		{"30850253615723594284324529", 1009},
		{"1134327302421596486779379019599", 1009},
		{35311753, 10009},
		{"104272300687", 10009},
		{"10409374085465521", 10009},
		{"890928517778601397463", 10009},
		{6400193, 100003},
	}

	f := func(d *big.Int, max, e uint32) {
		if g := FromFactorBigInt(d, max); g != e {
			t.Fatalf("%s %d %d %d", d, max, g, e)
		}
	}

	var d big.Int
	for _, v := range data {
		bigInt(&d, v.d)
		switch {
		case v.n > 0:
			f(&d, v.n-1, 0)
		default: // v.n == 0
			f(&d, 100, 0)
		}
		f(&d, v.n, v.n)
	}
}

var f20b = big.NewInt(200000447) // 20 bit factor of M100000223

func benchmarkFromFactorBigInt(b *testing.B, f *big.Int, max uint32) {
	var n uint32
	for i := 0; i < b.N; i++ {
		n = FromFactorBigInt(f, max)
	}
	if n != 0 {
		b.Fatal(n)
	}
}

func BenchmarkFromFactorBigInt20b_1e1(b *testing.B) {
	benchmarkFromFactorBigInt(b, f20b, 1e1)
}

func BenchmarkFromFactorBigInt20b_1e2(b *testing.B) {
	benchmarkFromFactorBigInt(b, f20b, 1e2)
}

func BenchmarkFromFactorBigInt20b_1e3(b *testing.B) {
	benchmarkFromFactorBigInt(b, f20b, 1e3)
}

func BenchmarkFromFactorBigInt20b_1e4(b *testing.B) {
	benchmarkFromFactorBigInt(b, f20b, 1e4)
}

func BenchmarkFromFactorBigInt20b_1e5(b *testing.B) {
	benchmarkFromFactorBigInt(b, f20b, 1e5)
}

func BenchmarkFromFactorBigInt20b_1e6(b *testing.B) {
	benchmarkFromFactorBigInt(b, f20b, 1e6)
}

func BenchmarkFromFactorBigInt137b_1e1(b *testing.B) {
	benchmarkFromFactorBigInt(b, f137b, 1e1)
}

func BenchmarkFromFactorBigInt137b_1e2(b *testing.B) {
	benchmarkFromFactorBigInt(b, f137b, 1e2)
}

func BenchmarkFromFactorBigInt137b_1e3(b *testing.B) {
	benchmarkFromFactorBigInt(b, f137b, 1e3)
}

func BenchmarkFromFactorBigInt137b_1e4(b *testing.B) {
	benchmarkFromFactorBigInt(b, f137b, 1e4)
}

func BenchmarkFromFactorBigInt137b_1e5(b *testing.B) {
	benchmarkFromFactorBigInt(b, f137b, 1e5)
}

func BenchmarkFromFactorBigInt137b_1e6(b *testing.B) {
	benchmarkFromFactorBigInt(b, f137b, 1e6)
}
func TestMod(t *testing.T) {
	const N = 1e4
	data := []struct {
		mod, n int64
		exp    uint32
	}{
		{0, 0x00, 3},
		{1, 0x01, 3},
		{3, 0x03, 3},
		{0, 0x07, 3},
		{1, 0x0f, 3},
		{3, 0x1f, 3},
		{0, 0x3f, 3},
		{1, 0x7f, 3},
		{3, 0xff, 3},
		{0, 0x1ff, 3},
	}

	var mod, n big.Int
	for _, v := range data {
		n.SetInt64(v.n)
		p := Mod(&mod, &n, v.exp)
		if p != &mod {
			t.Fatal(p)
		}

		if g, e := mod.Int64(), v.mod; g != e {
			t.Fatal(v.n, v.exp, g, e)
		}
	}

	f := func(in int64, exp uint32) {
		n.SetInt64(in)
		mod.Mod(&n, New(exp))
		e := mod.Int64()
		Mod(&mod, &n, exp)
		g := mod.Int64()
		if g != e {
			t.Fatal(in, exp, g, e)
		}
	}

	r32, _ := mathutil.NewFC32(1, 1e6, true)
	r64, _ := mathutil.NewFCBig(_0, big.NewInt(math.MaxInt64), true)
	for i := 0; i < N; i++ {
		f(r64.Next().Int64(), uint32(r32.Next()))
	}
}

func benchmarkMod(b *testing.B, w, exp uint32) {
	b.StopTimer()
	var n, mod big.Int
	n.Rand(rand.New(rand.NewSource(1)), New(w))
	n.SetBit(&n, int(w), 1)
	runtime.GC()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		Mod(&mod, &n, exp)
	}
}

func benchmarkModBig(b *testing.B, w, exp uint32) {
	b.StopTimer()
	var n, mod big.Int
	n.Rand(rand.New(rand.NewSource(1)), New(w))
	n.SetBit(&n, int(w), 1)
	runtime.GC()
	runtime.GC()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		mod.Mod(&n, New(exp))
	}
}

func BenchmarkMod_1e2(b *testing.B) {
	benchmarkMod(b, 1e2+2, 1e2)
}

func BenchmarkModBig_1e2(b *testing.B) {
	benchmarkModBig(b, 1e2+2, 1e2)
}

func BenchmarkMod_1e3(b *testing.B) {
	benchmarkMod(b, 1e3+2, 1e3)
}

func BenchmarkModBig_1e3(b *testing.B) {
	benchmarkModBig(b, 1e3+2, 1e3)
}

func BenchmarkMod_1e4(b *testing.B) {
	benchmarkMod(b, 1e4+2, 1e4)
}

func BenchmarkModBig_1e4(b *testing.B) {
	benchmarkModBig(b, 1e4+2, 1e4)
}

func BenchmarkMod_1e5(b *testing.B) {
	benchmarkMod(b, 1e5+2, 1e5)
}

func BenchmarkModBig_1e5(b *testing.B) {
	benchmarkModBig(b, 1e5+2, 1e5)
}

func BenchmarkMod_1e6(b *testing.B) {
	benchmarkMod(b, 1e6+2, 1e6)
}

func BenchmarkModBig_1e6(b *testing.B) {
	benchmarkModBig(b, 1e6+2, 1e6)
}

func BenchmarkMod_1e7(b *testing.B) {
	benchmarkMod(b, 1e7+2, 1e7)
}

func BenchmarkModBig_1e7(b *testing.B) {
	benchmarkModBig(b, 1e7+2, 1e7)
}

func BenchmarkMod_1e8(b *testing.B) {
	benchmarkMod(b, 1e8+2, 1e8)
}

func BenchmarkModBig_1e8(b *testing.B) {
	benchmarkModBig(b, 1e8+2, 1e8)
}

func BenchmarkMod_5e8(b *testing.B) {
	benchmarkMod(b, 5e8+2, 5e8)
}

func BenchmarkModBig_5e8(b *testing.B) {
	benchmarkModBig(b, 5e8+2, 5e8)
}

func TestModPow(t *testing.T) {
	const N = 2e2
	data := []struct{ b, e, m, r uint32 }{
		{0, 1, 1, 0},
		{0, 2, 1, 0},
		{0, 3, 1, 0},

		{1, 0, 1, 0},
		{1, 1, 1, 0},
		{1, 2, 1, 0},
		{1, 3, 1, 0},

		{2, 0, 1, 0},
		{2, 1, 1, 0},
		{2, 2, 1, 0},
		{2, 3, 1, 0},

		{2, 3, 4, 8},
		{2, 3, 5, 4},
		{2, 4, 3, 1},
		{3, 3, 3, 3},
		{3, 4, 5, 30},
	}

	f := func(b, e, m uint32, expect *big.Int) {
		got := ModPow(b, e, m)
		if got.Cmp(expect) != 0 {
			t.Fatal(b, e, m, got, expect)
		}
	}

	var r big.Int
	for _, v := range data {
		r.SetInt64(int64(v.r))
		f(v.b, v.e, v.m, &r)
	}

	rg, _ := mathutil.NewFC32(2, 1<<10, true)
	var bb big.Int
	for i := 0; i < N; i++ {
		b, e, m := uint32(rg.Next()), uint32(rg.Next()), uint32(rg.Next())
		bb.SetInt64(int64(b))
		f(b, e, m, mathutil.ModPowBigInt(&bb, New(e), New(m)))
	}
}

func benchmarkModPow2(b *testing.B, e, m uint32) {
	b.StopTimer()
	runtime.GC()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		ModPow2(e, m)
	}
}

func benchmarkModPow(b *testing.B, base, e, m uint32) {
	b.StopTimer()
	runtime.GC()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		ModPow(base, e, m)
	}
}

func benchmarkModPowBig(b *testing.B, base, e, m uint32) {
	b.StopTimer()
	bb := big.NewInt(int64(base))
	ee := New(e)
	mm := New(m)
	runtime.GC()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		mathutil.ModPowBigInt(bb, ee, mm)
	}
}

func BenchmarkModPow2_1e2(b *testing.B) {
	benchmarkModPow2(b, 1e2, 1e2+1)
}

func BenchmarkModPow_2_1e2(b *testing.B) {
	benchmarkModPow(b, 2, 1e2, 1e2+1)
}

func BenchmarkModPowB_2_1e2(b *testing.B) {
	benchmarkModPowBig(b, 2, 1e2, 1e2+1)
}

func BenchmarkModPow_3_1e2(b *testing.B) {
	benchmarkModPow(b, 3, 1e2, 1e2+1)
}

func BenchmarkModPowB_3_1e2(b *testing.B) {
	benchmarkModPowBig(b, 3, 1e2, 1e2+1)
}

// ----

func BenchmarkModPow2_1e3(b *testing.B) {
	benchmarkModPow2(b, 1e3, 1e3+1)
}

func BenchmarkModPow_2_1e3(b *testing.B) {
	benchmarkModPow(b, 2, 1e3, 1e3+1)
}

func BenchmarkModPowB_2_1e3(b *testing.B) {
	benchmarkModPowBig(b, 2, 1e3, 1e3+1)
}

func BenchmarkModPow_3_1e3(b *testing.B) {
	benchmarkModPow(b, 3, 1e3, 1e3+1)
}

func BenchmarkModPowB_3_1e3(b *testing.B) {
	benchmarkModPowBig(b, 3, 1e3, 1e3+1)
}

// ----

func BenchmarkModPow2_1e4(b *testing.B) {
	benchmarkModPow2(b, 1e4, 1e4+1)
}

func BenchmarkModPow_2_1e4(b *testing.B) {
	benchmarkModPow(b, 2, 1e4, 1e4+1)
}

func BenchmarkModPowB_2_1e4(b *testing.B) {
	benchmarkModPowBig(b, 2, 1e4, 1e4+1)
}

func BenchmarkModPow_3_1e4(b *testing.B) {
	benchmarkModPow(b, 3, 1e4, 1e4+1)
}

func BenchmarkModPowB_3_1e4(b *testing.B) {
	benchmarkModPowBig(b, 3, 1e4, 1e4+1)
}

// ----

func BenchmarkModPow2_1e5(b *testing.B) {
	benchmarkModPow2(b, 1e5, 1e5+1)
}

func BenchmarkModPow2_1e6(b *testing.B) {
	benchmarkModPow2(b, 1e6, 1e6+1)
}

func BenchmarkModPow2_1e7(b *testing.B) {
	benchmarkModPow2(b, 1e7, 1e7+1)
}

func BenchmarkModPow2_1e8(b *testing.B) {
	benchmarkModPow2(b, 1e8, 1e8+1)
}

func BenchmarkModPow2_1e9(b *testing.B) {
	benchmarkModPow2(b, 1e9, 1e9+1)
}

func TestModPow2(t *testing.T) {
	const N = 1e3
	data := []struct{ e, m uint32 }{
		// e == 0 -> x == 0
		{0, 2},
		{0, 3},
		{0, 4},

		{1, 2},
		{1, 3},
		{1, 4},
		{1, 5},

		{2, 2},
		{2, 3},
		{2, 4},
		{2, 5},

		{3, 2},
		{3, 3},
		{3, 4},
		{3, 5},
		{3, 6},
		{3, 7},
		{3, 8},
		{3, 9},

		{4, 2},
		{4, 3},
		{4, 4},
		{4, 5},
		{4, 6},
		{4, 7},
		{4, 8},
		{4, 9},
	}

	var got big.Int
	f := func(e, m uint32) {
		x := ModPow2(e, m)
		exp := ModPow(2, e, m)
		got.SetInt64(0)
		got.SetBit(&got, int(x), 1)
		if got.Cmp(exp) != 0 {
			t.Fatalf("\ne %d, m %d\ng: %s\ne: %s", e, m, &got, exp)
		}
	}

	for _, v := range data {
		f(v.e, v.m)
	}

	rg, _ := mathutil.NewFC32(2, 1<<10, true)
	for i := 0; i < N; i++ {
		f(uint32(rg.Next()), uint32(rg.Next()))
	}
}
