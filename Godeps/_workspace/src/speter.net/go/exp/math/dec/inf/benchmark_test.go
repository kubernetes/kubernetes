package inf

import (
	"fmt"
	"math/big"
	"math/rand"
	"sync"
	"testing"
)

const maxcap = 1024 * 1024
const bits = 256
const maxscale = 32

var once sync.Once

var decInput [][2]Dec
var intInput [][2]big.Int

var initBench = func() {
	decInput = make([][2]Dec, maxcap)
	intInput = make([][2]big.Int, maxcap)
	max := new(big.Int).Lsh(big.NewInt(1), bits)
	r := rand.New(rand.NewSource(0))
	for i := 0; i < cap(decInput); i++ {
		decInput[i][0].SetUnscaledBig(new(big.Int).Rand(r, max)).
			SetScale(Scale(r.Int31n(int32(2*maxscale-1)) - int32(maxscale)))
		decInput[i][1].SetUnscaledBig(new(big.Int).Rand(r, max)).
			SetScale(Scale(r.Int31n(int32(2*maxscale-1)) - int32(maxscale)))
	}
	for i := 0; i < cap(intInput); i++ {
		intInput[i][0].Rand(r, max)
		intInput[i][1].Rand(r, max)
	}
}

func doBenchmarkDec1(b *testing.B, f func(z *Dec)) {
	once.Do(initBench)
	b.ResetTimer()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		f(&decInput[i%maxcap][0])
	}
}

func doBenchmarkDec2(b *testing.B, f func(x, y *Dec)) {
	once.Do(initBench)
	b.ResetTimer()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		f(&decInput[i%maxcap][0], &decInput[i%maxcap][1])
	}
}

func doBenchmarkInt1(b *testing.B, f func(z *big.Int)) {
	once.Do(initBench)
	b.ResetTimer()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		f(&intInput[i%maxcap][0])
	}
}

func doBenchmarkInt2(b *testing.B, f func(x, y *big.Int)) {
	once.Do(initBench)
	b.ResetTimer()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		f(&intInput[i%maxcap][0], &intInput[i%maxcap][1])
	}
}

func Benchmark_Dec_String(b *testing.B) {
	doBenchmarkDec1(b, func(x *Dec) {
		x.String()
	})
}

func Benchmark_Dec_StringScan(b *testing.B) {
	doBenchmarkDec1(b, func(x *Dec) {
		s := x.String()
		d := new(Dec)
		fmt.Sscan(s, d)
	})
}

func Benchmark_Dec_GobEncode(b *testing.B) {
	doBenchmarkDec1(b, func(x *Dec) {
		x.GobEncode()
	})
}

func Benchmark_Dec_GobEnDecode(b *testing.B) {
	doBenchmarkDec1(b, func(x *Dec) {
		g, _ := x.GobEncode()
		new(Dec).GobDecode(g)
	})
}

func Benchmark_Dec_Add(b *testing.B) {
	doBenchmarkDec2(b, func(x, y *Dec) {
		ys := y.Scale()
		y.SetScale(x.Scale())
		_ = new(Dec).Add(x, y)
		y.SetScale(ys)
	})
}

func Benchmark_Dec_AddMixed(b *testing.B) {
	doBenchmarkDec2(b, func(x, y *Dec) {
		_ = new(Dec).Add(x, y)
	})
}

func Benchmark_Dec_Sub(b *testing.B) {
	doBenchmarkDec2(b, func(x, y *Dec) {
		ys := y.Scale()
		y.SetScale(x.Scale())
		_ = new(Dec).Sub(x, y)
		y.SetScale(ys)
	})
}

func Benchmark_Dec_SubMixed(b *testing.B) {
	doBenchmarkDec2(b, func(x, y *Dec) {
		_ = new(Dec).Sub(x, y)
	})
}

func Benchmark_Dec_Mul(b *testing.B) {
	doBenchmarkDec2(b, func(x, y *Dec) {
		_ = new(Dec).Mul(x, y)
	})
}

func Benchmark_Dec_Mul_QuoExact(b *testing.B) {
	doBenchmarkDec2(b, func(x, y *Dec) {
		v := new(Dec).Mul(x, y)
		_ = new(Dec).QuoExact(v, y)
	})
}

func Benchmark_Dec_QuoRound_Fixed_Down(b *testing.B) {
	doBenchmarkDec2(b, func(x, y *Dec) {
		_ = new(Dec).QuoRound(x, y, 0, RoundDown)
	})
}

func Benchmark_Dec_QuoRound_Fixed_HalfUp(b *testing.B) {
	doBenchmarkDec2(b, func(x, y *Dec) {
		_ = new(Dec).QuoRound(x, y, 0, RoundHalfUp)
	})
}

func Benchmark_Int_String(b *testing.B) {
	doBenchmarkInt1(b, func(x *big.Int) {
		x.String()
	})
}

func Benchmark_Int_StringScan(b *testing.B) {
	doBenchmarkInt1(b, func(x *big.Int) {
		s := x.String()
		d := new(big.Int)
		fmt.Sscan(s, d)
	})
}

func Benchmark_Int_GobEncode(b *testing.B) {
	doBenchmarkInt1(b, func(x *big.Int) {
		x.GobEncode()
	})
}

func Benchmark_Int_GobEnDecode(b *testing.B) {
	doBenchmarkInt1(b, func(x *big.Int) {
		g, _ := x.GobEncode()
		new(big.Int).GobDecode(g)
	})
}

func Benchmark_Int_Add(b *testing.B) {
	doBenchmarkInt2(b, func(x, y *big.Int) {
		_ = new(big.Int).Add(x, y)
	})
}

func Benchmark_Int_Sub(b *testing.B) {
	doBenchmarkInt2(b, func(x, y *big.Int) {
		_ = new(big.Int).Sub(x, y)
	})
}

func Benchmark_Int_Mul(b *testing.B) {
	doBenchmarkInt2(b, func(x, y *big.Int) {
		_ = new(big.Int).Mul(x, y)
	})
}

func Benchmark_Int_Quo(b *testing.B) {
	doBenchmarkInt2(b, func(x, y *big.Int) {
		_ = new(big.Int).Quo(x, y)
	})
}

func Benchmark_Int_QuoRem(b *testing.B) {
	doBenchmarkInt2(b, func(x, y *big.Int) {
		_, _ = new(big.Int).QuoRem(x, y, new(big.Int))
	})
}
