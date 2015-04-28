// Copyright 2014 The zappy Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Copyright 2011 The Snappy-Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the SNAPPY-GO-LICENSE file.

package zappy

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"math/rand"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"code.google.com/p/snappy-go/snappy"
)

var dbg = func(s string, va ...interface{}) {
	_, fn, fl, _ := runtime.Caller(1)
	fmt.Printf("%s:%d: ", path.Base(fn), fl)
	fmt.Printf(s, va...)
	fmt.Println()
}

func use(...interface{}) {}

var (
	download = flag.Bool("download", false, "If true, download any missing files before running benchmarks")
	pureGo   = flag.String("purego", "", "verify 'purego' build tag functionality for value `false` or `true`")
)

func roundtrip(b, ebuf, dbuf []byte) error {
	e, err := Encode(ebuf, b)
	if err != nil {
		return fmt.Errorf("encoding error: %v", err)
	}
	d, err := Decode(dbuf, e)
	if err != nil {
		return fmt.Errorf("decoding error: %v", err)
	}
	if !bytes.Equal(b, d) {
		return fmt.Errorf("roundtrip mismatch:\n\twant %v\n\tgot  %v", b, d)
	}
	return nil
}

func TestEmpty(t *testing.T) {
	if err := roundtrip(nil, nil, nil); err != nil {
		t.Fatal(err)
	}
}

func TestSmallCopy(t *testing.T) {
	for _, ebuf := range [][]byte{nil, make([]byte, 20), make([]byte, 64)} {
		for _, dbuf := range [][]byte{nil, make([]byte, 20), make([]byte, 64)} {
			for i := 0; i < 32; i++ {
				s := "aaaa" + strings.Repeat("b", i) + "aaaabbbb"
				if err := roundtrip([]byte(s), ebuf, dbuf); err != nil {
					t.Fatalf("len(ebuf)=%d, len(dbuf)=%d, i=%d: %v", len(ebuf), len(dbuf), i, err)
				}
			}
		}
	}
}

func TestSmallRand(t *testing.T) {
	rand.Seed(27354294)
	for n := 1; n < 20000; n += 23 {
		b := make([]byte, n)
		for i := range b {
			b[i] = uint8(rand.Uint32())
		}
		if err := roundtrip(b, nil, nil); err != nil {
			t.Fatal(err)
		}
	}
}

func TestSmallRegular(t *testing.T) {
	for n := 1; n < 20000; n += 23 {
		b := make([]byte, n)
		for i := range b {
			b[i] = uint8(i%10 + 'a')
		}
		if err := roundtrip(b, nil, nil); err != nil {
			t.Fatal(n, err)
		}
	}
}

func benchDecode(b *testing.B, src []byte) {
	encoded, err := Encode(nil, src)
	if err != nil {
		b.Fatal(err)
	}
	// Bandwidth is in amount of uncompressed data.
	b.SetBytes(int64(len(src)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Decode(src, encoded)
	}
}

func benchEncode(b *testing.B, src []byte) {
	// Bandwidth is in amount of uncompressed data.
	b.SetBytes(int64(len(src)))
	dst := make([]byte, MaxEncodedLen(len(src)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Encode(dst, src)
	}
}

func readFile(b *testing.B, filename string) []byte {
	src, err := ioutil.ReadFile(filename)
	if err != nil {
		b.Fatalf("failed reading %s: %s", filename, err)
	}
	if len(src) == 0 {
		b.Fatalf("%s has zero length", filename)
	}
	return src
}

func readFile2(t *testing.T, filename string) []byte {
	src, err := ioutil.ReadFile(filename)
	if err != nil {
		t.Fatalf("failed reading %s: %s", filename, err)
	}
	if len(src) == 0 {
		t.Fatalf("%s has zero length", filename)
	}
	return src
}

// expand returns a slice of length n containing repeated copies of src.
func expand(src []byte, n int) []byte {
	dst := make([]byte, n)
	for x := dst; len(x) > 0; {
		i := copy(x, src)
		x = x[i:]
	}
	return dst
}

func benchWords(b *testing.B, n int, decode bool) {
	// Note: the file is OS-language dependent so the resulting values are not
	// directly comparable for non-US-English OS installations.
	data := expand(readFile(b, "/usr/share/dict/words"), n)
	if decode {
		benchDecode(b, data)
	} else {
		benchEncode(b, data)
	}
}

func BenchmarkWordsDecode1e3(b *testing.B) { benchWords(b, 1e3, true) }
func BenchmarkWordsDecode1e4(b *testing.B) { benchWords(b, 1e4, true) }
func BenchmarkWordsDecode1e5(b *testing.B) { benchWords(b, 1e5, true) }
func BenchmarkWordsDecode1e6(b *testing.B) { benchWords(b, 1e6, true) }
func BenchmarkWordsEncode1e3(b *testing.B) { benchWords(b, 1e3, false) }
func BenchmarkWordsEncode1e4(b *testing.B) { benchWords(b, 1e4, false) }
func BenchmarkWordsEncode1e5(b *testing.B) { benchWords(b, 1e5, false) }
func BenchmarkWordsEncode1e6(b *testing.B) { benchWords(b, 1e6, false) }

// testFiles' values are copied directly from
// https://code.google.com/p/snappy/source/browse/trunk/snappy_unittest.cc.
// The label field is unused in zappy.
var testFiles = []struct {
	label    string
	filename string
}{
	{"html", "html"},
	{"urls", "urls.10K"},
	{"jpg", "house.jpg"},
	{"pdf", "mapreduce-osdi-1.pdf"},
	{"html4", "html_x_4"},
	{"cp", "cp.html"},
	{"c", "fields.c"},
	{"lsp", "grammar.lsp"},
	{"xls", "kennedy.xls"},
	{"txt1", "alice29.txt"},
	{"txt2", "asyoulik.txt"},
	{"txt3", "lcet10.txt"},
	{"txt4", "plrabn12.txt"},
	{"bin", "ptt5"},
	{"sum", "sum"},
	{"man", "xargs.1"},
	{"pb", "geo.protodata"},
	{"gaviota", "kppkn.gtb"},
}

// The test data files are present at this canonical URL.
const baseURL = "https://snappy.googlecode.com/svn/trunk/testdata/"

func downloadTestdata(basename string) (errRet error) {
	filename := filepath.Join("testdata", basename)
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create %s: %s", filename, err)
	}

	defer f.Close()
	defer func() {
		if errRet != nil {
			os.Remove(filename)
		}
	}()
	resp, err := http.Get(baseURL + basename)
	if err != nil {
		return fmt.Errorf("failed to download %s: %s", baseURL+basename, err)
	}
	defer resp.Body.Close()
	_, err = io.Copy(f, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to write %s: %s", filename, err)
	}
	return nil
}

func benchFile(b *testing.B, n int, decode bool) {
	filename := filepath.Join("testdata", testFiles[n].filename)
	if stat, err := os.Stat(filename); err != nil || stat.Size() == 0 {
		if !*download {
			b.Fatal("test data not found; skipping benchmark without the -download flag")
		}
		// Download the official snappy C++ implementation reference test data
		// files for benchmarking.
		if err := os.Mkdir("testdata", 0777); err != nil && !os.IsExist(err) {
			b.Fatalf("failed to create testdata: %s", err)
		}
		for _, tf := range testFiles {
			if err := downloadTestdata(tf.filename); err != nil {
				b.Fatalf("failed to download testdata: %s", err)
			}
		}
	}
	data := readFile(b, filename)
	if decode {
		benchDecode(b, data)
	} else {
		benchEncode(b, data)
	}
}

// Naming convention is kept similar to what snappy's C++ implementation uses.
func Benchmark_UFlat0(b *testing.B)  { benchFile(b, 0, true) }
func Benchmark_UFlat1(b *testing.B)  { benchFile(b, 1, true) }
func Benchmark_UFlat2(b *testing.B)  { benchFile(b, 2, true) }
func Benchmark_UFlat3(b *testing.B)  { benchFile(b, 3, true) }
func Benchmark_UFlat4(b *testing.B)  { benchFile(b, 4, true) }
func Benchmark_UFlat5(b *testing.B)  { benchFile(b, 5, true) }
func Benchmark_UFlat6(b *testing.B)  { benchFile(b, 6, true) }
func Benchmark_UFlat7(b *testing.B)  { benchFile(b, 7, true) }
func Benchmark_UFlat8(b *testing.B)  { benchFile(b, 8, true) }
func Benchmark_UFlat9(b *testing.B)  { benchFile(b, 9, true) }
func Benchmark_UFlat10(b *testing.B) { benchFile(b, 10, true) }
func Benchmark_UFlat11(b *testing.B) { benchFile(b, 11, true) }
func Benchmark_UFlat12(b *testing.B) { benchFile(b, 12, true) }
func Benchmark_UFlat13(b *testing.B) { benchFile(b, 13, true) }
func Benchmark_UFlat14(b *testing.B) { benchFile(b, 14, true) }
func Benchmark_UFlat15(b *testing.B) { benchFile(b, 15, true) }
func Benchmark_UFlat16(b *testing.B) { benchFile(b, 16, true) }
func Benchmark_UFlat17(b *testing.B) { benchFile(b, 17, true) }
func Benchmark_ZFlat0(b *testing.B)  { benchFile(b, 0, false) }
func Benchmark_ZFlat1(b *testing.B)  { benchFile(b, 1, false) }
func Benchmark_ZFlat2(b *testing.B)  { benchFile(b, 2, false) }
func Benchmark_ZFlat3(b *testing.B)  { benchFile(b, 3, false) }
func Benchmark_ZFlat4(b *testing.B)  { benchFile(b, 4, false) }
func Benchmark_ZFlat5(b *testing.B)  { benchFile(b, 5, false) }
func Benchmark_ZFlat6(b *testing.B)  { benchFile(b, 6, false) }
func Benchmark_ZFlat7(b *testing.B)  { benchFile(b, 7, false) }
func Benchmark_ZFlat8(b *testing.B)  { benchFile(b, 8, false) }
func Benchmark_ZFlat9(b *testing.B)  { benchFile(b, 9, false) }
func Benchmark_ZFlat10(b *testing.B) { benchFile(b, 10, false) }
func Benchmark_ZFlat11(b *testing.B) { benchFile(b, 11, false) }
func Benchmark_ZFlat12(b *testing.B) { benchFile(b, 12, false) }
func Benchmark_ZFlat13(b *testing.B) { benchFile(b, 13, false) }
func Benchmark_ZFlat14(b *testing.B) { benchFile(b, 14, false) }
func Benchmark_ZFlat15(b *testing.B) { benchFile(b, 15, false) }
func Benchmark_ZFlat16(b *testing.B) { benchFile(b, 16, false) }
func Benchmark_ZFlat17(b *testing.B) { benchFile(b, 17, false) }

func TestCmp(t *testing.T) {
	var ts, tz, to int
	for i := 0; i <= 17; i++ {
		filename := filepath.Join("testdata", testFiles[i].filename)
		if stat, err := os.Stat(filename); err != nil || stat.Size() == 0 {
			if !*download {
				t.Fatal("test data not found; skipping test without the -download flag")
			}
			// Download the official snappy C++ implementation reference test data
			// files for benchmarking.
			if err := os.Mkdir("testdata", 0777); err != nil && !os.IsExist(err) {
				t.Fatalf("failed to create testdata: %s", err)
			}
			for _, tf := range testFiles {
				if err := downloadTestdata(tf.filename); err != nil {
					t.Fatalf("failed to download testdata: %s", err)
				}
			}
		}
		data := readFile2(t, filename)
		orig := len(data)
		to += orig
		senc, err := snappy.Encode(nil, data)
		if err != nil {
			t.Fatal(err)
		}

		ns := len(senc)
		zenc, err := Encode(nil, data)
		if err != nil {
			t.Fatal(err)
		}

		nz := len(zenc)
		t.Logf("%35s: snappy %7d, zappy %7d, %.3f, orig %7d", filename, ns, nz, float64(nz)/float64(ns), orig)
		ts += ns
		tz += nz
	}
	t.Logf("%35s: snappy %7d, zappy %7d, %.3f, orig %7d", "TOTAL", ts, tz, float64(tz)/float64(ts), to)
}

func TestBitIndex(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for n := 16; n <= 1<<16; n <<= 1 {
		data := make([]byte, n)
		for i := 0; i < n/1000+1; i++ {
			data[rng.Int()%n] = 1
		}

		senc, err := snappy.Encode(nil, data)
		if err != nil {
			t.Fatal(err)
		}

		ns := len(senc)
		zenc, err := Encode(nil, data)
		if err != nil {
			t.Fatal(err)
		}

		nz := len(zenc)
		t.Logf("Sparse bit index %7d B: snappy %7d, zappy %7d, %.3f", n, ns, nz, float64(nz)/float64(ns))
	}
}

func TestPureGo(t *testing.T) {
	var purego bool
	switch s := *pureGo; s {
	case "false":
		// nop
	case "true":
		purego = true
	default:
		t.Logf("Not performed: %q", s)
		return
	}

	if g, e := puregoDecode(), purego; g != e {
		t.Fatal("Decode", g, e)
	}

	if g, e := puregoEncode(), purego; g != e {
		t.Fatal("Encode", g, e)
	}
}
