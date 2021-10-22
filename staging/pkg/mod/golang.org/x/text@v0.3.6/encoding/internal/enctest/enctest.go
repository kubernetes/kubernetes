// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package enctest

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"strings"
	"testing"

	"golang.org/x/text/encoding"
	"golang.org/x/text/encoding/internal/identifier"
	"golang.org/x/text/transform"
)

// Encoder or Decoder
type Transcoder interface {
	transform.Transformer
	Bytes([]byte) ([]byte, error)
	String(string) (string, error)
}

func TestEncoding(t *testing.T, e encoding.Encoding, encoded, utf8, prefix, suffix string) {
	for _, direction := range []string{"Decode", "Encode"} {
		t.Run(fmt.Sprintf("%v/%s", e, direction), func(t *testing.T) {

			var coder Transcoder
			var want, src, wPrefix, sPrefix, wSuffix, sSuffix string
			if direction == "Decode" {
				coder, want, src = e.NewDecoder(), utf8, encoded
				wPrefix, sPrefix, wSuffix, sSuffix = "", prefix, "", suffix
			} else {
				coder, want, src = e.NewEncoder(), encoded, utf8
				wPrefix, sPrefix, wSuffix, sSuffix = prefix, "", suffix, ""
			}

			dst := make([]byte, len(wPrefix)+len(want)+len(wSuffix))
			nDst, nSrc, err := coder.Transform(dst, []byte(sPrefix+src+sSuffix), true)
			if err != nil {
				t.Fatal(err)
			}
			if nDst != len(wPrefix)+len(want)+len(wSuffix) {
				t.Fatalf("nDst got %d, want %d",
					nDst, len(wPrefix)+len(want)+len(wSuffix))
			}
			if nSrc != len(sPrefix)+len(src)+len(sSuffix) {
				t.Fatalf("nSrc got %d, want %d",
					nSrc, len(sPrefix)+len(src)+len(sSuffix))
			}
			if got := string(dst); got != wPrefix+want+wSuffix {
				t.Fatalf("\ngot  %q\nwant %q", got, wPrefix+want+wSuffix)
			}

			for _, n := range []int{0, 1, 2, 10, 123, 4567} {
				input := sPrefix + strings.Repeat(src, n) + sSuffix
				g, err := coder.String(input)
				if err != nil {
					t.Fatalf("Bytes: n=%d: %v", n, err)
				}
				if len(g) == 0 && len(input) == 0 {
					// If the input is empty then the output can be empty,
					// regardless of whatever wPrefix is.
					continue
				}
				got1, want1 := string(g), wPrefix+strings.Repeat(want, n)+wSuffix
				if got1 != want1 {
					t.Fatalf("ReadAll: n=%d\ngot  %q\nwant %q",
						n, trim(got1), trim(want1))
				}
			}
		})
	}
}

func TestFile(t *testing.T, e encoding.Encoding) {
	for _, dir := range []string{"Decode", "Encode"} {
		t.Run(fmt.Sprintf("%s/%s", e, dir), func(t *testing.T) {
			dst, src, transformer, err := load(dir, e)
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			buf, err := transformer.Bytes(src)
			if err != nil {
				t.Fatalf("transform: %v", err)
			}
			if !bytes.Equal(buf, dst) {
				t.Error("transformed bytes did not match golden file")
			}
		})
	}
}

func Benchmark(b *testing.B, enc encoding.Encoding) {
	for _, direction := range []string{"Decode", "Encode"} {
		b.Run(fmt.Sprintf("%s/%s", enc, direction), func(b *testing.B) {
			_, src, transformer, err := load(direction, enc)
			if err != nil {
				b.Fatal(err)
			}
			b.SetBytes(int64(len(src)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				r := transform.NewReader(bytes.NewReader(src), transformer)
				io.Copy(ioutil.Discard, r)
			}
		})
	}
}

// testdataFiles are files in testdata/*.txt.
var testdataFiles = []struct {
	mib           identifier.MIB
	basename, ext string
}{
	{identifier.Windows1252, "candide", "windows-1252"},
	{identifier.EUCPkdFmtJapanese, "rashomon", "euc-jp"},
	{identifier.ISO2022JP, "rashomon", "iso-2022-jp"},
	{identifier.ShiftJIS, "rashomon", "shift-jis"},
	{identifier.EUCKR, "unsu-joh-eun-nal", "euc-kr"},
	{identifier.GBK, "sunzi-bingfa-simplified", "gbk"},
	{identifier.HZGB2312, "sunzi-bingfa-gb-levels-1-and-2", "hz-gb2312"},
	{identifier.Big5, "sunzi-bingfa-traditional", "big5"},
	{identifier.UTF16LE, "candide", "utf-16le"},
	{identifier.UTF8, "candide", "utf-8"},
	{identifier.UTF32BE, "candide", "utf-32be"},

	// GB18030 is a superset of GBK and is nominally a Simplified Chinese
	// encoding, but it can also represent the entire Basic Multilingual
	// Plane, including codepoints like 'â' that aren't encodable by GBK.
	// GB18030 on Simplified Chinese should perform similarly to GBK on
	// Simplified Chinese. GB18030 on "candide" is more interesting.
	{identifier.GB18030, "candide", "gb18030"},
}

func load(direction string, enc encoding.Encoding) ([]byte, []byte, Transcoder, error) {
	basename, ext, count := "", "", 0
	for _, tf := range testdataFiles {
		if mib, _ := enc.(identifier.Interface).ID(); tf.mib == mib {
			basename, ext = tf.basename, tf.ext
			count++
		}
	}
	if count != 1 {
		if count == 0 {
			return nil, nil, nil, fmt.Errorf("no testdataFiles for %s", enc)
		}
		return nil, nil, nil, fmt.Errorf("too many testdataFiles for %s", enc)
	}
	dstFile := fmt.Sprintf("../testdata/%s-%s.txt", basename, ext)
	srcFile := fmt.Sprintf("../testdata/%s-utf-8.txt", basename)
	var coder Transcoder = encoding.ReplaceUnsupported(enc.NewEncoder())
	if direction == "Decode" {
		dstFile, srcFile = srcFile, dstFile
		coder = enc.NewDecoder()
	}
	dst, err := ioutil.ReadFile(dstFile)
	if err != nil {
		if dst, err = ioutil.ReadFile("../" + dstFile); err != nil {
			return nil, nil, nil, err
		}
	}
	src, err := ioutil.ReadFile(srcFile)
	if err != nil {
		if src, err = ioutil.ReadFile("../" + srcFile); err != nil {
			return nil, nil, nil, err
		}
	}
	return dst, src, coder, nil
}

func trim(s string) string {
	if len(s) < 120 {
		return s
	}
	return s[:50] + "..." + s[len(s)-50:]
}
