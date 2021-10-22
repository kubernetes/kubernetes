// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package norm

import (
	"bytes"
	"fmt"
	"testing"
)

var bufSizes = []int{1, 2, 3, 4, 5, 6, 7, 8, 100, 101, 102, 103, 4000, 4001, 4002, 4003}

func readFunc(size int) appendFunc {
	return func(f Form, out []byte, s string) []byte {
		out = append(out, s...)
		r := f.Reader(bytes.NewBuffer(out))
		buf := make([]byte, size)
		result := []byte{}
		for n, err := 0, error(nil); err == nil; {
			n, err = r.Read(buf)
			result = append(result, buf[:n]...)
		}
		return result
	}
}

func TestReader(t *testing.T) {
	for _, s := range bufSizes {
		name := fmt.Sprintf("TestReader%d", s)
		runNormTests(t, name, readFunc(s))
	}
}

func writeFunc(size int) appendFunc {
	return func(f Form, out []byte, s string) []byte {
		in := append(out, s...)
		result := new(bytes.Buffer)
		w := f.Writer(result)
		buf := make([]byte, size)
		for n := 0; len(in) > 0; in = in[n:] {
			n = copy(buf, in)
			_, _ = w.Write(buf[:n])
		}
		w.Close()
		return result.Bytes()
	}
}

func TestWriter(t *testing.T) {
	for _, s := range bufSizes {
		name := fmt.Sprintf("TestWriter%d", s)
		runNormTests(t, name, writeFunc(s))
	}
}
