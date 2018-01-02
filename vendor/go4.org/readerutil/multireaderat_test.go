/*
Copyright 2016 The Go4 Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package readerutil

import (
	"io"
	"io/ioutil"
	"strings"
	"testing"
)

func TestMultiReaderAt(t *testing.T) {
	sra := NewMultiReaderAt(
		io.NewSectionReader(strings.NewReader("xaaax"), 1, 3),
		io.NewSectionReader(strings.NewReader("xxbbbbxx"), 2, 3),
		io.NewSectionReader(strings.NewReader("cccx"), 0, 3),
	)
	if sra.Size() != 9 {
		t.Fatalf("Size = %d; want 9", sra.Size())
	}
	const full = "aaabbbccc"
	for start := 0; start < len(full); start++ {
		for end := start; end < len(full); end++ {
			want := full[start:end]
			got, err := ioutil.ReadAll(io.NewSectionReader(sra, int64(start), int64(end-start)))
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != want {
				t.Errorf("for start=%d, end=%d: ReadAll = %q; want %q", start, end, got, want)
			}
		}
	}
}
