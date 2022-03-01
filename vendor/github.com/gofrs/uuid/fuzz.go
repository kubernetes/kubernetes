// Copyright (c) 2018 Andrei Tudor Călin <mail@acln.ro>
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// +build gofuzz

package uuid

// Fuzz implements a simple fuzz test for FromString / UnmarshalText.
//
// To run:
//
//     $ go get github.com/dvyukov/go-fuzz/...
//     $ cd $GOPATH/src/github.com/gofrs/uuid
//     $ go-fuzz-build github.com/gofrs/uuid
//     $ go-fuzz -bin=uuid-fuzz.zip -workdir=./testdata
//
// If you make significant changes to FromString / UnmarshalText and add
// new cases to fromStringTests (in codec_test.go), please run
//
//    $ go test -seed_fuzz_corpus
//
// to seed the corpus with the new interesting inputs, then run the fuzzer.
func Fuzz(data []byte) int {
	_, err := FromString(string(data))
	if err != nil {
		return 0
	}
	return 1
}
