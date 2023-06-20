// Copyright 2014 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Build only when actually fuzzing
//go:build gofuzz
// +build gofuzz

package expfmt

import "bytes"

// Fuzz text metric parser with with github.com/dvyukov/go-fuzz:
//
//     go-fuzz-build github.com/prometheus/common/expfmt
//     go-fuzz -bin expfmt-fuzz.zip -workdir fuzz
//
// Further input samples should go in the folder fuzz/corpus.
func Fuzz(in []byte) int {
	parser := TextParser{}
	_, err := parser.TextToMetricFamilies(bytes.NewReader(in))

	if err != nil {
		return 0
	}

	return 1
}
