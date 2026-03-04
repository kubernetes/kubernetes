/*
Copyright 2019 The Kubernetes Authors.

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

package ioutils

import (
	"bytes"
	"fmt"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLimitWriter(t *testing.T) {
	r := rand.New(rand.NewSource(1234)) // Fixed source to prevent flakes.

	tests := []struct {
		inputSize, limit, writeSize int64
	}{
		// Single write tests
		{100, 101, 100},
		{100, 100, 100},
		{100, 99, 100},
		{1, 1, 1},
		{100, 10, 100},
		{100, 0, 100},
		{100, -1, 100},
		// Multi write tests
		{100, 101, 10},
		{100, 100, 10},
		{100, 99, 10},
		{100, 10, 10},
		{100, 0, 10},
		{100, -1, 10},
	}

	for _, test := range tests {
		t.Run(fmt.Sprintf("inputSize=%d limit=%d writes=%d", test.inputSize, test.limit, test.writeSize), func(t *testing.T) {
			input := make([]byte, test.inputSize)
			r.Read(input)
			output := &bytes.Buffer{}
			w := LimitWriter(output, test.limit)

			var (
				err     error
				written int64
				n       int
			)
			for written < test.inputSize && err == nil {
				n, err = w.Write(input[written : written+test.writeSize])
				written += int64(n)
			}

			expectWritten := bounded(0, test.inputSize, test.limit)
			assert.EqualValues(t, expectWritten, written)
			if expectWritten <= 0 {
				assert.Empty(t, output)
			} else {
				assert.Equal(t, input[:expectWritten], output.Bytes())
			}

			if test.limit < test.inputSize {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func bounded(min, val, max int64) int64 {
	if max < val {
		val = max
	}
	if val < min {
		val = min
	}
	return val
}
