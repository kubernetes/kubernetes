/*
Copyright 2016 The Kubernetes Authors.

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

package tail

import (
	"os"
	"strings"
	"testing"
)

func TestReadAtMost(t *testing.T) {
	file, err := os.CreateTemp("", "TestFileReadAtMost")
	if err != nil {
		t.Fatalf("unable to create temp file")
	}
	defer os.Remove(file.Name())

	line := strings.Repeat("a", blockSize)
	testBytes := []byte(line + "\n" +
		line + "\n" +
		line + "\n" +
		line + "\n" +
		line[blockSize/2:]) // incomplete line

	file.Write(testBytes)
	testCases := []struct {
		name          string
		max           int64
		longerThanMax bool
		expected      string
	}{
		{
			name:          "the max is negative",
			max:           -1,
			longerThanMax: true,
			expected:      "",
		},
		{
			name:          "the max is zero",
			max:           0,
			longerThanMax: true,
			expected:      "",
		},
		{
			name:          "the file length is longer than max",
			max:           1,
			longerThanMax: true,
			expected:      "a",
		},
		{
			name:          "the file length is longer than max and contains newlines",
			max:           blockSize,
			longerThanMax: true,
			expected:      strings.Repeat("a", blockSize/2-1) + "\n" + strings.Repeat("a", blockSize/2),
		},
		{
			name:          "the max is longer than file length ",
			max:           4613,
			longerThanMax: false,
			expected:      string(testBytes),
		},
	}

	for _, test := range testCases {
		readAtMostBytes, longerThanMax, err := ReadAtMost(file.Name(), test.max)
		if err != nil {
			t.Fatalf("Unexpected failure %v", err)
		}
		if test.longerThanMax != longerThanMax {
			t.Fatalf("Unexpected result on whether the file length longer than the max, want: %t, got: %t", test.longerThanMax, longerThanMax)
		}
		if test.expected != string(readAtMostBytes) {
			t.Fatalf("Unexpected most max bytes, want: %s, got: %s", test.expected, readAtMostBytes)
		}
	}
}
