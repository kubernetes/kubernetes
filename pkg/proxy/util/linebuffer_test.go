/*
Copyright 2023 The Kubernetes Authors.

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

package util

import (
	"math/rand"
	"strings"
	"testing"
)

func TestLineBufferWrite(t *testing.T) {
	testCases := []struct {
		name     string
		input    []interface{}
		expected string
	}{
		{
			name:     "none",
			input:    []interface{}{},
			expected: "\n",
		},
		{
			name:     "one string",
			input:    []interface{}{"test1"},
			expected: "test1\n",
		},
		{
			name:     "one slice",
			input:    []interface{}{[]string{"test1", "test2"}},
			expected: "test1 test2\n",
		},
		{
			name:     "mixed",
			input:    []interface{}{"s1", "s2", []string{"s3", "s4"}, "", "s5", []string{}, []string{"s6"}, "s7"},
			expected: "s1 s2 s3 s4  s5  s6 s7\n",
		},
	}
	testBuffer := NewLineBuffer()
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			testBuffer.Reset()
			testBuffer.Write(testCase.input...)
			if want, got := testCase.expected, testBuffer.String(); !strings.EqualFold(want, got) {
				t.Fatalf("write word is %v\n expected: %q, got: %q", testCase.input, want, got)
			}
			if testBuffer.Lines() != 1 {
				t.Fatalf("expected 1 line, got: %d", testBuffer.Lines())
			}
		})
	}
}

func TestLineBufferWritePanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("did not panic")
		}
	}()
	testBuffer := NewLineBuffer()
	testBuffer.Write("string", []string{"a", "slice"}, 1234)
}

func TestLineBufferWriteBytes(t *testing.T) {
	testCases := []struct {
		name     string
		bytes    []byte
		expected string
	}{
		{
			name:     "empty bytes",
			bytes:    []byte{},
			expected: "\n",
		},
		{
			name:     "test bytes",
			bytes:    []byte("test write bytes line"),
			expected: "test write bytes line\n",
		},
	}

	testBuffer := NewLineBuffer()
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			testBuffer.Reset()
			testBuffer.WriteBytes(testCase.bytes)
			if want, got := testCase.expected, testBuffer.String(); !strings.EqualFold(want, got) {
				t.Fatalf("write bytes is %v\n expected: %s, got: %s", testCase.bytes, want, got)
			}
		})
	}
}

// obtained from https://stackoverflow.com/a/22892986
var letters = []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

func randSeq() string {
	b := make([]rune, 30)
	for i := range b {
		b[i] = letters[rand.Intn(len(letters))]
	}
	return string(b)
}

func TestWriteCountLines(t *testing.T) {
	testCases := []struct {
		name     string
		expected int
	}{
		{
			name:     "write no line",
			expected: 0,
		},
		{
			name:     "write one line",
			expected: 1,
		},
		{
			name:     "write 100 lines",
			expected: 100,
		},
		{
			name:     "write 1000 lines",
			expected: 1000,
		},
		{
			name:     "write 10000 lines",
			expected: 10000,
		},
		{
			name:     "write 100000 lines",
			expected: 100000,
		},
	}
	testBuffer := NewLineBuffer()
	discardBuffer := NewDiscardLineBuffer()
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			testBuffer.Reset()
			discardBuffer.Reset()
			for i := 0; i < testCase.expected; i++ {
				testBuffer.Write(randSeq())
				discardBuffer.Write(randSeq())
			}
			n := testBuffer.Lines()
			if n != testCase.expected {
				t.Fatalf("lines expected: %d, got: %d", testCase.expected, n)
			}
			n = discardBuffer.Lines()
			if n != testCase.expected {
				t.Fatalf("discardBuffer lines expected: %d, got: %d", testCase.expected, n)
			}
		})
	}
}
