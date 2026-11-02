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

package term

import (
	"bytes"
	"strings"
	"testing"
)

const test = "Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube Kube"

func TestWordWrapWriter(t *testing.T) {
	testcases := map[string]struct {
		input    string
		maxWidth uint
	}{
		"max 10":   {input: test, maxWidth: 10},
		"max 80":   {input: test, maxWidth: 80},
		"max 120":  {input: test, maxWidth: 120},
		"max 5000": {input: test, maxWidth: 5000},
	}
	for k, tc := range testcases {
		b := bytes.NewBufferString("")
		w := NewWordWrapWriter(b, tc.maxWidth)
		_, err := w.Write([]byte(tc.input))
		if err != nil {
			t.Errorf("%s: Unexpected error: %v", k, err)
		}
		result := b.String()
		if !strings.Contains(result, "Kube") {
			t.Errorf("%s: Expected to contain \"Kube\"", k)
		}
		if len(result) < len(tc.input) {
			t.Errorf("%s: Unexpectedly short string, got %d wanted at least %d chars: %q", k, len(result), len(tc.input), result)
		}
		for _, line := range strings.Split(result, "\n") {
			if len(line) > int(tc.maxWidth) {
				t.Errorf("%s: Every line must be at most %d chars long, got %d: %q", k, tc.maxWidth, len(line), line)
			}
		}
		for _, word := range strings.Split(result, " ") {
			if !strings.Contains(word, "Kube") {
				t.Errorf("%s: Unexpected broken word: %q", k, word)
			}
		}
	}
}

func TestMaxWidthWriter(t *testing.T) {
	testcases := map[string]struct {
		input    string
		maxWidth uint
	}{
		"max 10":   {input: test, maxWidth: 10},
		"max 80":   {input: test, maxWidth: 80},
		"max 120":  {input: test, maxWidth: 120},
		"max 5000": {input: test, maxWidth: 5000},
	}
	for k, tc := range testcases {
		b := bytes.NewBufferString("")
		w := NewMaxWidthWriter(b, tc.maxWidth)
		_, err := w.Write([]byte(tc.input))
		if err != nil {
			t.Errorf("%s: Unexpected error: %v", k, err)
		}
		result := b.String()
		if !strings.Contains(result, "Kube") {
			t.Errorf("%s: Expected to contain \"Kube\"", k)
		}
		if len(result) < len(tc.input) {
			t.Errorf("%s: Unexpectedly short string, got %d wanted at least %d chars: %q", k, len(result), len(tc.input), result)
		}
		lines := strings.Split(result, "\n")
		for i, line := range lines {
			if len(line) > int(tc.maxWidth) {
				t.Errorf("%s: Every line must be at most %d chars long, got %d: %q", k, tc.maxWidth, len(line), line)
			}
			if i < len(lines)-1 && len(line) != int(tc.maxWidth) {
				t.Errorf("%s: Lines except the last one are expected to be exactly %d chars long, got %d: %q", k, tc.maxWidth, len(line), line)
			}
		}
	}
}

func TestMaxWidthWriterNewlineAccounting(t *testing.T) {
	testcases := map[string]struct {
		input    string
		maxWidth uint
		want     string
	}{
		"embedded newline resets the column": {
			input:    "ab\ncdefg",
			maxWidth: 5,
			want:     "ab\ncdefg",
		},
		"newline after a full line does not add a blank line": {
			input:    "12345\nabc",
			maxWidth: 5,
			want:     "12345\nabc",
		},
	}
	for k, tc := range testcases {
		b := bytes.NewBufferString("")
		w := NewMaxWidthWriter(b, tc.maxWidth)
		if _, err := w.Write([]byte(tc.input)); err != nil {
			t.Errorf("%s: Unexpected error: %v", k, err)
		}
		if got := b.String(); got != tc.want {
			t.Errorf("%s: got %q, want %q", k, got, tc.want)
		}
	}
}

func TestMaxWidthWriterAcrossWrites(t *testing.T) {
	b := bytes.NewBufferString("")
	w := NewMaxWidthWriter(b, 5)
	for _, chunk := range []string{"abc", "defgh"} {
		if _, err := w.Write([]byte(chunk)); err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
	}
	if got, want := b.String(), "abcde\nfgh"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}
