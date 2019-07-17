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
