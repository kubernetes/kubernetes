/*
Copyright 2025 The Kubernetes Authors.

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

package modes

import (
	"errors"
	"strings"
	"testing"
)

func TestTranscodeFromJSON(t *testing.T) {
	for _, tc := range []struct {
		name string
		json string
		cbor string
		err  error
	}{
		{
			name: "whole number",
			json: "42",
			cbor: "\x18\x2a",
		},
		{
			name: "decimal number",
			json: "1.5",
			cbor: "\xf9\x3e\x00",
		},
		{
			name: "false",
			json: "false",
			cbor: "\xf4",
		},
		{
			name: "true",
			json: "true",
			cbor: "\xf5",
		},
		{
			name: "null",
			json: "null",
			cbor: "\xf6",
		},
		{
			name: "string",
			json: `"foo"`,
			cbor: "\x43foo",
		},
		{
			name: "array",
			json: `[]`,
			cbor: "\x80",
		},
		{
			name: "object",
			json: `{"foo":"bar"}`,
			cbor: "\xa1\x43foo\x43bar",
		},
		{
			name: "extraneous data",
			json: "{}{}",
			err:  errors.New("extraneous data"),
		},
		{
			name: "eof",
			json: "",
			err:  errors.New("EOF"),
		},
		{
			name: "unexpected eof",
			json: "{",
			err:  errors.New("unexpected EOF"),
		},
		{
			name: "malformed json",
			json: "}",
			err:  errors.New("invalid character '}' looking for beginning of value"),
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var out strings.Builder
			err := TranscodeFromJSON(&out, strings.NewReader(tc.json))
			if (err == nil) != (tc.err == nil) || tc.err != nil && tc.err.Error() != err.Error() {
				t.Fatalf("unexpected error: want %v got %v", tc.err, err)
			}
			if got, want := out.String(), tc.cbor; got != want {
				t.Errorf("unexpected transcoding: want 0x%x got 0x%x", want, got)
			}
		})
	}
}

func TestTranscodeToJSON(t *testing.T) {
	for _, tc := range []struct {
		name string
		cbor string
		json string
		err  error
	}{
		{
			name: "whole number",
			cbor: "\x18\x2a",
			json: "42",
		},
		{
			name: "decimal number",
			cbor: "\xf9\x3e\x00",
			json: "1.5",
		},
		{
			name: "false",
			cbor: "\xf4",
			json: "false",
		},
		{
			name: "true",
			cbor: "\xf5",
			json: "true",
		},
		{
			name: "null",
			cbor: "\xf6",
			json: "null",
		},
		{
			name: "string",
			cbor: "\x43foo",
			json: `"foo"`,
		},
		{
			name: "array",
			cbor: "\x80",
			json: `[]`,
		},
		{
			name: "object",
			cbor: "\xa1\x43foo\x43bar",
			json: `{"foo":"bar"}`,
		},
		{
			name: "extraneous data",
			cbor: "\xa0\xa0",
			err:  errors.New("extraneous data"),
		},
		{
			name: "unexpected eof",
			cbor: "\xa1",
			err:  errors.New("unexpected EOF"),
		},
		{
			name: "malformed cbor",
			cbor: "\xff",
			err:  errors.New(`cbor: unexpected "break" code`),
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var out strings.Builder
			err := TranscodeToJSON(&out, strings.NewReader(tc.cbor))
			if (err == nil) != (tc.err == nil) || tc.err != nil && tc.err.Error() != err.Error() {
				t.Fatalf("unexpected error: want %v got %v", tc.err, err)
			}
			if got, want := out.String(), tc.json; got != want {
				t.Errorf("unexpected transcoding: want %q got %q", want, got)
			}
		})
	}
}

type write struct {
	p   string
	n   int
	err error
}

type mockWriter struct {
	t     testing.TB
	calls []write
}

func (m *mockWriter) Write(p []byte) (int, error) {
	if len(m.calls) == 0 {
		m.t.Fatalf("unexpected call (p=%q)", string(p))
	}
	if got, want := string(p), m.calls[0].p; got != want {
		m.t.Errorf("unexpected argument: want %q, got %q", want, got)
	}
	n, err := m.calls[0].n, m.calls[0].err
	m.calls = m.calls[1:]
	return n, err
}

func TestTrailingLinefeedSuppressor(t *testing.T) {
	for _, tc := range []struct {
		name      string
		calls     []write
		delegated []write
	}{
		{
			name:      "one write without newline",
			calls:     []write{{"foo", 3, nil}},
			delegated: []write{{"foo", 3, nil}},
		},
		{
			name:      "one write with newline",
			calls:     []write{{"foo\n", 4, nil}},
			delegated: []write{{"foo", 3, nil}},
		},
		{
			name:      "one write with only newline",
			calls:     []write{{"\n", 1, nil}},
			delegated: nil,
		},
		{
			name:      "one empty write",
			calls:     []write{{"", 0, nil}},
			delegated: nil,
		},
		{
			name:      "three writes, all with only newline",
			calls:     []write{{"\n", 1, nil}, {"\n", 1, nil}, {"\n", 1, nil}},
			delegated: []write{{"\n", 1, nil}, {"\n", 1, nil}},
		},
		{
			name:      "buffered linefeed not flushed on empty write",
			calls:     []write{{"\n", 1, nil}, {"", 0, nil}},
			delegated: nil,
		},

		{
			name:      "two writes, last with trailing newline",
			calls:     []write{{"foo", 3, nil}, {"bar\n", 4, nil}},
			delegated: []write{{"foo", 3, nil}, {"bar", 3, nil}},
		},
		{
			name:      "two writes, first with trailing newline",
			calls:     []write{{"foo\n", 4, nil}, {"bar", 3, nil}},
			delegated: []write{{"foo", 3, nil}, {"\n", 1, nil}, {"bar", 3, nil}},
		},
		{
			name:      "two writes, both with trailing newlines",
			calls:     []write{{"foo\n", 4, nil}, {"bar\n", 4, nil}},
			delegated: []write{{"foo", 3, nil}, {"\n", 1, nil}, {"bar", 3, nil}},
		},
		{
			name:      "two writes, neither with newlines",
			calls:     []write{{"foo", 3, nil}, {"bar", 3, nil}},
			delegated: []write{{"foo", 3, nil}, {"bar", 3, nil}},
		},
		{
			name:      "delegate error before reaching newline",
			calls:     []write{{"foo\n", 1, errors.New("test")}, {"oo\n", 3, nil}},
			delegated: []write{{"foo", 1, errors.New("test")}, {"oo", 2, nil}},
		},
		{
			name:      "delegate error after reaching newline",
			calls:     []write{{"foo\n", 4, errors.New("test")}},
			delegated: []write{{"foo", 3, errors.New("test")}},
		},
		{
			name:      "delegate error flushing newline",
			calls:     []write{{"foo\n", 4, nil}, {"\n", 0, errors.New("test")}, {"\n", 1, nil}},
			delegated: []write{{"foo", 3, nil}, {"\n", 0, errors.New("test")}, {"\n", 1, nil}},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := &trailingLinefeedSuppressor{delegate: &mockWriter{t: t, calls: tc.delegated}}
			for _, call := range tc.calls {
				n, err := w.Write([]byte(call.p))
				if n != call.n {
					t.Errorf("unexpected n: want %d got %d", call.n, n)
				}
				if (err == nil) != (call.err == nil) || call.err != nil && call.err.Error() != err.Error() {
					t.Errorf("unexpected error: want %v got %v", call.err, err)
				}
			}
		})
	}
}
