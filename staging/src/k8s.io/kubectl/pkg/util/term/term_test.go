/*
Copyright The Kubernetes Authors.

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
	"errors"
	"io"
	"strings"
	"testing"
)

// errorReader is a helper that always returns a fixed error.
type errorReader struct {
	err error
}

func (e *errorReader) Read(p []byte) (n int, err error) {
	return 0, e.err
}

func TestDetachableReader(t *testing.T) {
	tests := []struct {
		name       string
		detachKeys string
		input      string
		bufSize    int
		wantN      int
		wantErr    error
		wantData   string
	}{
		{
			name:       "normal read without detach sequence",
			detachKeys: "ctrl-p,ctrl-q",
			input:      "hello world this is a test",
			bufSize:    20,
			wantN:      20,
			wantErr:    nil,
			wantData:   "hello world this is ",
		},
		{
			name:       "detach sequence triggers EOF",
			detachKeys: "ctrl-p,ctrl-q",
			input:      "data before\x10\x11data after",
			bufSize:    30,
			wantN:      11,
			wantErr:    io.EOF,
			wantData:   "data before",
		},
		{
			name:       "partial detach sequence is treated as normal data",
			detachKeys: "ctrl-p,ctrl-q",
			input:      "normal text\x10only ctrl-p no q",
			bufSize:    30,
			wantN:      28,
			wantErr:    nil,
			wantData:   "normal text\x10only ctrl-p no q",
		},
		{
			name:       "underlying error is passed through unchanged",
			detachKeys: "ctrl-a",
			input:      "", // dummy, error comes from mock
			bufSize:    10,
			wantN:      0,
			wantErr:    errors.New("simulated io error"),
			wantData:   "",
		},
		{
			name:       "detach sequence exactly at buffer boundary",
			detachKeys: "ctrl-a",
			input:      "short\x01rest",
			bufSize:    6,
			wantN:      5,
			wantErr:    io.EOF,
			wantData:   "short",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var r io.Reader
			var err error

			if tt.wantErr != nil && !errors.Is(tt.wantErr, io.EOF) {
				// Special case: simulate underlying read error
				r = &errorReader{err: tt.wantErr}
			} else {
				r, err = NewDetachableReader(strings.NewReader(tt.input), tt.detachKeys)
				if err != nil {
					t.Fatalf("NewDetachableReader failed: %v", err)
				}
			}

			buf := make([]byte, tt.bufSize)
			n, readErr := r.Read(buf)

			if n != tt.wantN {
				t.Errorf("Read() got n = %d, want %d", n, tt.wantN)
			}

			if !errors.Is(readErr, tt.wantErr) {
				t.Errorf("Read() got err = %v, want %v", readErr, tt.wantErr)
			}

			if tt.wantN > 0 {
				got := string(buf[:tt.wantN])
				if got != tt.wantData {
					t.Errorf("Read() got data = %q, want %q", got, tt.wantData)
				}
			}
		})
	}
}
