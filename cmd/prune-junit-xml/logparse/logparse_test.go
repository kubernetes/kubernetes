/*
Copyright 2024 The Kubernetes Authors.

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

package logparse

import (
	"errors"
	"fmt"
	"io"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestParse(t *testing.T) {
	t.Logf("Full regular expression:\n%s", klogPrefix.String())
	fakeErr := errors.New("fake error")

	testcases := map[string]struct {
		log           string
		err           error
		expectEntries []Entry
	}{
		"empty": {},

		"one-other": {
			log: "other",
			expectEntries: []Entry{
				&OtherEntry{Data: "other"},
			},
		},

		"one-klog": {
			log: `I1007 13:16:55.727802 1146763 example.go:57] "Key/value encoding" logger="example"`,
			expectEntries: []Entry{
				&KlogEntry{
					Data:     `I1007 13:16:55.727802 1146763 example.go:57] "Key/value encoding" logger="example"`,
					Severity: SeverityInfo,
				},
			},
		},

		"one-unit": {
			log: `example_test.go:45: E1007 13:28:21.908998] hello world`,
			expectEntries: []Entry{
				&KlogEntry{
					Data:     `example_test.go:45: E1007 13:28:21.908998] hello world`,
					Severity: SeverityError,
				},
			},
		},
		"mixture": {
			log: `other
I1007 13:16:55.727802 1146763 example.go:57] "a"
   E1007 13:16:55.727802 1146763 example.go:58] "b" foo=<
   	indented
    >
middle
example_test.go:45: E1007 13:28:21.908998] hello world
`,
			expectEntries: []Entry{
				&OtherEntry{Data: "other\n"},
				&KlogEntry{
					Data: `I1007 13:16:55.727802 1146763 example.go:57] "a"
`,
					Severity: SeverityInfo,
				},
				&KlogEntry{
					Data: `   E1007 13:16:55.727802 1146763 example.go:58] "b" foo=<
   	indented
    >
`, Severity: SeverityError,
				},
				&OtherEntry{Data: "middle\n"},
				&KlogEntry{
					Data: `example_test.go:45: E1007 13:28:21.908998] hello world
`,
					Severity: SeverityError,
				},
			},
		},
		"truncated": {
			log: `other
   E1007 13:16:55.727802 1146763 example.go:58] "b" foo=<
   	indented
middle
`,
			expectEntries: []Entry{
				&OtherEntry{Data: "other\n"},
				&KlogEntry{
					Data: `   E1007 13:16:55.727802 1146763 example.go:58] "b" foo=<
   	indented
`, Severity: SeverityError,
				},
				&OtherEntry{Data: "middle\n"},
			},
		},
		"error": {
			log: "hello\nworld",
			err: fakeErr,
			expectEntries: []Entry{
				&OtherEntry{Data: "hello\n"},
				&OtherEntry{Data: "world"},
				&ErrorEntry{Err: fakeErr},
			},
		},
		"truncated-error": {
			log: `other
   E1007 13:16:55.727802 1146763 example.go:58] "b" foo=<
   	indented`,
			err: fakeErr,
			expectEntries: []Entry{
				&OtherEntry{Data: "other\n"},
				&KlogEntry{
					Data: `   E1007 13:16:55.727802 1146763 example.go:58] "b" foo=<
   	indented`, Severity: SeverityError,
				},
				&ErrorEntry{Err: fakeErr},
			},
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			actualEntries := Parse(&fakeReader{log: tc.log, err: tc.err})
			require.Equal(t, tc.expectEntries, actualEntries)

			var buf strings.Builder
			for _, entry := range actualEntries {
				_, _ = buf.WriteString(entry.LogData())
			}
			require.Equal(t, tc.log, buf.String())

			for i := 0; i < len(tc.expectEntries); i++ {
				t.Run(fmt.Sprintf("stop-after-%d", i), func(t *testing.T) {
					var actualEntries []Entry
					seq := All(&fakeReader{log: tc.log, err: tc.err})
					e := 0
					seq(func(entry Entry) bool {
						actualEntries = append(actualEntries, entry)
						if e >= i {
							return false
						}
						e++
						return true
					})
					require.Equal(t, tc.expectEntries[0:i+1], actualEntries)
				})
			}
		})
	}
}

type fakeReader struct {
	log    string
	err    error
	offset int
}

func (f *fakeReader) Read(buf []byte) (int, error) {
	n := min(len(buf), len(f.log)-f.offset)
	copy(buf, []byte(f.log[f.offset:f.offset+n]))
	f.offset += n
	var err error
	if f.offset >= len(f.log) {
		if f.err == nil {
			err = io.EOF
		} else {
			err = f.err
		}
	}
	return n, err
}
