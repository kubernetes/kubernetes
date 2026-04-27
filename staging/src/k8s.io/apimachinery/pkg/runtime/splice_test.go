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

package runtime_test

import (
	"bytes"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
)

func TestSpliceBuffer(t *testing.T) {
	testBytes0 := []byte{0x01, 0x02, 0x03, 0x04}
	testBytes1 := []byte{0x04, 0x03, 0x02, 0x02}

	testCases := []struct {
		name string
		run  func(sb runtime.Splice, buf *bytes.Buffer)
	}{
		{
			name: "Basic Write",
			run: func(sb runtime.Splice, buf *bytes.Buffer) {
				sb.Write(testBytes0)
				buf.Write(testBytes0)
			},
		},
		{
			name: "Multiple Writes",
			run: func(sb runtime.Splice, buf *bytes.Buffer) {
				for _, b := range testBytes0 {
					sb.Write([]byte{b})
					buf.Write([]byte{b})
				}
			},
		},
		{
			name: "Write and Reset",
			run: func(sb runtime.Splice, buf *bytes.Buffer) {
				sb.Write(testBytes0)
				buf.Write(testBytes0)

				sb.Reset()
				buf.Reset()
			},
		},
		{
			name: "Write/Splice",
			run: func(sb runtime.Splice, buf *bytes.Buffer) {
				sb.Splice(testBytes0)
				buf.Write(testBytes0)
			},
		},
		{
			name: "Write/Splice and Reset",
			run: func(sb runtime.Splice, buf *bytes.Buffer) {
				sb.Splice(testBytes0)
				buf.Write(testBytes0)

				sb.Reset()
				buf.Reset()
			},
		},
		{
			name: "Write/Splice, Reset, Write/Splice",
			run: func(sb runtime.Splice, buf *bytes.Buffer) {
				sb.Splice(testBytes0)
				buf.Write(testBytes0)

				sb.Reset()
				buf.Reset()

				sb.Splice(testBytes1)
				buf.Write(testBytes1)
			},
		},
		{
			name: "Write, Reset, Splice",
			run: func(sb runtime.Splice, buf *bytes.Buffer) {
				sb.Write(testBytes0)
				buf.Write(testBytes0)

				sb.Reset()
				buf.Reset()

				sb.Splice(testBytes1)
				buf.Write(testBytes1)
			},
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			sb := runtime.NewSpliceBuffer()
			buf := &bytes.Buffer{}
			tt.run(sb, buf)

			if sb.Bytes() == nil {
				t.Errorf("Unexpected nil")
			}
			if string(sb.Bytes()) != string(buf.Bytes()) {
				t.Errorf("Expected sb.Bytes() == %q, buf.Bytes() == %q", sb.Bytes(), buf.Bytes())
			}
		})

	}

}
