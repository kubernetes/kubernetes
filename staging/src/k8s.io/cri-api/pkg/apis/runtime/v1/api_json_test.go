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

package v1

import (
	"bytes"
	"encoding/json"
	"testing"
)

func TestKeyValueCompat(t *testing.T) {
	testcases := []struct {
		name                 string
		envs                 []*KeyValue
		variantJSON          string
		expectedJSON         string
		expectedRoundTripped []*KeyValue
	}{
		{
			name:                 "null",
			envs:                 nil,
			expectedJSON:         `null`,
			expectedRoundTripped: nil,
		},
		{
			name:                 "zero-length list",
			envs:                 []*KeyValue{},
			expectedJSON:         `[]`,
			expectedRoundTripped: []*KeyValue{},
		},
		{
			name:                 "zero-value env",
			envs:                 []*KeyValue{{}},
			expectedJSON:         `[{}]`,
			expectedRoundTripped: []*KeyValue{{}},
		},
		{
			name:                 "ascii env",
			envs:                 []*KeyValue{{Key: "key", Value: []byte("value")}},
			expectedJSON:         `[{"key":"key","value":"value"}]`,
			expectedRoundTripped: []*KeyValue{{Key: "key", Value: []byte("value")}},
		},
		{
			name:                 "utf8 env",
			envs:                 []*KeyValue{{Key: "key", Value: []byte("Iñtërnâtiônàlizætiøn🐹")}},
			expectedJSON:         `[{"key":"key","value":"Iñtërnâtiônàlizætiøn🐹"}]`,
			expectedRoundTripped: []*KeyValue{{Key: "key", Value: []byte("Iñtërnâtiônàlizætiøn🐹")}},
		},
		{
			name:                 "non-utf8 env",
			envs:                 []*KeyValue{{Key: "key", Value: []byte{'A', 0x80, 'Z'}}},              // invalid utf8 continuation byte (0x80)
			variantJSON:          `[{"key":"key","value":"A` + "\x80" + `Z"}]`,                          // an alternate JSON input containing the invalid utf8 byte that should coerce to the same result
			expectedJSON:         `[{"key":"key","value":"A` + stdlibSerializedReplacementChar + `Z"}]`, // coerced to utf8 replacement character (\ufffd) on marshal
			expectedRoundTripped: []*KeyValue{{Key: "key", Value: []byte("A\ufffdZ")}},                  // round-trips to replacement character (\ufffd) on unmarshal
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			data, err := json.Marshal(tc.envs)
			if err != nil {
				t.Fatal(err)
			}

			if string(data) != tc.expectedJSON {
				t.Fatalf("json differed:\nwant: %s\ngot:  %s", tc.expectedJSON, string(data))
			}

			verifyJSON(t, data, tc.expectedRoundTripped)
			if len(tc.variantJSON) > 0 {
				verifyJSON(t, []byte(tc.variantJSON), tc.expectedRoundTripped)
			}
		})
	}
}

func verifyJSON(t *testing.T, data []byte, expectedRoundTripped []*KeyValue) {
	t.Helper()

	var rt []*KeyValue
	if err := json.Unmarshal(data, &rt); err != nil {
		t.Fatal(err)
	}
	if (rt == nil) != (expectedRoundTripped == nil) {
		t.Fatalf("expected value (%#v) does not match actual round-tripped value (%#v) for nil", expectedRoundTripped, rt)
	}
	if rt == nil {
		return
	}
	if len(rt) != len(expectedRoundTripped) {
		t.Fatalf("length of expected value (%#v) does not match length of actual round-tripped value (%#v)", expectedRoundTripped, rt)
	}
	for i := range expectedRoundTripped {
		if want, got := expectedRoundTripped[i].Key, rt[i].Key; want != got {
			t.Fatalf("item[%d].key does not match: %s vs %s", i, want, got)
		}
		if want, got := expectedRoundTripped[i].Value, rt[i].Value; !bytes.Equal(want, got) {
			t.Fatalf("item[%d].value does not match: %v vs %v", i, want, got)
		}
	}
}
