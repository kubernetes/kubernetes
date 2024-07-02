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

package direct_test

import (
	"testing"

	"k8s.io/apimachinery/pkg/runtime/serializer/cbor/direct"
)

func TestSniff(t *testing.T) {
	for _, tc := range []struct {
		In      []byte
		CBOR    bool
		Unknown bool
	}{
		{In: []byte{0xd9, 0xd9, 0xf7}, CBOR: true, Unknown: true},
		{In: []byte{}, CBOR: false, Unknown: false},
		{In: []byte{}, CBOR: false, Unknown: false},
		{In: []byte{0xff}, CBOR: false, Unknown: false},
		{In: []byte{0xa0}, CBOR: true, Unknown: false},
		{In: append([]byte{0x9f}, make([]byte, 100)...), CBOR: true, Unknown: true},
	} {
		cbor, unknown := direct.Sniff(tc.In)
		if cbor != tc.CBOR || unknown != tc.Unknown {
			t.Errorf("0x%x: got (%t, %t) want (%t, %t)", tc.In, cbor, unknown, tc.CBOR, tc.Unknown)
		}
	}
}
