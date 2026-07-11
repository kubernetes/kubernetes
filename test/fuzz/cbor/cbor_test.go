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

package cbor_test

import (
	"testing"

	"k8s.io/kubernetes/test/fuzz/cbor"
)

// FuzzDecodeAllocations wraps the FuzzDecodeAllocations go-fuzz target as a "go test" fuzz test.
func FuzzDecodeAllocations(f *testing.F) {
	f.Add([]byte("\xa2\x4aapiVersion\x41x\x44kind\x41y")) // {'apiVersion': 'x', 'kind': 'y'}
	f.Fuzz(func(t *testing.T, in []byte) {
		defer func() {
			if p := recover(); p != nil {
				t.Fatal(p)
			}
		}()
		cbor.FuzzDecodeAllocations(in)
	})
}
