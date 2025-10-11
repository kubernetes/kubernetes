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

package api

import "testing"

// BenchmarkUniqueStrString demonstrates that retrieving the string in a unique string triggers no memory allocation.
func BenchmarkUniqueStrString(b *testing.B) {
	expect := "hello-world"
	u := MakeUniqueString(expect)
	var actual string
	for b.Loop() {
		actual = u.String()
	}
	if expect != actual {
		b.Fatalf("expected %q, got %q", expect, actual)
	}
}
