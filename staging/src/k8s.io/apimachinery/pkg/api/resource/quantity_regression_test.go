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

package resource

import (
	"fmt"
	"math"
	"strconv"
	"testing"
)

// TestMustParseMaxInt64 reproduces the bug where MustParse fails for MaxInt64.
// See Issue #135487
func TestMustParseMaxInt64(t *testing.T) {
	maxIntStr := strconv.FormatInt(math.MaxInt64, 10)

	// This function should NOT panic.
	// We use defer to catch the panic if it happens.
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("MustParse panicked with MaxInt64 input: %v", r)
		}
	}()

	q := MustParse(maxIntStr)

	// Verify the value is correct
	val, ok := q.AsInt64()
	if !ok {
		t.Errorf("AsInt64() returned error (false) for valid int64 input")
	}
	if val != math.MaxInt64 {
		t.Errorf("Expected %d, got %d", int64(math.MaxInt64), val)
	}

	fmt.Printf("Successfully parsed MaxInt64: %d\n", val)
}
