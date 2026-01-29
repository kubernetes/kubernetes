/*
Copyright 2014 The Kubernetes Authors.

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
	"testing"
)

func TestDecimalExponentEdgeCases(t *testing.T) {
	table := []struct {
		input        string
		expectBase   int32
		expectExp    int32
		expectFormat Format
		expectStatus bool
	}{
		{"E6024865272343", 0, 0, DecimalExponent, false},
		{"e14", 10, 14, DecimalExponent, true},
		{"E2147483647", 10, 2147483647, DecimalExponent, true},
		{"E2147483648", 0, 0, DecimalExponent, false},
		{"E-2147483648", 10, -2147483648, DecimalExponent, true},
		{"E-2147483649", 0, 0, DecimalExponent, false},
	}

	for _, item := range table {
		base, exp, format, ok := quantitySuffixer.interpret(suffix(item.input))
		if base != item.expectBase || exp != item.expectExp || format != item.expectFormat || item.expectStatus != ok {
			t.Errorf("expected base=%v exp=%v format=%v status=%v; got base=%v exp=%v format=%v status=%v",
				item.expectBase, item.expectExp, item.expectFormat, item.expectStatus,
				base, exp, format, ok)
		}
	}
}
