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
