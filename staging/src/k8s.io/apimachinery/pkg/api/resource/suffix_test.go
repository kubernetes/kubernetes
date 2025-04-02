package resource

import (
	"testing"
)

func TestDecimalExponentEdgeCases(t *testing.T) {
	table := []struct {
		input         string
		expect_base   int32
		expect_exp    int32
		expect_format Format
		expect_status bool
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
		if base != item.expect_base || exp != item.expect_exp || format != item.expect_format || item.expect_status != ok {
			t.Errorf("expected base=%v exp=%v format=%v status=%v; got base=%v exp=%v format=%v status=%v",
				item.expect_base, item.expect_exp, item.expect_format, item.expect_status,
				base, exp, format, ok)
		}
	}
}
