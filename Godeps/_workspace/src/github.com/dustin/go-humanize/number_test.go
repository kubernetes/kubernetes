package humanize

import (
	"math"
	"testing"
)

type TestStruct struct {
	name      string
	format    string
	num       float64
	formatted string
}

func TestFormatFloat(t *testing.T) {
	tests := []TestStruct{
		{"default", "", 12345.6789, "12,345.68"},
		{"#", "#", 12345.6789, "12345.678900000"},
		{"#.", "#.", 12345.6789, "12346"},
		{"#,#", "#,#", 12345.6789, "12345,7"},
		{"#,##", "#,##", 12345.6789, "12345,68"},
		{"#,###", "#,###", 12345.6789, "12345,679"},
		{"#,###.", "#,###.", 12345.6789, "12,346"},
		{"#,###.##", "#,###.##", 12345.6789, "12,345.68"},
		{"#,###.###", "#,###.###", 12345.6789, "12,345.679"},
		{"#,###.####", "#,###.####", 12345.6789, "12,345.6789"},
		{"#.###,######", "#.###,######", 12345.6789, "12.345,678900"},
		{"#\u202f###,##", "#\u202f###,##", 12345.6789, "12 345,68"},

		// special cases
		{"NaN", "#", math.NaN(), "NaN"},
		{"+Inf", "#", math.Inf(1), "Infinity"},
		{"-Inf", "#", math.Inf(-1), "-Infinity"},
		{"signStr <= -0.000000001", "", -0.000000002, "-0.00"},
		{"signStr = 0", "", 0, "0.00"},
		{"Format directive must start with +", "+000", 12345.6789, "+12345.678900000"},
	}

	for _, test := range tests {
		got := FormatFloat(test.format, test.num)
		if got != test.formatted {
			t.Errorf("On %v (%v, %v), got %v, wanted %v",
				test.name, test.format, test.num, got, test.formatted)
		}
	}
	// Test a single integer
	got := FormatInteger("#", 12345)
	if got != "12345.000000000" {
		t.Errorf("On %v (%v, %v), got %v, wanted %v",
			"integerTest", "#", 12345, got, "12345.000000000")
	}
	// Test the things that could panic
	panictests := []TestStruct{
		{"RenderFloat(): invalid positive sign directive", "-", 12345.6789, "12,345.68"},
		{"RenderFloat(): thousands separator directive must be followed by 3 digit-specifiers", "0.01", 12345.6789, "12,345.68"},
	}
	for _, test := range panictests {
		didPanic := false
		var message interface{}
		func() {

			defer func() {
				if message = recover(); message != nil {
					didPanic = true
				}
			}()

			// call the target function
			_ = FormatFloat(test.format, test.num)

		}()
		if didPanic != true {
			t.Errorf("On %v, should have panic and did not.",
				test.name)
		}
	}

}
