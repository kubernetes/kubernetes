package humanize

import (
	"math"
	"testing"
)

func TestSI(t *testing.T) {
	tests := []struct {
		name      string
		num       float64
		formatted string
	}{
		{"e-24", 1e-24, "1yF"},
		{"e-21", 1e-21, "1zF"},
		{"e-18", 1e-18, "1aF"},
		{"e-15", 1e-15, "1fF"},
		{"e-12", 1e-12, "1pF"},
		{"e-12", 2.2345e-12, "2.2345pF"},
		{"e-12", 2.23e-12, "2.23pF"},
		{"e-11", 2.23e-11, "22.3pF"},
		{"e-10", 2.2e-10, "220pF"},
		{"e-9", 2.2e-9, "2.2nF"},
		{"e-8", 2.2e-8, "22nF"},
		{"e-7", 2.2e-7, "220nF"},
		{"e-6", 2.2e-6, "2.2µF"},
		{"e-6", 1e-6, "1µF"},
		{"e-5", 2.2e-5, "22µF"},
		{"e-4", 2.2e-4, "220µF"},
		{"e-3", 2.2e-3, "2.2mF"},
		{"e-2", 2.2e-2, "22mF"},
		{"e-1", 2.2e-1, "220mF"},
		{"e+0", 2.2e-0, "2.2F"},
		{"e+0", 2.2, "2.2F"},
		{"e+1", 2.2e+1, "22F"},
		{"0", 0, "0F"},
		{"e+1", 22, "22F"},
		{"e+2", 2.2e+2, "220F"},
		{"e+2", 220, "220F"},
		{"e+3", 2.2e+3, "2.2kF"},
		{"e+3", 2200, "2.2kF"},
		{"e+4", 2.2e+4, "22kF"},
		{"e+4", 22000, "22kF"},
		{"e+5", 2.2e+5, "220kF"},
		{"e+6", 2.2e+6, "2.2MF"},
		{"e+6", 1e+6, "1MF"},
		{"e+7", 2.2e+7, "22MF"},
		{"e+8", 2.2e+8, "220MF"},
		{"e+9", 2.2e+9, "2.2GF"},
		{"e+10", 2.2e+10, "22GF"},
		{"e+11", 2.2e+11, "220GF"},
		{"e+12", 2.2e+12, "2.2TF"},
		{"e+15", 2.2e+15, "2.2PF"},
		{"e+18", 2.2e+18, "2.2EF"},
		{"e+21", 2.2e+21, "2.2ZF"},
		{"e+24", 2.2e+24, "2.2YF"},

		// special case
		{"1F", 1000 * 1000, "1MF"},
		{"1F", 1e6, "1MF"},
	}

	for _, test := range tests {
		got := SI(test.num, "F")
		if got != test.formatted {
			t.Errorf("On %v (%v), got %v, wanted %v",
				test.name, test.num, got, test.formatted)
		}

		gotf, gotu, err := ParseSI(test.formatted)
		if err != nil {
			t.Errorf("Error parsing %v (%v): %v", test.name, test.formatted, err)
			continue
		}

		if math.Abs(1-(gotf/test.num)) > 0.01 {
			t.Errorf("On %v (%v), got %v, wanted %v (±%v)",
				test.name, test.formatted, gotf, test.num,
				math.Abs(1-(gotf/test.num)))
		}
		if gotu != "F" {
			t.Errorf("On %v (%v), expected unit F, got %v",
				test.name, test.formatted, gotu)
		}
	}

	// Parse error
	gotf, gotu, err := ParseSI("x1.21JW") // 1.21 jigga whats
	if err == nil {
		t.Errorf("Expected error on x1.21JW, got %v %v", gotf, gotu)
	}
}

func BenchmarkParseSI(b *testing.B) {
	for i := 0; i < b.N; i++ {
		ParseSI("2.2346ZB")
	}
}
