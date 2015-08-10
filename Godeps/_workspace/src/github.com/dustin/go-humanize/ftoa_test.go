package humanize

import (
	"fmt"
	"regexp"
	"strconv"
	"testing"
)

func TestFtoa(t *testing.T) {
	testList{
		{"200", Ftoa(200), "200"},
		{"2", Ftoa(2), "2"},
		{"2.2", Ftoa(2.2), "2.2"},
		{"2.02", Ftoa(2.02), "2.02"},
		{"200.02", Ftoa(200.02), "200.02"},
	}.validate(t)
}

func BenchmarkFtoaRegexTrailing(b *testing.B) {
	trailingZerosRegex := regexp.MustCompile(`\.?0+$`)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		trailingZerosRegex.ReplaceAllString("2.00000", "")
		trailingZerosRegex.ReplaceAllString("2.0000", "")
		trailingZerosRegex.ReplaceAllString("2.000", "")
		trailingZerosRegex.ReplaceAllString("2.00", "")
		trailingZerosRegex.ReplaceAllString("2.0", "")
		trailingZerosRegex.ReplaceAllString("2", "")
	}
}

func BenchmarkFtoaFunc(b *testing.B) {
	for i := 0; i < b.N; i++ {
		stripTrailingZeros("2.00000")
		stripTrailingZeros("2.0000")
		stripTrailingZeros("2.000")
		stripTrailingZeros("2.00")
		stripTrailingZeros("2.0")
		stripTrailingZeros("2")
	}
}

func BenchmarkFmtF(b *testing.B) {
	for i := 0; i < b.N; i++ {
		fmt.Sprintf("%f", 2.03584)
	}
}

func BenchmarkStrconvF(b *testing.B) {
	for i := 0; i < b.N; i++ {
		strconv.FormatFloat(2.03584, 'f', 6, 64)
	}
}
