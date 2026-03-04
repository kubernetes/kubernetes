package humanize

import (
	"strconv"
	"strings"
)

func stripTrailingZeros(s string) string {
	if !strings.ContainsRune(s, '.') {
		return s
	}
	offset := len(s) - 1
	for offset > 0 {
		if s[offset] == '.' {
			offset--
			break
		}
		if s[offset] != '0' {
			break
		}
		offset--
	}
	return s[:offset+1]
}

func stripTrailingDigits(s string, digits int) string {
	if i := strings.Index(s, "."); i >= 0 {
		if digits <= 0 {
			return s[:i]
		}
		i++
		if i+digits >= len(s) {
			return s
		}
		return s[:i+digits]
	}
	return s
}

// Ftoa converts a float to a string with no trailing zeros.
func Ftoa(num float64) string {
	return stripTrailingZeros(strconv.FormatFloat(num, 'f', 6, 64))
}

// FtoaWithDigits converts a float to a string but limits the resulting string
// to the given number of decimal places, and no trailing zeros.
func FtoaWithDigits(num float64, digits int) string {
	return stripTrailingZeros(stripTrailingDigits(strconv.FormatFloat(num, 'f', 6, 64), digits))
}
