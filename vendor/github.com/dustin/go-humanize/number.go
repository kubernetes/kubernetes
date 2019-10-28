package humanize

/*
Slightly adapted from the source to fit go-humanize.

Author: https://github.com/gorhill
Source: https://gist.github.com/gorhill/5285193

*/

import (
	"math"
	"strconv"
)

var (
	renderFloatPrecisionMultipliers = [...]float64{
		1,
		10,
		100,
		1000,
		10000,
		100000,
		1000000,
		10000000,
		100000000,
		1000000000,
	}

	renderFloatPrecisionRounders = [...]float64{
		0.5,
		0.05,
		0.005,
		0.0005,
		0.00005,
		0.000005,
		0.0000005,
		0.00000005,
		0.000000005,
		0.0000000005,
	}
)

// FormatFloat produces a formatted number as string based on the following user-specified criteria:
// * thousands separator
// * decimal separator
// * decimal precision
//
// Usage: s := RenderFloat(format, n)
// The format parameter tells how to render the number n.
//
// See examples: http://play.golang.org/p/LXc1Ddm1lJ
//
// Examples of format strings, given n = 12345.6789:
// "#,###.##" => "12,345.67"
// "#,###." => "12,345"
// "#,###" => "12345,678"
// "#\u202F###,##" => "12 345,68"
// "#.###,###### => 12.345,678900
// "" (aka default format) => 12,345.67
//
// The highest precision allowed is 9 digits after the decimal symbol.
// There is also a version for integer number, FormatInteger(),
// which is convenient for calls within template.
func FormatFloat(format string, n float64) string {
	// Special cases:
	//   NaN = "NaN"
	//   +Inf = "+Infinity"
	//   -Inf = "-Infinity"
	if math.IsNaN(n) {
		return "NaN"
	}
	if n > math.MaxFloat64 {
		return "Infinity"
	}
	if n < -math.MaxFloat64 {
		return "-Infinity"
	}

	// default format
	precision := 2
	decimalStr := "."
	thousandStr := ","
	positiveStr := ""
	negativeStr := "-"

	if len(format) > 0 {
		format := []rune(format)

		// If there is an explicit format directive,
		// then default values are these:
		precision = 9
		thousandStr = ""

		// collect indices of meaningful formatting directives
		formatIndx := []int{}
		for i, char := range format {
			if char != '#' && char != '0' {
				formatIndx = append(formatIndx, i)
			}
		}

		if len(formatIndx) > 0 {
			// Directive at index 0:
			//   Must be a '+'
			//   Raise an error if not the case
			// index: 0123456789
			//        +0.000,000
			//        +000,000.0
			//        +0000.00
			//        +0000
			if formatIndx[0] == 0 {
				if format[formatIndx[0]] != '+' {
					panic("RenderFloat(): invalid positive sign directive")
				}
				positiveStr = "+"
				formatIndx = formatIndx[1:]
			}

			// Two directives:
			//   First is thousands separator
			//   Raise an error if not followed by 3-digit
			// 0123456789
			// 0.000,000
			// 000,000.00
			if len(formatIndx) == 2 {
				if (formatIndx[1] - formatIndx[0]) != 4 {
					panic("RenderFloat(): thousands separator directive must be followed by 3 digit-specifiers")
				}
				thousandStr = string(format[formatIndx[0]])
				formatIndx = formatIndx[1:]
			}

			// One directive:
			//   Directive is decimal separator
			//   The number of digit-specifier following the separator indicates wanted precision
			// 0123456789
			// 0.00
			// 000,0000
			if len(formatIndx) == 1 {
				decimalStr = string(format[formatIndx[0]])
				precision = len(format) - formatIndx[0] - 1
			}
		}
	}

	// generate sign part
	var signStr string
	if n >= 0.000000001 {
		signStr = positiveStr
	} else if n <= -0.000000001 {
		signStr = negativeStr
		n = -n
	} else {
		signStr = ""
		n = 0.0
	}

	// split number into integer and fractional parts
	intf, fracf := math.Modf(n + renderFloatPrecisionRounders[precision])

	// generate integer part string
	intStr := strconv.FormatInt(int64(intf), 10)

	// add thousand separator if required
	if len(thousandStr) > 0 {
		for i := len(intStr); i > 3; {
			i -= 3
			intStr = intStr[:i] + thousandStr + intStr[i:]
		}
	}

	// no fractional part, we can leave now
	if precision == 0 {
		return signStr + intStr
	}

	// generate fractional part
	fracStr := strconv.Itoa(int(fracf * renderFloatPrecisionMultipliers[precision]))
	// may need padding
	if len(fracStr) < precision {
		fracStr = "000000000000000"[:precision-len(fracStr)] + fracStr
	}

	return signStr + intStr + decimalStr + fracStr
}

// FormatInteger produces a formatted number as string.
// See FormatFloat.
func FormatInteger(format string, n int) string {
	return FormatFloat(format, float64(n))
}
