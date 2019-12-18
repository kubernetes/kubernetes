package humanize

import (
	"errors"
	"math"
	"regexp"
	"strconv"
)

var siPrefixTable = map[float64]string{
	-24: "y", // yocto
	-21: "z", // zepto
	-18: "a", // atto
	-15: "f", // femto
	-12: "p", // pico
	-9:  "n", // nano
	-6:  "µ", // micro
	-3:  "m", // milli
	0:   "",
	3:   "k", // kilo
	6:   "M", // mega
	9:   "G", // giga
	12:  "T", // tera
	15:  "P", // peta
	18:  "E", // exa
	21:  "Z", // zetta
	24:  "Y", // yotta
}

var revSIPrefixTable = revfmap(siPrefixTable)

// revfmap reverses the map and precomputes the power multiplier
func revfmap(in map[float64]string) map[string]float64 {
	rv := map[string]float64{}
	for k, v := range in {
		rv[v] = math.Pow(10, k)
	}
	return rv
}

var riParseRegex *regexp.Regexp

func init() {
	ri := `^([\-0-9.]+)\s?([`
	for _, v := range siPrefixTable {
		ri += v
	}
	ri += `]?)(.*)`

	riParseRegex = regexp.MustCompile(ri)
}

// ComputeSI finds the most appropriate SI prefix for the given number
// and returns the prefix along with the value adjusted to be within
// that prefix.
//
// See also: SI, ParseSI.
//
// e.g. ComputeSI(2.2345e-12) -> (2.2345, "p")
func ComputeSI(input float64) (float64, string) {
	if input == 0 {
		return 0, ""
	}
	mag := math.Abs(input)
	exponent := math.Floor(logn(mag, 10))
	exponent = math.Floor(exponent/3) * 3

	value := mag / math.Pow(10, exponent)

	// Handle special case where value is exactly 1000.0
	// Should return 1 M instead of 1000 k
	if value == 1000.0 {
		exponent += 3
		value = mag / math.Pow(10, exponent)
	}

	value = math.Copysign(value, input)

	prefix := siPrefixTable[exponent]
	return value, prefix
}

// SI returns a string with default formatting.
//
// SI uses Ftoa to format float value, removing trailing zeros.
//
// See also: ComputeSI, ParseSI.
//
// e.g. SI(1000000, "B") -> 1 MB
// e.g. SI(2.2345e-12, "F") -> 2.2345 pF
func SI(input float64, unit string) string {
	value, prefix := ComputeSI(input)
	return Ftoa(value) + " " + prefix + unit
}

// SIWithDigits works like SI but limits the resulting string to the
// given number of decimal places.
//
// e.g. SIWithDigits(1000000, 0, "B") -> 1 MB
// e.g. SIWithDigits(2.2345e-12, 2, "F") -> 2.23 pF
func SIWithDigits(input float64, decimals int, unit string) string {
	value, prefix := ComputeSI(input)
	return FtoaWithDigits(value, decimals) + " " + prefix + unit
}

var errInvalid = errors.New("invalid input")

// ParseSI parses an SI string back into the number and unit.
//
// See also: SI, ComputeSI.
//
// e.g. ParseSI("2.2345 pF") -> (2.2345e-12, "F", nil)
func ParseSI(input string) (float64, string, error) {
	found := riParseRegex.FindStringSubmatch(input)
	if len(found) != 4 {
		return 0, "", errInvalid
	}
	mag := revSIPrefixTable[found[2]]
	unit := found[3]

	base, err := strconv.ParseFloat(found[1], 64)
	return base * mag, unit, err
}
