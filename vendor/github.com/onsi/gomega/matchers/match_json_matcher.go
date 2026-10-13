package matchers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math/big"
	"strconv"
	"strings"

	"github.com/onsi/gomega/format"
)

type MatchJSONMatcher struct {
	JSONToMatch      any
	firstFailurePath []any
}

func (matcher *MatchJSONMatcher) Match(actual any) (success bool, err error) {
	actualString, expectedString, err := matcher.prettyPrint(actual)
	if err != nil {
		return false, err
	}

	var aval any
	var eval any

	// prettyPrint has already checked the syntax, so decoding is not expected to fail
	if aval, err = decodeJSON(actualString); err != nil {
		return false, fmt.Errorf("Actual '%s' should be valid JSON, but it is not.\nUnderlying error:%s", actualString, err)
	}
	if eval, err = decodeJSON(expectedString); err != nil {
		return false, fmt.Errorf("Expected '%s' should be valid JSON, but it is not.\nUnderlying error:%s", expectedString, err)
	}
	var equal bool
	equal, matcher.firstFailurePath = deepEqual(aval, eval)
	return equal, nil
}

// decodeJSON decodes s as json.Unmarshal would into an any, except that
// integers too large to be represented exactly by a float64 are decoded as
// canonicalJSONNumbers, so that they are compared exactly rather than after
// rounding to the nearest float64.
func decodeJSON(s string) (any, error) {
	decoder := json.NewDecoder(strings.NewReader(s))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, err
	}
	return decodeJSONNumbers(value), nil
}

func decodeJSONNumbers(value any) any {
	switch v := value.(type) {
	case []any:
		for i, element := range v {
			v[i] = decodeJSONNumbers(element)
		}
	case map[string]any:
		for key, element := range v {
			v[key] = decodeJSONNumbers(element)
		}
	case json.Number:
		return decodeJSONNumber(v)
	}
	return value
}

// canonicalJSONNumber is a JSON number written as its significant digits
// followed by a base-10 exponent, e.g. 12345678901234567890 and
// 1.234567890123456789e19 are both 123456789012345678900e1. Two JSON numbers
// have the same value exactly when they have the same canonical form.
type canonicalJSONNumber string

// maxExactFloat64Integer is 2^53: every integer with a magnitude no greater
// than this is represented exactly by a float64.
var maxExactFloat64Integer = new(big.Int).Lsh(big.NewInt(1), 53)

// decodeJSONNumber returns integers whose magnitude exceeds 2^53 in canonical
// form, and every other number as the float64 that json.Unmarshal would
// produce.  Numbers that are not integers but are too large for a float64 are
// also returned in canonical form.
func decodeJSONNumber(n json.Number) any {
	s, sign := string(n), ""
	if rest, negative := strings.CutPrefix(s, "-"); negative {
		s, sign = rest, "-"
	}
	mantissa, exponentString, _ := strings.Cut(strings.ToLower(s), "e")
	integerPart, fractionPart, _ := strings.Cut(mantissa, ".")

	digits := strings.TrimLeft(integerPart+fractionPart, "0")
	significantDigits := strings.TrimRight(digits, "0")

	// the exponent is parsed as a big.Int as JSON puts no limit on its size
	exponent := new(big.Int)
	if exponentString != "" {
		exponent.SetString(exponentString, 10) // the syntax has been checked by the decoder
	}
	exponent.Add(exponent, big.NewInt(int64(len(digits)-len(significantDigits)-len(fractionPart))))
	canonical := canonicalJSONNumber(sign + significantDigits + "e" + exponent.String())

	if significantDigits != "" && exponent.Sign() >= 0 && exceedsMaxExactFloat64Integer(significantDigits, exponent) {
		return canonical
	}
	f, err := strconv.ParseFloat(string(n), 64)
	if err != nil {
		// the syntax has been checked, so the number must be too large for a float64
		return canonical
	}
	return f
}

// exceedsMaxExactFloat64Integer reports whether significantDigits * 10^exponent,
// with exponent >= 0, is greater than 2^53.
func exceedsMaxExactFloat64Integer(significantDigits string, exponent *big.Int) bool {
	if exponent.Cmp(big.NewInt(16)) >= 0 {
		return true // 10^16 > 2^53
	}
	value, _ := new(big.Int).SetString(significantDigits, 10)
	value.Mul(value, new(big.Int).Exp(big.NewInt(10), exponent, nil))
	return value.Cmp(maxExactFloat64Integer) > 0
}

func (matcher *MatchJSONMatcher) FailureMessage(actual any) (message string) {
	actualString, expectedString, _ := matcher.prettyPrint(actual)
	return formattedMessage(format.Message(actualString, "to match JSON of", expectedString), matcher.firstFailurePath)
}

func (matcher *MatchJSONMatcher) NegatedFailureMessage(actual any) (message string) {
	actualString, expectedString, _ := matcher.prettyPrint(actual)
	return formattedMessage(format.Message(actualString, "not to match JSON of", expectedString), matcher.firstFailurePath)
}

func (matcher *MatchJSONMatcher) prettyPrint(actual any) (actualFormatted, expectedFormatted string, err error) {
	actualString, ok := toString(actual)
	if !ok {
		return "", "", fmt.Errorf("MatchJSONMatcher matcher requires a string, stringer, or []byte.  Got actual:\n%s", format.Object(actual, 1))
	}
	expectedString, ok := toString(matcher.JSONToMatch)
	if !ok {
		return "", "", fmt.Errorf("MatchJSONMatcher matcher requires a string, stringer, or []byte.  Got expected:\n%s", format.Object(matcher.JSONToMatch, 1))
	}

	abuf := new(bytes.Buffer)
	ebuf := new(bytes.Buffer)

	if err := json.Indent(abuf, []byte(actualString), "", "  "); err != nil {
		return "", "", fmt.Errorf("Actual '%s' should be valid JSON, but it is not.\nUnderlying error:%s", actualString, err)
	}

	if err := json.Indent(ebuf, []byte(expectedString), "", "  "); err != nil {
		return "", "", fmt.Errorf("Expected '%s' should be valid JSON, but it is not.\nUnderlying error:%s", expectedString, err)
	}

	return abuf.String(), ebuf.String(), nil
}
