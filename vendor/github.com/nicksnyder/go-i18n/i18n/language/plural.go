package language

import (
	"fmt"
)

// Plural represents a language pluralization form as defined here:
// http://cldr.unicode.org/index/cldr-spec/plural-rules
type Plural string

// All defined plural categories.
const (
	Invalid Plural = "invalid"
	Zero           = "zero"
	One            = "one"
	Two            = "two"
	Few            = "few"
	Many           = "many"
	Other          = "other"
)

// NewPlural returns src as a Plural
// or Invalid and a non-nil error if src is not a valid Plural.
func NewPlural(src string) (Plural, error) {
	switch src {
	case "zero":
		return Zero, nil
	case "one":
		return One, nil
	case "two":
		return Two, nil
	case "few":
		return Few, nil
	case "many":
		return Many, nil
	case "other":
		return Other, nil
	}
	return Invalid, fmt.Errorf("invalid plural category %s", src)
}
