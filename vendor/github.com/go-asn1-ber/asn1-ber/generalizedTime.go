package ber

import (
	"bytes"
	"errors"
	"fmt"
	"strconv"
	"time"
)

// ErrInvalidTimeFormat is returned when the generalizedTime string was not correct.
var ErrInvalidTimeFormat = errors.New("invalid time format")

var zeroTime = time.Time{}

// ParseGeneralizedTime parses a string value and if it conforms to
// GeneralizedTime[^0] format, will return a time.Time for that value.
//
// [^0]: https://www.itu.int/rec/T-REC-X.690-201508-I/en Section 11.7
func ParseGeneralizedTime(v []byte) (time.Time, error) {
	var format string
	var fract time.Duration

	str := []byte(DecodeString(v))
	tzIndex := bytes.IndexAny(str, "Z+-")
	if tzIndex < 0 {
		return zeroTime, ErrInvalidTimeFormat
	}

	dot := bytes.IndexAny(str, ".,")
	switch dot {
	case -1:
		switch tzIndex {
		case 10:
			format = `2006010215Z`
		case 12:
			format = `200601021504Z`
		case 14:
			format = `20060102150405Z`
		default:
			return zeroTime, ErrInvalidTimeFormat
		}

	case 10, 12:
		if tzIndex < dot {
			return zeroTime, ErrInvalidTimeFormat
		}
		// a "," is also allowed, but would not be parsed by time.Parse():
		str[dot] = '.'

		// If <minute> is omitted, then <fraction> represents a fraction of an
		// hour; otherwise, if <second> and <leap-second> are omitted, then
		// <fraction> represents a fraction of a minute; otherwise, <fraction>
		// represents a fraction of a second.

		// parse as float from dot to timezone
		f, err := strconv.ParseFloat(string(str[dot:tzIndex]), 64)
		if err != nil {
			return zeroTime, fmt.Errorf("failed to parse float: %s", err)
		}
		// ...and strip that part
		str = append(str[:dot], str[tzIndex:]...)
		tzIndex = dot

		if dot == 10 {
			fract = time.Duration(int64(f * float64(time.Hour)))
			format = `2006010215Z`
		} else {
			fract = time.Duration(int64(f * float64(time.Minute)))
			format = `200601021504Z`
		}

	case 14:
		if tzIndex < dot {
			return zeroTime, ErrInvalidTimeFormat
		}
		str[dot] = '.'
		// no need for fractional seconds, time.Parse() handles that
		format = `20060102150405Z`

	default:
		return zeroTime, ErrInvalidTimeFormat
	}

	l := len(str)
	switch l - tzIndex {
	case 1:
		if str[l-1] != 'Z' {
			return zeroTime, ErrInvalidTimeFormat
		}
	case 3:
		format += `0700`
		str = append(str, []byte("00")...)
	case 5:
		format += `0700`
	default:
		return zeroTime, ErrInvalidTimeFormat
	}

	t, err := time.Parse(format, string(str))
	if err != nil {
		return zeroTime, fmt.Errorf("%s: %s", ErrInvalidTimeFormat, err)
	}
	return t.Add(fract), nil
}
