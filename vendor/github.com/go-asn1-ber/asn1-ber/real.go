package ber

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
)

func encodeFloat(v float64) []byte {
	switch {
	case math.IsInf(v, 1):
		return []byte{0x40}
	case math.IsInf(v, -1):
		return []byte{0x41}
	case math.IsNaN(v):
		return []byte{0x42}
	case v == 0.0:
		if math.Signbit(v) {
			return []byte{0x43}
		}
		return []byte{}
	default:
		// we take the easy part ;-)
		value := []byte(strconv.FormatFloat(v, 'G', -1, 64))
		var ret []byte
		if bytes.Contains(value, []byte{'E'}) {
			ret = []byte{0x03}
		} else {
			ret = []byte{0x02}
		}
		ret = append(ret, value...)
		return ret
	}
}

func ParseReal(v []byte) (val float64, err error) {
	if len(v) == 0 {
		return 0.0, nil
	}
	switch {
	case v[0]&0x80 == 0x80:
		val, err = parseBinaryFloat(v)
	case v[0]&0xC0 == 0x40:
		val, err = parseSpecialFloat(v)
	case v[0]&0xC0 == 0x0:
		val, err = parseDecimalFloat(v)
	default:
		return 0.0, fmt.Errorf("invalid info block")
	}
	if err != nil {
		return 0.0, err
	}

	if val == 0.0 && !math.Signbit(val) {
		return 0.0, errors.New("REAL value +0 must be encoded with zero-length value block")
	}
	return val, nil
}

func parseBinaryFloat(v []byte) (float64, error) {
	var info byte
	var buf []byte

	info, v = v[0], v[1:]

	var base int
	switch info & 0x30 {
	case 0x00:
		base = 2
	case 0x10:
		base = 8
	case 0x20:
		base = 16
	case 0x30:
		return 0.0, errors.New("bits 6 and 5 of information octet for REAL are equal to 11")
	}

	scale := uint((info & 0x0c) >> 2)

	var expLen int
	switch info & 0x03 {
	case 0x00:
		expLen = 1
	case 0x01:
		expLen = 2
	case 0x02:
		expLen = 3
	case 0x03:
		expLen = int(v[0])
		if expLen > 8 {
			return 0.0, errors.New("too big value of exponent")
		}
		v = v[1:]
	}
	buf, v = v[:expLen], v[expLen:]
	exponent, err := ParseInt64(buf)
	if err != nil {
		return 0.0, err
	}

	if len(v) > 8 {
		return 0.0, errors.New("too big value of mantissa")
	}

	mant, err := ParseInt64(v)
	if err != nil {
		return 0.0, err
	}
	mantissa := mant << scale

	if info&0x40 == 0x40 {
		mantissa = -mantissa
	}

	return float64(mantissa) * math.Pow(float64(base), float64(exponent)), nil
}

func parseDecimalFloat(v []byte) (val float64, err error) {
	switch v[0] & 0x3F {
	case 0x01: // NR form 1
		var iVal int64
		iVal, err = strconv.ParseInt(strings.TrimLeft(string(v[1:]), " "), 10, 64)
		val = float64(iVal)
	case 0x02, 0x03: // NR form 2, 3
		val, err = strconv.ParseFloat(strings.Replace(strings.TrimLeft(string(v[1:]), " "), ",", ".", -1), 64)
	default:
		err = errors.New("incorrect NR form")
	}
	if err != nil {
		return 0.0, err
	}

	if val == 0.0 && math.Signbit(val) {
		return 0.0, errors.New("REAL value -0 must be encoded as a special value")
	}
	return val, nil
}

func parseSpecialFloat(v []byte) (float64, error) {
	if len(v) != 1 {
		return 0.0, errors.New(`encoding of "special value" must not contain exponent and mantissa`)
	}
	switch v[0] {
	case 0x40:
		return math.Inf(1), nil
	case 0x41:
		return math.Inf(-1), nil
	case 0x42:
		return math.NaN(), nil
	case 0x43:
		return math.Copysign(0, -1), nil
	}
	return 0.0, errors.New(`encoding of "special value" not from ASN.1 standard`)
}
