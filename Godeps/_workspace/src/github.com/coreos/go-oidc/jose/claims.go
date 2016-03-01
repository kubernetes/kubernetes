package jose

import (
	"encoding/json"
	"fmt"
	"math"
	"time"
)

type Claims map[string]interface{}

func (c Claims) Add(name string, value interface{}) {
	c[name] = value
}

func (c Claims) StringClaim(name string) (string, bool, error) {
	cl, ok := c[name]
	if !ok {
		return "", false, nil
	}

	v, ok := cl.(string)
	if !ok {
		return "", false, fmt.Errorf("unable to parse claim as string: %v", name)
	}

	return v, true, nil
}

func (c Claims) StringsClaim(name string) ([]string, bool, error) {
	cl, ok := c[name]
	if !ok {
		return nil, false, nil
	}

	if v, ok := cl.([]string); ok {
		return v, true, nil
	}

	// When unmarshaled, []string will become []interface{}.
	if v, ok := cl.([]interface{}); ok {
		var ret []string
		for _, vv := range v {
			str, ok := vv.(string)
			if !ok {
				return nil, false, fmt.Errorf("unable to parse claim as string array: %v", name)
			}
			ret = append(ret, str)
		}
		return ret, true, nil
	}

	return nil, false, fmt.Errorf("unable to parse claim as string array: %v", name)
}

func (c Claims) Int64Claim(name string) (int64, bool, error) {
	cl, ok := c[name]
	if !ok {
		return 0, false, nil
	}

	v, ok := cl.(int64)
	if !ok {
		vf, ok := cl.(float64)
		if !ok {
			return 0, false, fmt.Errorf("unable to parse claim as int64: %v", name)
		}
		v = int64(vf)
	}

	return v, true, nil
}

func (c Claims) Float64Claim(name string) (float64, bool, error) {
	cl, ok := c[name]
	if !ok {
		return 0, false, nil
	}

	v, ok := cl.(float64)
	if !ok {
		vi, ok := cl.(int64)
		if !ok {
			return 0, false, fmt.Errorf("unable to parse claim as float64: %v", name)
		}
		v = float64(vi)
	}

	return v, true, nil
}

func (c Claims) TimeClaim(name string) (time.Time, bool, error) {
	v, ok, err := c.Float64Claim(name)
	if !ok || err != nil {
		return time.Time{}, ok, err
	}

	s := math.Trunc(v)
	ns := (v - s) * math.Pow(10, 9)
	return time.Unix(int64(s), int64(ns)).UTC(), true, nil
}

func decodeClaims(payload []byte) (Claims, error) {
	var c Claims
	if err := json.Unmarshal(payload, &c); err != nil {
		return nil, fmt.Errorf("malformed JWT claims, unable to decode: %v", err)
	}
	return c, nil
}

func marshalClaims(c Claims) ([]byte, error) {
	b, err := json.Marshal(c)
	if err != nil {
		return nil, err
	}
	return b, nil
}

func encodeClaims(c Claims) (string, error) {
	b, err := marshalClaims(c)
	if err != nil {
		return "", err
	}

	return encodeSegment(b), nil
}
