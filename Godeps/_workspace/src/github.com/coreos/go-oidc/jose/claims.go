package jose

import (
	"encoding/json"
	"fmt"
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

func (c Claims) TimeClaim(name string) (time.Time, bool, error) {
	v, ok, err := c.Int64Claim(name)
	if !ok || err != nil {
		return time.Time{}, ok, err
	}

	return time.Unix(v, 0).UTC(), true, nil
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
