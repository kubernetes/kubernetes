package jwt

import (
	"encoding/json"
	"errors"
	"time"
	// "fmt"
)

// MapClaims is a claims type that uses the map[string]interface{} for JSON decoding.
// This is the default claims type if you don't supply one
type MapClaims map[string]interface{}

// VerifyAudience Compares the aud claim against cmp.
// If required is false, this method will return true if the value matches or is unset
func (m MapClaims) VerifyAudience(cmp string, req bool) bool {
	var aud []string
	switch v := m["aud"].(type) {
	case string:
		aud = append(aud, v)
	case []string:
		aud = v
	case []interface{}:
		for _, a := range v {
			vs, ok := a.(string)
			if !ok {
				return false
			}
			aud = append(aud, vs)
		}
	}
	return verifyAud(aud, cmp, req)
}

// VerifyExpiresAt compares the exp claim against cmp (cmp <= exp).
// If req is false, it will return true, if exp is unset.
func (m MapClaims) VerifyExpiresAt(cmp int64, req bool) bool {
	cmpTime := time.Unix(cmp, 0)

	v, ok := m["exp"]
	if !ok {
		return !req
	}

	switch exp := v.(type) {
	case float64:
		if exp == 0 {
			return verifyExp(nil, cmpTime, req)
		}

		return verifyExp(&newNumericDateFromSeconds(exp).Time, cmpTime, req)
	case json.Number:
		v, _ := exp.Float64()

		return verifyExp(&newNumericDateFromSeconds(v).Time, cmpTime, req)
	}

	return false
}

// VerifyIssuedAt compares the exp claim against cmp (cmp >= iat).
// If req is false, it will return true, if iat is unset.
func (m MapClaims) VerifyIssuedAt(cmp int64, req bool) bool {
	cmpTime := time.Unix(cmp, 0)

	v, ok := m["iat"]
	if !ok {
		return !req
	}

	switch iat := v.(type) {
	case float64:
		if iat == 0 {
			return verifyIat(nil, cmpTime, req)
		}

		return verifyIat(&newNumericDateFromSeconds(iat).Time, cmpTime, req)
	case json.Number:
		v, _ := iat.Float64()

		return verifyIat(&newNumericDateFromSeconds(v).Time, cmpTime, req)
	}

	return false
}

// VerifyNotBefore compares the nbf claim against cmp (cmp >= nbf).
// If req is false, it will return true, if nbf is unset.
func (m MapClaims) VerifyNotBefore(cmp int64, req bool) bool {
	cmpTime := time.Unix(cmp, 0)

	v, ok := m["nbf"]
	if !ok {
		return !req
	}

	switch nbf := v.(type) {
	case float64:
		if nbf == 0 {
			return verifyNbf(nil, cmpTime, req)
		}

		return verifyNbf(&newNumericDateFromSeconds(nbf).Time, cmpTime, req)
	case json.Number:
		v, _ := nbf.Float64()

		return verifyNbf(&newNumericDateFromSeconds(v).Time, cmpTime, req)
	}

	return false
}

// VerifyIssuer compares the iss claim against cmp.
// If required is false, this method will return true if the value matches or is unset
func (m MapClaims) VerifyIssuer(cmp string, req bool) bool {
	iss, _ := m["iss"].(string)
	return verifyIss(iss, cmp, req)
}

// Valid validates time based claims "exp, iat, nbf".
// There is no accounting for clock skew.
// As well, if any of the above claims are not in the token, it will still
// be considered a valid claim.
func (m MapClaims) Valid() error {
	vErr := new(ValidationError)
	now := TimeFunc().Unix()

	if !m.VerifyExpiresAt(now, false) {
		vErr.Inner = errors.New("Token is expired")
		vErr.Errors |= ValidationErrorExpired
	}

	if !m.VerifyIssuedAt(now, false) {
		vErr.Inner = errors.New("Token used before issued")
		vErr.Errors |= ValidationErrorIssuedAt
	}

	if !m.VerifyNotBefore(now, false) {
		vErr.Inner = errors.New("Token is not valid yet")
		vErr.Errors |= ValidationErrorNotValidYet
	}

	if vErr.valid() {
		return nil
	}

	return vErr
}
