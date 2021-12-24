package jwt

import (
	"encoding/json"
	"errors"
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

// VerifyExpiresAt compares the exp claim against cmp.
// If required is false, this method will return true if the value matches or is unset
func (m MapClaims) VerifyExpiresAt(cmp int64, req bool) bool {
	exp, ok := m["exp"]
	if !ok {
		return !req
	}
	switch expType := exp.(type) {
	case float64:
		return verifyExp(int64(expType), cmp, req)
	case json.Number:
		v, _ := expType.Int64()
		return verifyExp(v, cmp, req)
	}
	return false
}

// VerifyIssuedAt compares the iat claim against cmp.
// If required is false, this method will return true if the value matches or is unset
func (m MapClaims) VerifyIssuedAt(cmp int64, req bool) bool {
	iat, ok := m["iat"]
	if !ok {
		return !req
	}
	switch iatType := iat.(type) {
	case float64:
		return verifyIat(int64(iatType), cmp, req)
	case json.Number:
		v, _ := iatType.Int64()
		return verifyIat(v, cmp, req)
	}
	return false
}

// VerifyIssuer compares the iss claim against cmp.
// If required is false, this method will return true if the value matches or is unset
func (m MapClaims) VerifyIssuer(cmp string, req bool) bool {
	iss, _ := m["iss"].(string)
	return verifyIss(iss, cmp, req)
}

// VerifyNotBefore compares the nbf claim against cmp.
// If required is false, this method will return true if the value matches or is unset
func (m MapClaims) VerifyNotBefore(cmp int64, req bool) bool {
	nbf, ok := m["nbf"]
	if !ok {
		return !req
	}
	switch nbfType := nbf.(type) {
	case float64:
		return verifyNbf(int64(nbfType), cmp, req)
	case json.Number:
		v, _ := nbfType.Int64()
		return verifyNbf(v, cmp, req)
	}
	return false
}

// Valid calidates time based claims "exp, iat, nbf".
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
