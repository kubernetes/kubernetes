package jwt

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
)

const tokenDelimiter = "."

type Parser struct {
	// If populated, only these methods will be considered valid.
	validMethods []string

	// Use JSON Number format in JSON decoder.
	useJSONNumber bool

	// Skip claims validation during token parsing.
	skipClaimsValidation bool

	validator *Validator

	decodeStrict bool

	decodePaddingAllowed bool
}

// NewParser creates a new Parser with the specified options
func NewParser(options ...ParserOption) *Parser {
	p := &Parser{
		validator: &Validator{},
	}

	// Loop through our parsing options and apply them
	for _, option := range options {
		option(p)
	}

	return p
}

// Parse parses, validates, verifies the signature and returns the parsed token.
// keyFunc will receive the parsed token and should return the key for validating.
func (p *Parser) Parse(tokenString string, keyFunc Keyfunc) (*Token, error) {
	return p.ParseWithClaims(tokenString, MapClaims{}, keyFunc)
}

// ParseWithClaims parses, validates, and verifies like Parse, but supplies a default object implementing the Claims
// interface. This provides default values which can be overridden and allows a caller to use their own type, rather
// than the default MapClaims implementation of Claims.
//
// Note: If you provide a custom claim implementation that embeds one of the standard claims (such as RegisteredClaims),
// make sure that a) you either embed a non-pointer version of the claims or b) if you are using a pointer, allocate the
// proper memory for it before passing in the overall claims, otherwise you might run into a panic.
func (p *Parser) ParseWithClaims(tokenString string, claims Claims, keyFunc Keyfunc) (*Token, error) {
	token, parts, err := p.ParseUnverified(tokenString, claims)
	if err != nil {
		return token, err
	}

	// Verify signing method is in the required set
	if p.validMethods != nil {
		var signingMethodValid = false
		var alg = token.Method.Alg()
		for _, m := range p.validMethods {
			if m == alg {
				signingMethodValid = true
				break
			}
		}
		if !signingMethodValid {
			// signing method is not in the listed set
			return token, newError(fmt.Sprintf("signing method %v is invalid", alg), ErrTokenSignatureInvalid)
		}
	}

	// Decode signature
	token.Signature, err = p.DecodeSegment(parts[2])
	if err != nil {
		return token, newError("could not base64 decode signature", ErrTokenMalformed, err)
	}
	text := strings.Join(parts[0:2], ".")

	// Lookup key(s)
	if keyFunc == nil {
		// keyFunc was not provided.  short circuiting validation
		return token, newError("no keyfunc was provided", ErrTokenUnverifiable)
	}

	got, err := keyFunc(token)
	if err != nil {
		return token, newError("error while executing keyfunc", ErrTokenUnverifiable, err)
	}

	switch have := got.(type) {
	case VerificationKeySet:
		if len(have.Keys) == 0 {
			return token, newError("keyfunc returned empty verification key set", ErrTokenUnverifiable)
		}
		// Iterate through keys and verify signature, skipping the rest when a match is found.
		// Return the last error if no match is found.
		for _, key := range have.Keys {
			if err = token.Method.Verify(text, token.Signature, key); err == nil {
				break
			}
		}
	default:
		err = token.Method.Verify(text, token.Signature, have)
	}
	if err != nil {
		return token, newError("", ErrTokenSignatureInvalid, err)
	}

	// Validate Claims
	if !p.skipClaimsValidation {
		// Make sure we have at least a default validator
		if p.validator == nil {
			p.validator = NewValidator()
		}

		if err := p.validator.Validate(claims); err != nil {
			return token, newError("", ErrTokenInvalidClaims, err)
		}
	}

	// No errors so far, token is valid.
	token.Valid = true

	return token, nil
}

// ParseUnverified parses the token but doesn't validate the signature.
//
// WARNING: Don't use this method unless you know what you're doing.
//
// It's only ever useful in cases where you know the signature is valid (since it has already
// been or will be checked elsewhere in the stack) and you want to extract values from it.
func (p *Parser) ParseUnverified(tokenString string, claims Claims) (token *Token, parts []string, err error) {
	var ok bool
	parts, ok = splitToken(tokenString)
	if !ok {
		return nil, nil, newError("token contains an invalid number of segments", ErrTokenMalformed)
	}

	token = &Token{Raw: tokenString}

	// parse Header
	var headerBytes []byte
	if headerBytes, err = p.DecodeSegment(parts[0]); err != nil {
		return token, parts, newError("could not base64 decode header", ErrTokenMalformed, err)
	}
	if err = json.Unmarshal(headerBytes, &token.Header); err != nil {
		return token, parts, newError("could not JSON decode header", ErrTokenMalformed, err)
	}

	// parse Claims
	token.Claims = claims

	claimBytes, err := p.DecodeSegment(parts[1])
	if err != nil {
		return token, parts, newError("could not base64 decode claim", ErrTokenMalformed, err)
	}

	// If `useJSONNumber` is enabled then we must use *json.Decoder to decode
	// the claims. However, this comes with a performance penalty so only use
	// it if we must and, otherwise, simple use json.Unmarshal.
	if !p.useJSONNumber {
		// JSON Unmarshal. Special case for map type to avoid weird pointer behavior.
		if c, ok := token.Claims.(MapClaims); ok {
			err = json.Unmarshal(claimBytes, &c)
		} else {
			err = json.Unmarshal(claimBytes, &claims)
		}
	} else {
		dec := json.NewDecoder(bytes.NewBuffer(claimBytes))
		dec.UseNumber()
		// JSON Decode. Special case for map type to avoid weird pointer behavior.
		if c, ok := token.Claims.(MapClaims); ok {
			err = dec.Decode(&c)
		} else {
			err = dec.Decode(&claims)
		}
	}
	if err != nil {
		return token, parts, newError("could not JSON decode claim", ErrTokenMalformed, err)
	}

	// Lookup signature method
	if method, ok := token.Header["alg"].(string); ok {
		if token.Method = GetSigningMethod(method); token.Method == nil {
			return token, parts, newError("signing method (alg) is unavailable", ErrTokenUnverifiable)
		}
	} else {
		return token, parts, newError("signing method (alg) is unspecified", ErrTokenUnverifiable)
	}

	return token, parts, nil
}

// splitToken splits a token string into three parts: header, claims, and signature. It will only
// return true if the token contains exactly two delimiters and three parts. In all other cases, it
// will return nil parts and false.
func splitToken(token string) ([]string, bool) {
	parts := make([]string, 3)
	header, remain, ok := strings.Cut(token, tokenDelimiter)
	if !ok {
		return nil, false
	}
	parts[0] = header
	claims, remain, ok := strings.Cut(remain, tokenDelimiter)
	if !ok {
		return nil, false
	}
	parts[1] = claims
	// One more cut to ensure the signature is the last part of the token and there are no more
	// delimiters. This avoids an issue where malicious input could contain additional delimiters
	// causing unecessary overhead parsing tokens.
	signature, _, unexpected := strings.Cut(remain, tokenDelimiter)
	if unexpected {
		return nil, false
	}
	parts[2] = signature

	return parts, true
}

// DecodeSegment decodes a JWT specific base64url encoding. This function will
// take into account whether the [Parser] is configured with additional options,
// such as [WithStrictDecoding] or [WithPaddingAllowed].
func (p *Parser) DecodeSegment(seg string) ([]byte, error) {
	encoding := base64.RawURLEncoding

	if p.decodePaddingAllowed {
		if l := len(seg) % 4; l > 0 {
			seg += strings.Repeat("=", 4-l)
		}
		encoding = base64.URLEncoding
	}

	if p.decodeStrict {
		encoding = encoding.Strict()
	}
	return encoding.DecodeString(seg)
}

// Parse parses, validates, verifies the signature and returns the parsed token.
// keyFunc will receive the parsed token and should return the cryptographic key
// for verifying the signature. The caller is strongly encouraged to set the
// WithValidMethods option to validate the 'alg' claim in the token matches the
// expected algorithm. For more details about the importance of validating the
// 'alg' claim, see
// https://auth0.com/blog/critical-vulnerabilities-in-json-web-token-libraries/
func Parse(tokenString string, keyFunc Keyfunc, options ...ParserOption) (*Token, error) {
	return NewParser(options...).Parse(tokenString, keyFunc)
}

// ParseWithClaims is a shortcut for NewParser().ParseWithClaims().
//
// Note: If you provide a custom claim implementation that embeds one of the
// standard claims (such as RegisteredClaims), make sure that a) you either
// embed a non-pointer version of the claims or b) if you are using a pointer,
// allocate the proper memory for it before passing in the overall claims,
// otherwise you might run into a panic.
func ParseWithClaims(tokenString string, claims Claims, keyFunc Keyfunc, options ...ParserOption) (*Token, error) {
	return NewParser(options...).ParseWithClaims(tokenString, claims, keyFunc)
}
