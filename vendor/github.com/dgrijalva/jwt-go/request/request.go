package request

import (
	"github.com/dgrijalva/jwt-go"
	"net/http"
)

// Extract and parse a JWT token from an HTTP request.
// This behaves the same as Parse, but accepts a request and an extractor
// instead of a token string.  The Extractor interface allows you to define
// the logic for extracting a token.  Several useful implementations are provided.
func ParseFromRequest(req *http.Request, extractor Extractor, keyFunc jwt.Keyfunc) (token *jwt.Token, err error) {
	return ParseFromRequestWithClaims(req, extractor, jwt.MapClaims{}, keyFunc)
}

// ParseFromRequest but with custom Claims type
func ParseFromRequestWithClaims(req *http.Request, extractor Extractor, claims jwt.Claims, keyFunc jwt.Keyfunc) (token *jwt.Token, err error) {
	// Extract token from request
	if tokStr, err := extractor.ExtractToken(req); err == nil {
		return jwt.ParseWithClaims(tokStr, claims, keyFunc)
	} else {
		return nil, err
	}
}
