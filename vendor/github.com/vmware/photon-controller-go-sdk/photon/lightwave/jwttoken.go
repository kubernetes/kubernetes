package lightwave

import (
	"encoding/base64"
	"encoding/json"
	"strings"
)

type JWTToken struct {
	TokenId    string   `json:"jti"`
	Algorithm  string   `json:"alg"`
	Subject    string   `json:"sub"`
	Audience   []string `json:"aud"`
	Groups     []string `json:"groups"`
	Issuer     string   `json:"iss"`
	IssuedAt   int64    `json:"iat"`
	Expires    int64    `json:"exp"`
	Scope      string   `json:"scope"`
	TokenType  string   `json:"token_type"`
	TokenClass string   `json:"token_class"`
	Tenant     string   `json:"tenant"`
	// It's possible to have more fields depending on how Lightwave defines the token.
	// This covers all the fields we currently have.
}

// A JSON web token is a set of Base64 encoded strings separated by a period (.)
// When decoded, it will either be JSON text or a signature
// Here we decode the strings into a single token structure. We do not parse the signature.
func ParseTokenDetails(token string) (jwtToken *JWTToken) {
	jwtToken = &JWTToken{}

	chunks := strings.Split(token, ".")
	for _, chunk := range chunks {
		json_string, err := base64.RawURLEncoding.DecodeString(chunk)
		if err == nil {
			// Ignore errors. We expect that the signature is not JSON,
			// so unmarshalling it will fail. That's fine. We'll extract
			// all the data we can.
			_ = json.Unmarshal(json_string, &jwtToken)
		}
	}

	return jwtToken
}

// A JSON web token is a set of Base64 encoded strings separated by a period (.)
// When decoded, it will either be JSON text or a signature
// Here we parse the full JSON text. We do not parse the signature.
func ParseRawTokenDetails(token string) (jwtToken []string, err error) {
	chunks := strings.Split(token, ".")
	for _, chunk := range chunks {
		jsonString, err := base64.RawURLEncoding.DecodeString(chunk)
		if err == nil {
			jwtToken = append(jwtToken, string(jsonString))
		}
	}

	return jwtToken, err
}
