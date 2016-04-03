package jose

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
)

const (
	HeaderMediaType    = "typ"
	HeaderKeyAlgorithm = "alg"
	HeaderKeyID        = "kid"
)

const (
	// Encryption Algorithm Header Parameter Values for JWS
	// See: https://tools.ietf.org/html/draft-ietf-jose-json-web-algorithms-40#page-6
	AlgHS256 = "HS256"
	AlgHS384 = "HS384"
	AlgHS512 = "HS512"
	AlgRS256 = "RS256"
	AlgRS384 = "RS384"
	AlgRS512 = "RS512"
	AlgES256 = "ES256"
	AlgES384 = "ES384"
	AlgES512 = "ES512"
	AlgPS256 = "PS256"
	AlgPS384 = "PS384"
	AlgPS512 = "PS512"
	AlgNone  = "none"
)

const (
	// Algorithm Header Parameter Values for JWE
	// See: https://tools.ietf.org/html/draft-ietf-jose-json-web-algorithms-40#section-4.1
	AlgRSA15            = "RSA1_5"
	AlgRSAOAEP          = "RSA-OAEP"
	AlgRSAOAEP256       = "RSA-OAEP-256"
	AlgA128KW           = "A128KW"
	AlgA192KW           = "A192KW"
	AlgA256KW           = "A256KW"
	AlgDir              = "dir"
	AlgECDHES           = "ECDH-ES"
	AlgECDHESA128KW     = "ECDH-ES+A128KW"
	AlgECDHESA192KW     = "ECDH-ES+A192KW"
	AlgECDHESA256KW     = "ECDH-ES+A256KW"
	AlgA128GCMKW        = "A128GCMKW"
	AlgA192GCMKW        = "A192GCMKW"
	AlgA256GCMKW        = "A256GCMKW"
	AlgPBES2HS256A128KW = "PBES2-HS256+A128KW"
	AlgPBES2HS384A192KW = "PBES2-HS384+A192KW"
	AlgPBES2HS512A256KW = "PBES2-HS512+A256KW"
)

const (
	// Encryption Algorithm Header Parameter Values for JWE
	// See: https://tools.ietf.org/html/draft-ietf-jose-json-web-algorithms-40#page-22
	EncA128CBCHS256 = "A128CBC-HS256"
	EncA128CBCHS384 = "A128CBC-HS384"
	EncA256CBCHS512 = "A256CBC-HS512"
	EncA128GCM      = "A128GCM"
	EncA192GCM      = "A192GCM"
	EncA256GCM      = "A256GCM"
)

type JOSEHeader map[string]string

func (j JOSEHeader) Validate() error {
	if _, exists := j[HeaderKeyAlgorithm]; !exists {
		return fmt.Errorf("header missing %q parameter", HeaderKeyAlgorithm)
	}

	return nil
}

func decodeHeader(seg string) (JOSEHeader, error) {
	b, err := decodeSegment(seg)
	if err != nil {
		return nil, err
	}

	var h JOSEHeader
	err = json.Unmarshal(b, &h)
	if err != nil {
		return nil, err
	}

	return h, nil
}

func encodeHeader(h JOSEHeader) (string, error) {
	b, err := json.Marshal(h)
	if err != nil {
		return "", err
	}

	return encodeSegment(b), nil
}

// Decode JWT specific base64url encoding with padding stripped
func decodeSegment(seg string) ([]byte, error) {
	if l := len(seg) % 4; l != 0 {
		seg += strings.Repeat("=", 4-l)
	}
	return base64.URLEncoding.DecodeString(seg)
}

// Encode JWT specific base64url encoding with padding stripped
func encodeSegment(seg []byte) string {
	return strings.TrimRight(base64.URLEncoding.EncodeToString(seg), "=")
}
