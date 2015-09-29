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
