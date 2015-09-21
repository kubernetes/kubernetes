package jose

import (
	"fmt"
	"strings"
)

type JWS struct {
	RawHeader  string
	Header     JOSEHeader
	RawPayload string
	Payload    []byte
	Signature  []byte
}

// Given a raw encoded JWS token parses it and verifies the structure.
func ParseJWS(raw string) (JWS, error) {
	parts := strings.Split(raw, ".")
	if len(parts) != 3 {
		return JWS{}, fmt.Errorf("malformed JWS, only %d segments", len(parts))
	}

	rawSig := parts[2]
	jws := JWS{
		RawHeader:  parts[0],
		RawPayload: parts[1],
	}

	header, err := decodeHeader(jws.RawHeader)
	if err != nil {
		return JWS{}, fmt.Errorf("malformed JWS, unable to decode header, %s", err)
	}
	if err = header.Validate(); err != nil {
		return JWS{}, fmt.Errorf("malformed JWS, %s", err)
	}
	jws.Header = header

	payload, err := decodeSegment(jws.RawPayload)
	if err != nil {
		return JWS{}, fmt.Errorf("malformed JWS, unable to decode payload: %s", err)
	}
	jws.Payload = payload

	sig, err := decodeSegment(rawSig)
	if err != nil {
		return JWS{}, fmt.Errorf("malformed JWS, unable to decode signature: %s", err)
	}
	jws.Signature = sig

	return jws, nil
}
