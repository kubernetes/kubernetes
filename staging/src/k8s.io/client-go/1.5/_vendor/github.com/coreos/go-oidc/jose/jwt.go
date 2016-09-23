package jose

import "strings"

type JWT JWS

func ParseJWT(token string) (jwt JWT, err error) {
	jws, err := ParseJWS(token)
	if err != nil {
		return
	}

	return JWT(jws), nil
}

func NewJWT(header JOSEHeader, claims Claims) (jwt JWT, err error) {
	jwt = JWT{}

	jwt.Header = header
	jwt.Header[HeaderMediaType] = "JWT"

	claimBytes, err := marshalClaims(claims)
	if err != nil {
		return
	}
	jwt.Payload = claimBytes

	eh, err := encodeHeader(header)
	if err != nil {
		return
	}
	jwt.RawHeader = eh

	ec, err := encodeClaims(claims)
	if err != nil {
		return
	}
	jwt.RawPayload = ec

	return
}

func (j *JWT) KeyID() (string, bool) {
	kID, ok := j.Header[HeaderKeyID]
	return kID, ok
}

func (j *JWT) Claims() (Claims, error) {
	return decodeClaims(j.Payload)
}

// Encoded data part of the token which may be signed.
func (j *JWT) Data() string {
	return strings.Join([]string{j.RawHeader, j.RawPayload}, ".")
}

// Full encoded JWT token string in format: header.claims.signature
func (j *JWT) Encode() string {
	d := j.Data()
	s := encodeSegment(j.Signature)
	return strings.Join([]string{d, s}, ".")
}

func NewSignedJWT(claims Claims, s Signer) (*JWT, error) {
	header := JOSEHeader{
		HeaderKeyAlgorithm: s.Alg(),
		HeaderKeyID:        s.ID(),
	}

	jwt, err := NewJWT(header, claims)
	if err != nil {
		return nil, err
	}

	sig, err := s.Sign([]byte(jwt.Data()))
	if err != nil {
		return nil, err
	}
	jwt.Signature = sig

	return &jwt, nil
}
