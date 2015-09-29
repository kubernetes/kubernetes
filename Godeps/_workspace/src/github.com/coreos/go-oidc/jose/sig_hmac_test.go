package jose

import (
	"bytes"
	"encoding/base64"
	"testing"
)

var hmacTestCases = []struct {
	data  string
	sig   string
	jwk   JWK
	valid bool
	desc  string
}{
	{
		"test",
		"Aymga2LNFrM-tnkr6MYLFY2Jou46h2_Omogeu0iMCRQ=",
		JWK{
			ID:     "fake-key",
			Alg:    "HS256",
			Secret: []byte("secret"),
		},
		true,
		"valid case",
	},
	{
		"test",
		"Aymga2LNFrM-tnkr6MYLFY2Jou46h2_Omogeu0iMCRQ=",
		JWK{
			ID:     "different-key",
			Alg:    "HS256",
			Secret: []byte("secret"),
		},
		true,
		"invalid: different key, should not match",
	},
	{
		"test sig and non-matching data",
		"Aymga2LNFrM-tnkr6MYLFY2Jou46h2_Omogeu0iMCRQ=",
		JWK{
			ID:     "fake-key",
			Alg:    "HS256",
			Secret: []byte("secret"),
		},
		false,
		"invalid: sig and data should not match",
	},
}

func TestVerify(t *testing.T) {
	for _, tt := range hmacTestCases {
		v, err := NewVerifierHMAC(tt.jwk)
		if err != nil {
			t.Errorf("should construct hmac verifier. test: %s. err=%v", tt.desc, err)
		}

		decSig, _ := base64.URLEncoding.DecodeString(tt.sig)
		err = v.Verify(decSig, []byte(tt.data))
		if err == nil && !tt.valid {
			t.Errorf("verify failure. test: %s. expected: invalid, actual: valid.", tt.desc)
		}
		if err != nil && tt.valid {
			t.Errorf("verify failure. test: %s. expected: valid, actual: invalid. err=%v", tt.desc, err)
		}
	}
}

func TestSign(t *testing.T) {
	for _, tt := range hmacTestCases {
		s := NewSignerHMAC("test", tt.jwk.Secret)
		sig, err := s.Sign([]byte(tt.data))
		if err != nil {
			t.Errorf("sign failure. test: %s. err=%v", tt.desc, err)
		}

		expSig, _ := base64.URLEncoding.DecodeString(tt.sig)
		if tt.valid && !bytes.Equal(sig, expSig) {
			t.Errorf("sign failure. test: %s. expected: %s, actual: %s.", tt.desc, tt.sig, base64.URLEncoding.EncodeToString(sig))
		}
		if !tt.valid && bytes.Equal(sig, expSig) {
			t.Errorf("sign failure. test: %s. expected: invalid signature.", tt.desc)
		}
	}
}
