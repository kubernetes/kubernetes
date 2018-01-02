package jwt_test

import (
	"crypto/ecdsa"
	"io/ioutil"
	"strings"
	"testing"

	"github.com/dgrijalva/jwt-go"
)

var ecdsaTestData = []struct {
	name        string
	keys        map[string]string
	tokenString string
	alg         string
	claims      map[string]interface{}
	valid       bool
}{
	{
		"Basic ES256",
		map[string]string{"private": "test/ec256-private.pem", "public": "test/ec256-public.pem"},
		"eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NiJ9.eyJmb28iOiJiYXIifQ.feG39E-bn8HXAKhzDZq7yEAPWYDhZlwTn3sePJnU9VrGMmwdXAIEyoOnrjreYlVM_Z4N13eK9-TmMTWyfKJtHQ",
		"ES256",
		map[string]interface{}{"foo": "bar"},
		true,
	},
	{
		"Basic ES384",
		map[string]string{"private": "test/ec384-private.pem", "public": "test/ec384-public.pem"},
		"eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzM4NCJ9.eyJmb28iOiJiYXIifQ.ngAfKMbJUh0WWubSIYe5GMsA-aHNKwFbJk_wq3lq23aPp8H2anb1rRILIzVR0gUf4a8WzDtrzmiikuPWyCS6CN4-PwdgTk-5nehC7JXqlaBZU05p3toM3nWCwm_LXcld",
		"ES384",
		map[string]interface{}{"foo": "bar"},
		true,
	},
	{
		"Basic ES512",
		map[string]string{"private": "test/ec512-private.pem", "public": "test/ec512-public.pem"},
		"eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzUxMiJ9.eyJmb28iOiJiYXIifQ.AAU0TvGQOcdg2OvrwY73NHKgfk26UDekh9Prz-L_iWuTBIBqOFCWwwLsRiHB1JOddfKAls5do1W0jR_F30JpVd-6AJeTjGKA4C1A1H6gIKwRY0o_tFDIydZCl_lMBMeG5VNFAjO86-WCSKwc3hqaGkq1MugPRq_qrF9AVbuEB4JPLyL5",
		"ES512",
		map[string]interface{}{"foo": "bar"},
		true,
	},
	{
		"basic ES256 invalid: foo => bar",
		map[string]string{"private": "test/ec256-private.pem", "public": "test/ec256-public.pem"},
		"eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJmb28iOiJiYXIifQ.MEQCIHoSJnmGlPaVQDqacx_2XlXEhhqtWceVopjomc2PJLtdAiAUTeGPoNYxZw0z8mgOnnIcjoxRuNDVZvybRZF3wR1l8W",
		"ES256",
		map[string]interface{}{"foo": "bar"},
		false,
	},
}

func TestECDSAVerify(t *testing.T) {
	for _, data := range ecdsaTestData {
		var err error

		key, _ := ioutil.ReadFile(data.keys["public"])

		var ecdsaKey *ecdsa.PublicKey
		if ecdsaKey, err = jwt.ParseECPublicKeyFromPEM(key); err != nil {
			t.Errorf("Unable to parse ECDSA public key: %v", err)
		}

		parts := strings.Split(data.tokenString, ".")

		method := jwt.GetSigningMethod(data.alg)
		err = method.Verify(strings.Join(parts[0:2], "."), parts[2], ecdsaKey)
		if data.valid && err != nil {
			t.Errorf("[%v] Error while verifying key: %v", data.name, err)
		}
		if !data.valid && err == nil {
			t.Errorf("[%v] Invalid key passed validation", data.name)
		}
	}
}

func TestECDSASign(t *testing.T) {
	for _, data := range ecdsaTestData {
		var err error
		key, _ := ioutil.ReadFile(data.keys["private"])

		var ecdsaKey *ecdsa.PrivateKey
		if ecdsaKey, err = jwt.ParseECPrivateKeyFromPEM(key); err != nil {
			t.Errorf("Unable to parse ECDSA private key: %v", err)
		}

		if data.valid {
			parts := strings.Split(data.tokenString, ".")
			method := jwt.GetSigningMethod(data.alg)
			sig, err := method.Sign(strings.Join(parts[0:2], "."), ecdsaKey)
			if err != nil {
				t.Errorf("[%v] Error signing token: %v", data.name, err)
			}
			if sig == parts[2] {
				t.Errorf("[%v] Identical signatures\nbefore:\n%v\nafter:\n%v", data.name, parts[2], sig)
			}
		}
	}
}
