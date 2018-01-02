package jwt_test

import (
	"github.com/dgrijalva/jwt-go"
	"io/ioutil"
	"strings"
	"testing"
)

var hmacTestData = []struct {
	name        string
	tokenString string
	alg         string
	claims      map[string]interface{}
	valid       bool
}{
	{
		"web sample",
		"eyJ0eXAiOiJKV1QiLA0KICJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJqb2UiLA0KICJleHAiOjEzMDA4MTkzODAsDQogImh0dHA6Ly9leGFtcGxlLmNvbS9pc19yb290Ijp0cnVlfQ.dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk",
		"HS256",
		map[string]interface{}{"iss": "joe", "exp": 1300819380, "http://example.com/is_root": true},
		true,
	},
	{
		"HS384",
		"eyJhbGciOiJIUzM4NCIsInR5cCI6IkpXVCJ9.eyJleHAiOjEuMzAwODE5MzhlKzA5LCJodHRwOi8vZXhhbXBsZS5jb20vaXNfcm9vdCI6dHJ1ZSwiaXNzIjoiam9lIn0.KWZEuOD5lbBxZ34g7F-SlVLAQ_r5KApWNWlZIIMyQVz5Zs58a7XdNzj5_0EcNoOy",
		"HS384",
		map[string]interface{}{"iss": "joe", "exp": 1300819380, "http://example.com/is_root": true},
		true,
	},
	{
		"HS512",
		"eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJleHAiOjEuMzAwODE5MzhlKzA5LCJodHRwOi8vZXhhbXBsZS5jb20vaXNfcm9vdCI6dHJ1ZSwiaXNzIjoiam9lIn0.CN7YijRX6Aw1n2jyI2Id1w90ja-DEMYiWixhYCyHnrZ1VfJRaFQz1bEbjjA5Fn4CLYaUG432dEYmSbS4Saokmw",
		"HS512",
		map[string]interface{}{"iss": "joe", "exp": 1300819380, "http://example.com/is_root": true},
		true,
	},
	{
		"web sample: invalid",
		"eyJ0eXAiOiJKV1QiLA0KICJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJqb2UiLA0KICJleHAiOjEzMDA4MTkzODAsDQogImh0dHA6Ly9leGFtcGxlLmNvbS9pc19yb290Ijp0cnVlfQ.dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXo",
		"HS256",
		map[string]interface{}{"iss": "joe", "exp": 1300819380, "http://example.com/is_root": true},
		false,
	},
}

// Sample data from http://tools.ietf.org/html/draft-jones-json-web-signature-04#appendix-A.1
var hmacTestKey, _ = ioutil.ReadFile("test/hmacTestKey")

func TestHMACVerify(t *testing.T) {
	for _, data := range hmacTestData {
		parts := strings.Split(data.tokenString, ".")

		method := jwt.GetSigningMethod(data.alg)
		err := method.Verify(strings.Join(parts[0:2], "."), parts[2], hmacTestKey)
		if data.valid && err != nil {
			t.Errorf("[%v] Error while verifying key: %v", data.name, err)
		}
		if !data.valid && err == nil {
			t.Errorf("[%v] Invalid key passed validation", data.name)
		}
	}
}

func TestHMACSign(t *testing.T) {
	for _, data := range hmacTestData {
		if data.valid {
			parts := strings.Split(data.tokenString, ".")
			method := jwt.GetSigningMethod(data.alg)
			sig, err := method.Sign(strings.Join(parts[0:2], "."), hmacTestKey)
			if err != nil {
				t.Errorf("[%v] Error signing token: %v", data.name, err)
			}
			if sig != parts[2] {
				t.Errorf("[%v] Incorrect signature.\nwas:\n%v\nexpecting:\n%v", data.name, sig, parts[2])
			}
		}
	}
}

func BenchmarkHS256Signing(b *testing.B) {
	benchmarkSigning(b, jwt.SigningMethodHS256, hmacTestKey)
}

func BenchmarkHS384Signing(b *testing.B) {
	benchmarkSigning(b, jwt.SigningMethodHS384, hmacTestKey)
}

func BenchmarkHS512Signing(b *testing.B) {
	benchmarkSigning(b, jwt.SigningMethodHS512, hmacTestKey)
}
