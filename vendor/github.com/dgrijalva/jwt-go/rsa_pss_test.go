// +build go1.4

package jwt_test

import (
	"crypto/rsa"
	"io/ioutil"
	"strings"
	"testing"

	"github.com/dgrijalva/jwt-go"
)

var rsaPSSTestData = []struct {
	name        string
	tokenString string
	alg         string
	claims      map[string]interface{}
	valid       bool
}{
	{
		"Basic PS256",
		"eyJhbGciOiJQUzI1NiIsInR5cCI6IkpXVCJ9.eyJmb28iOiJiYXIifQ.PPG4xyDVY8ffp4CcxofNmsTDXsrVG2npdQuibLhJbv4ClyPTUtR5giNSvuxo03kB6I8VXVr0Y9X7UxhJVEoJOmULAwRWaUsDnIewQa101cVhMa6iR8X37kfFoiZ6NkS-c7henVkkQWu2HtotkEtQvN5hFlk8IevXXPmvZlhQhwzB1sGzGYnoi1zOfuL98d3BIjUjtlwii5w6gYG2AEEzp7HnHCsb3jIwUPdq86Oe6hIFjtBwduIK90ca4UqzARpcfwxHwVLMpatKask00AgGVI0ysdk0BLMjmLutquD03XbThHScC2C2_Pp4cHWgMzvbgLU2RYYZcZRKr46QeNgz9w",
		"PS256",
		map[string]interface{}{"foo": "bar"},
		true,
	},
	{
		"Basic PS384",
		"eyJhbGciOiJQUzM4NCIsInR5cCI6IkpXVCJ9.eyJmb28iOiJiYXIifQ.w7-qqgj97gK4fJsq_DCqdYQiylJjzWONvD0qWWWhqEOFk2P1eDULPnqHRnjgTXoO4HAw4YIWCsZPet7nR3Xxq4ZhMqvKW8b7KlfRTb9cH8zqFvzMmybQ4jv2hKc3bXYqVow3AoR7hN_CWXI3Dv6Kd2X5xhtxRHI6IL39oTVDUQ74LACe-9t4c3QRPuj6Pq1H4FAT2E2kW_0KOc6EQhCLWEhm2Z2__OZskDC8AiPpP8Kv4k2vB7l0IKQu8Pr4RcNBlqJdq8dA5D3hk5TLxP8V5nG1Ib80MOMMqoS3FQvSLyolFX-R_jZ3-zfq6Ebsqr0yEb0AH2CfsECF7935Pa0FKQ",
		"PS384",
		map[string]interface{}{"foo": "bar"},
		true,
	},
	{
		"Basic PS512",
		"eyJhbGciOiJQUzUxMiIsInR5cCI6IkpXVCJ9.eyJmb28iOiJiYXIifQ.GX1HWGzFaJevuSLavqqFYaW8_TpvcjQ8KfC5fXiSDzSiT9UD9nB_ikSmDNyDILNdtjZLSvVKfXxZJqCfefxAtiozEDDdJthZ-F0uO4SPFHlGiXszvKeodh7BuTWRI2wL9-ZO4mFa8nq3GMeQAfo9cx11i7nfN8n2YNQ9SHGovG7_T_AvaMZB_jT6jkDHpwGR9mz7x1sycckEo6teLdHRnH_ZdlHlxqknmyTu8Odr5Xh0sJFOL8BepWbbvIIn-P161rRHHiDWFv6nhlHwZnVzjx7HQrWSGb6-s2cdLie9QL_8XaMcUpjLkfOMKkDOfHo6AvpL7Jbwi83Z2ZTHjJWB-A",
		"PS512",
		map[string]interface{}{"foo": "bar"},
		true,
	},
	{
		"basic PS256 invalid: foo => bar",
		"eyJhbGciOiJQUzI1NiIsInR5cCI6IkpXVCJ9.eyJmb28iOiJiYXIifQ.PPG4xyDVY8ffp4CcxofNmsTDXsrVG2npdQuibLhJbv4ClyPTUtR5giNSvuxo03kB6I8VXVr0Y9X7UxhJVEoJOmULAwRWaUsDnIewQa101cVhMa6iR8X37kfFoiZ6NkS-c7henVkkQWu2HtotkEtQvN5hFlk8IevXXPmvZlhQhwzB1sGzGYnoi1zOfuL98d3BIjUjtlwii5w6gYG2AEEzp7HnHCsb3jIwUPdq86Oe6hIFjtBwduIK90ca4UqzARpcfwxHwVLMpatKask00AgGVI0ysdk0BLMjmLutquD03XbThHScC2C2_Pp4cHWgMzvbgLU2RYYZcZRKr46QeNgz9W",
		"PS256",
		map[string]interface{}{"foo": "bar"},
		false,
	},
}

func TestRSAPSSVerify(t *testing.T) {
	var err error

	key, _ := ioutil.ReadFile("test/sample_key.pub")
	var rsaPSSKey *rsa.PublicKey
	if rsaPSSKey, err = jwt.ParseRSAPublicKeyFromPEM(key); err != nil {
		t.Errorf("Unable to parse RSA public key: %v", err)
	}

	for _, data := range rsaPSSTestData {
		parts := strings.Split(data.tokenString, ".")

		method := jwt.GetSigningMethod(data.alg)
		err := method.Verify(strings.Join(parts[0:2], "."), parts[2], rsaPSSKey)
		if data.valid && err != nil {
			t.Errorf("[%v] Error while verifying key: %v", data.name, err)
		}
		if !data.valid && err == nil {
			t.Errorf("[%v] Invalid key passed validation", data.name)
		}
	}
}

func TestRSAPSSSign(t *testing.T) {
	var err error

	key, _ := ioutil.ReadFile("test/sample_key")
	var rsaPSSKey *rsa.PrivateKey
	if rsaPSSKey, err = jwt.ParseRSAPrivateKeyFromPEM(key); err != nil {
		t.Errorf("Unable to parse RSA private key: %v", err)
	}

	for _, data := range rsaPSSTestData {
		if data.valid {
			parts := strings.Split(data.tokenString, ".")
			method := jwt.GetSigningMethod(data.alg)
			sig, err := method.Sign(strings.Join(parts[0:2], "."), rsaPSSKey)
			if err != nil {
				t.Errorf("[%v] Error signing token: %v", data.name, err)
			}
			if sig == parts[2] {
				t.Errorf("[%v] Signatures shouldn't match\nnew:\n%v\noriginal:\n%v", data.name, sig, parts[2])
			}
		}
	}
}
