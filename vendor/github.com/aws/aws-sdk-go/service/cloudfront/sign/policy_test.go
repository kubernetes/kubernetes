package sign

import (
	"bytes"
	"crypto"
	"crypto/rsa"
	"crypto/sha1"
	"encoding/base64"
	"fmt"
	"math/rand"
	"strings"
	"testing"
	"time"
)

func TestEpochTimeMarshal(t *testing.T) {
	v := AWSEpochTime{time.Now()}
	b, err := v.MarshalJSON()
	if err != nil {
		t.Fatalf("Unexpected error, %#v", err)
	}

	expected := fmt.Sprintf(`{"AWS:EpochTime":%d}`, v.UTC().Unix())
	if string(b) != expected {
		t.Errorf("Expected marshaled time to match, expect: %s, actual: %s",
			expected, string(b))
	}
}

var testCreateResource = []struct {
	scheme, u string
	expect    string
	errPrefix string
}{
	{
		"https", "https://example.com/a?b=1",
		"https://example.com/a?b=1", "",
	},
	{
		"http", "http*://example.com/a?b=1",
		"http*://example.com/a?b=1", "",
	},
	{
		"rtmp", "https://example.com/a?b=1",
		"a?b=1", "",
	},
	{
		"ftp", "ftp://example.com/a?b=1",
		"", "invalid URL scheme",
	},
}

func TestCreateResource(t *testing.T) {
	for i, v := range testCreateResource {
		r, err := CreateResource(v.scheme, v.u)
		if err != nil {
			if v.errPrefix == "" {
				t.Errorf("%d, Unexpected error %s", i, err.Error())
				continue
			}
			if !strings.HasPrefix(err.Error(), v.errPrefix) {
				t.Errorf("%d, Expected to find prefix\nexpect: %s\nactual: %s", i, v.errPrefix, err.Error())
				continue
			}
		} else if v.errPrefix != "" {
			t.Errorf("%d, Expected error %s", i, v.errPrefix)
			continue
		}

		if v.expect != r {
			t.Errorf("%d, Expected to find prefix\nexpect: %s\nactual: %s", i, v.expect, r)
		}
	}
}

var testTime = time.Date(2009, time.November, 10, 23, 0, 0, 0, time.UTC)

const expectedJSONPolicy = `{"Statement":[{"Resource":"https://example.com/a","Condition":{"DateLessThan":{"AWS:EpochTime":1257894000}}}]}`
const expectedB64Policy = `eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9leGFtcGxlLmNvbS9hIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxMjU3ODk0MDAwfX19XX0=`

func TestEncodePolicy(t *testing.T) {
	p := NewCannedPolicy("https://example.com/a", testTime)

	b64Policy, jsonPolicy, err := encodePolicy(p)
	if err != nil {
		t.Fatalf("Unexpected error, %#v", err)
	}

	if string(jsonPolicy) != expectedJSONPolicy {
		t.Errorf("Expected json encoding to match, \nexpect: %s\nactual: %s\n", expectedJSONPolicy, jsonPolicy)
	}

	if string(b64Policy) != expectedB64Policy {
		t.Errorf("Expected b64 encoding to match, \nexpect: %s\nactual: %s\n", expectedB64Policy, b64Policy)
	}
}

func TestSignEncodedPolicy(t *testing.T) {
	p := NewCannedPolicy("https://example.com/a", testTime)
	_, jsonPolicy, err := encodePolicy(p)
	if err != nil {
		t.Fatalf("Unexpected policy encode error, %#v", err)
	}

	r := newRandomReader(rand.New(rand.NewSource(1)))

	privKey, err := rsa.GenerateKey(r, 1024)
	if err != nil {
		t.Fatalf("Unexpected priv key error, %#v", err)
	}

	b64Signature, err := signEncodedPolicy(r, jsonPolicy, privKey)
	if err != nil {
		t.Fatalf("Unexpected policy sign error, %#v", err)
	}

	hash := sha1.New()
	if _, err = bytes.NewReader(jsonPolicy).WriteTo(hash); err != nil {
		t.Fatalf("Unexpected hash error, %#v", err)
	}

	decodedSig, err := base64.StdEncoding.DecodeString(string(b64Signature))
	if err != nil {
		t.Fatalf("Unexpected base64 decode signature, %#v", err)
	}

	if err := rsa.VerifyPKCS1v15(&privKey.PublicKey, crypto.SHA1, hash.Sum(nil), decodedSig); err != nil {
		t.Fatalf("Unable to verify signature, %#v", err)
	}
}

func TestAWSEscape(t *testing.T) {
	expect := "a-b_c~"
	actual := []byte("a+b=c/")
	awsEscapeEncoded(actual)
	if string(actual) != expect {
		t.Errorf("expect: %s, actual: %s", expect, string(actual))
	}
}
