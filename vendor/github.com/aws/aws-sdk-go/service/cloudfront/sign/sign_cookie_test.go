package sign

import (
	"crypto/rsa"
	"testing"
	"time"
)

func TestNewCookieSigner(t *testing.T) {
	privKey, err := rsa.GenerateKey(randReader, 1024)
	if err != nil {
		t.Fatalf("Unexpected priv key error, %#v", err)
	}

	signer := NewCookieSigner("keyID", privKey)
	if e, a := "keyID", signer.keyID; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := privKey, signer.privKey; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestSignCookie(t *testing.T) {
	privKey, err := rsa.GenerateKey(randReader, 1024)
	if err != nil {
		t.Errorf("expect no error, got %v", err)
	}

	signer := NewCookieSigner("keyID", privKey)
	cookies, err := signer.Sign("http*://*", time.Now().Add(1*time.Hour))

	if err != nil {
		t.Errorf("expect no error, got %v", err)
	}
	if e, a := CookiePolicyName, cookies[0].Name; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := CookieSignatureName, cookies[1].Name; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := CookieKeyIDName, cookies[2].Name; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestSignCookie_WithPolicy(t *testing.T) {
	privKey, err := rsa.GenerateKey(randReader, 1024)
	if err != nil {
		t.Errorf("expect no error, got %v", err)
	}

	p := &Policy{
		Statements: []Statement{
			{
				Resource: "*",
				Condition: Condition{
					DateLessThan: &AWSEpochTime{time.Now().Add(1 * time.Hour)},
				},
			},
		},
	}

	signer := NewCookieSigner("keyID", privKey)
	cookies, err := signer.SignWithPolicy(p)

	if err != nil {
		t.Errorf("expect no error, got %v", err)
	}
	if e, a := CookiePolicyName, cookies[0].Name; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := CookieSignatureName, cookies[1].Name; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := CookieKeyIDName, cookies[2].Name; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestSignCookie_WithCookieOptions(t *testing.T) {
	privKey, err := rsa.GenerateKey(randReader, 1024)
	if err != nil {
		t.Errorf("expect no error, got %v", err)
	}

	expires := time.Now().Add(1 * time.Hour)

	signer := NewCookieSigner("keyID", privKey)
	cookies, err := signer.Sign("https://example.com/*", expires, func(o *CookieOptions) {
		o.Path = "/"
		o.Domain = ".example.com"
		o.Secure = true

	})

	if err != nil {
		t.Errorf("expect no error, got %v", err)
	}
	if e, a := CookiePolicyName, cookies[0].Name; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := CookieSignatureName, cookies[1].Name; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := CookieKeyIDName, cookies[2].Name; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

	for _, c := range cookies {
		if e, a := "/", c.Path; e != a {
			t.Errorf("expect %v, got %v", e, a)
		}
		if e, a := ".example.com", c.Domain; e != a {
			t.Errorf("expect %v, got %v", e, a)
		}
		if !c.Secure {
			t.Errorf("expect to be true")
		}
	}
}
