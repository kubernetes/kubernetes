package credentials

import (
	"testing"
)

func TestStaticProviderGet(t *testing.T) {
	s := StaticProvider{
		Value: Value{
			AccessKeyID:     "AKID",
			SecretAccessKey: "SECRET",
			SessionToken:    "",
		},
	}

	creds, err := s.Retrieve()
	if err != nil {
		t.Errorf("expect nil, got %v", err)
	}
	if e, a := "AKID", creds.AccessKeyID; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "SECRET", creds.SecretAccessKey; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if v := creds.SessionToken; len(v) != 0 {
		t.Errorf("Expect no session token, %v", v)
	}
}

func TestStaticProviderIsExpired(t *testing.T) {
	s := StaticProvider{
		Value: Value{
			AccessKeyID:     "AKID",
			SecretAccessKey: "SECRET",
			SessionToken:    "",
		},
	}

	if s.IsExpired() {
		t.Errorf("Expect static credentials to never expire")
	}
}
