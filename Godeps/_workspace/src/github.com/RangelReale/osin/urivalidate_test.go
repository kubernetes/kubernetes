package osin

import (
	"testing"
)

func TestURIValidate(t *testing.T) {
	// V1
	if err := ValidateUri("http://localhost:14000/appauth", "http://localhost:14000/appauth"); err != nil {
		t.Errorf("V1: %s", err)
	}

	// V2
	if err := ValidateUri("http://localhost:14000/appauth", "http://localhost:14000/app"); err == nil {
		t.Error("V2 should have failed")
	}

	// V3
	if err := ValidateUri("http://www.google.com/myapp", "http://www.google.com/myapp/interface/implementation"); err != nil {
		t.Errorf("V3: %s", err)
	}

	// V4
	if err := ValidateUri("http://www.google.com/myapp", "http://www2.google.com/myapp"); err == nil {
		t.Error("V4 should have failed")
	}
}

func TestURIListValidate(t *testing.T) {
	// V1
	if err := ValidateUriList("http://localhost:14000/appauth", "http://localhost:14000/appauth", ""); err != nil {
		t.Errorf("V1: %s", err)
	}

	// V2
	if err := ValidateUriList("http://localhost:14000/appauth", "http://localhost:14000/app", ""); err == nil {
		t.Error("V2 should have failed")
	}

	// V3
	if err := ValidateUriList("http://xxx:14000/appauth;http://localhost:14000/appauth", "http://localhost:14000/appauth", ";"); err != nil {
		t.Errorf("V3: %s", err)
	}

	// V4
	if err := ValidateUriList("http://xxx:14000/appauth;http://localhost:14000/appauth", "http://localhost:14000/app", ";"); err == nil {
		t.Error("V4 should have failed")
	}
}
