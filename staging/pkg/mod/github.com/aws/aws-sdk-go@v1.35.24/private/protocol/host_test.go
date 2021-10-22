// +build go1.7

package protocol

import (
	"strconv"
	"testing"
)

func TestValidHostLabel(t *testing.T) {
	cases := []struct {
		Input string
		Valid bool
	}{
		{Input: "abc123", Valid: true},
		{Input: "123", Valid: true},
		{Input: "abc", Valid: true},
		{Input: "123-abc", Valid: true},
		{Input: "{thing}-abc", Valid: false},
		{Input: "abc.123", Valid: false},
		{Input: "abc/123", Valid: false},
		{Input: "012345678901234567890123456789012345678901234567890123456789123", Valid: true},
		{Input: "0123456789012345678901234567890123456789012345678901234567891234", Valid: false},
		{Input: "", Valid: false},
	}

	for i, c := range cases {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			valid := ValidHostLabel(c.Input)
			if e, a := c.Valid, valid; e != a {
				t.Errorf("expect valid %v, got %v", e, a)
			}
		})
	}
}

func TestValidateEndpointHostHandler(t *testing.T) {
	cases := map[string]struct {
		Input string
		Valid bool
	}{
		"valid host":  {Input: "abc.123", Valid: true},
		"fqdn host":   {Input: "abc.123.", Valid: true},
		"empty label": {Input: "abc..", Valid: false},
		"max host len": {
			Input: "123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.12345",
			Valid: true,
		},
		"too long host": {
			Input: "123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456",
			Valid: false,
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			err := ValidateEndpointHost("OpName", c.Input)
			if e, a := c.Valid, err == nil; e != a {
				t.Errorf("expect valid %v, got %v, %v", e, a, err)
			}
		})
	}
}
