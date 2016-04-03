package jose

import (
	"strings"
	"testing"
)

type testCase struct{ t string }

var validInput []testCase

var invalidInput []testCase

func init() {
	validInput = []testCase{
		{
			"eyJ0eXAiOiJKV1QiLA0KICJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJqb2UiLA0KICJleHAiOjEzMDA4MTkzODAsDQogImh0dHA6Ly9leGFtcGxlLmNvbS9pc19yb290Ijp0cnVlfQ.dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk",
		},
	}

	invalidInput = []testCase{
		// empty
		{
			"",
		},
		// undecodeable
		{
			"aaa.bbb.ccc",
		},
		// missing parts
		{
			"aaa",
		},
		// missing parts
		{
			"aaa.bbb",
		},
		// too many parts
		{
			"aaa.bbb.ccc.ddd",
		},
		// invalid header
		// EncodeHeader(map[string]string{"foo": "bar"})
		{
			"eyJmb28iOiJiYXIifQ.bbb.ccc",
		},
	}
}

func TestParseJWS(t *testing.T) {
	for i, tt := range validInput {
		jws, err := ParseJWS(tt.t)
		if err != nil {
			t.Errorf("test: %d. expected: valid, actual: invalid", i)
		}

		expectedHeader := strings.Split(tt.t, ".")[0]
		if jws.RawHeader != expectedHeader {
			t.Errorf("test: %d. expected: %s, actual: %s", i, expectedHeader, jws.RawHeader)
		}

		expectedPayload := strings.Split(tt.t, ".")[1]
		if jws.RawPayload != expectedPayload {
			t.Errorf("test: %d. expected: %s, actual: %s", i, expectedPayload, jws.RawPayload)
		}
	}

	for i, tt := range invalidInput {
		_, err := ParseJWS(tt.t)
		if err == nil {
			t.Errorf("test: %d. expected: invalid, actual: valid", i)
		}
	}
}
