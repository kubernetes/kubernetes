/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1beta1

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/pkg/errors"
)

func TestMarshalJSON(t *testing.T) {
	var tests = []struct {
		bts      BootstrapTokenString
		expected string
	}{
		{BootstrapTokenString{ID: "abcdef", Secret: "abcdef0123456789"}, `"abcdef.abcdef0123456789"`},
		{BootstrapTokenString{ID: "foo", Secret: "bar"}, `"foo.bar"`},
		{BootstrapTokenString{ID: "h", Secret: "b"}, `"h.b"`},
	}
	for _, rt := range tests {
		t.Run(rt.bts.ID, func(t *testing.T) {
			b, err := json.Marshal(rt.bts)
			if err != nil {
				t.Fatalf("json.Marshal returned an unexpected error: %v", err)
			}
			if string(b) != rt.expected {
				t.Errorf(
					"failed BootstrapTokenString.MarshalJSON:\n\texpected: %s\n\t  actual: %s",
					rt.expected,
					string(b),
				)
			}
		})
	}
}

func TestUnmarshalJSON(t *testing.T) {
	var tests = []struct {
		input         string
		bts           *BootstrapTokenString
		expectedError bool
	}{
		{`"f.s"`, &BootstrapTokenString{}, true},
		{`"abcdef."`, &BootstrapTokenString{}, true},
		{`"abcdef:abcdef0123456789"`, &BootstrapTokenString{}, true},
		{`abcdef.abcdef0123456789`, &BootstrapTokenString{}, true},
		{`"abcdef.abcdef0123456789`, &BootstrapTokenString{}, true},
		{`"abcdef.ABCDEF0123456789"`, &BootstrapTokenString{}, true},
		{`"abcdef.abcdef0123456789"`, &BootstrapTokenString{ID: "abcdef", Secret: "abcdef0123456789"}, false},
		{`"123456.aabbccddeeffgghh"`, &BootstrapTokenString{ID: "123456", Secret: "aabbccddeeffgghh"}, false},
	}
	for _, rt := range tests {
		t.Run(rt.input, func(t *testing.T) {
			newbts := &BootstrapTokenString{}
			err := json.Unmarshal([]byte(rt.input), newbts)
			if (err != nil) != rt.expectedError {
				t.Errorf("failed BootstrapTokenString.UnmarshalJSON:\n\texpected error: %t\n\t  actual error: %v", rt.expectedError, err)
			} else if !reflect.DeepEqual(rt.bts, newbts) {
				t.Errorf(
					"failed BootstrapTokenString.UnmarshalJSON:\n\texpected: %v\n\t  actual: %v",
					rt.bts,
					newbts,
				)
			}
		})
	}
}

func TestJSONRoundtrip(t *testing.T) {
	var tests = []struct {
		input string
		bts   *BootstrapTokenString
	}{
		{`"abcdef.abcdef0123456789"`, nil},
		{"", &BootstrapTokenString{ID: "abcdef", Secret: "abcdef0123456789"}},
	}
	for _, rt := range tests {
		t.Run(rt.input, func(t *testing.T) {
			if err := roundtrip(rt.input, rt.bts); err != nil {
				t.Errorf("failed BootstrapTokenString JSON roundtrip with error: %v", err)
			}
		})
	}
}

func roundtrip(input string, bts *BootstrapTokenString) error {
	var b []byte
	var err error
	newbts := &BootstrapTokenString{}
	// If string input was specified, roundtrip like this: string -> (unmarshal) -> object -> (marshal) -> string
	if len(input) > 0 {
		if err := json.Unmarshal([]byte(input), newbts); err != nil {
			return errors.Wrap(err, "expected no unmarshal error, got error")
		}
		if b, err = json.Marshal(newbts); err != nil {
			return errors.Wrap(err, "expected no marshal error, got error")
		}
		if input != string(b) {
			return errors.Errorf(
				"expected token: %s\n\t  actual: %s",
				input,
				string(b),
			)
		}
	} else { // Otherwise, roundtrip like this: object -> (marshal) -> string -> (unmarshal) -> object
		if b, err = json.Marshal(bts); err != nil {
			return errors.Wrap(err, "expected no marshal error, got error")
		}
		if err := json.Unmarshal(b, newbts); err != nil {
			return errors.Wrap(err, "expected no unmarshal error, got error")
		}
		if !reflect.DeepEqual(bts, newbts) {
			return errors.Errorf(
				"expected object: %v\n\t  actual: %v",
				bts,
				newbts,
			)
		}
	}
	return nil
}

func TestTokenFromIDAndSecret(t *testing.T) {
	var tests = []struct {
		bts      BootstrapTokenString
		expected string
	}{
		{BootstrapTokenString{ID: "foo", Secret: "bar"}, "foo.bar"},
		{BootstrapTokenString{ID: "abcdef", Secret: "abcdef0123456789"}, "abcdef.abcdef0123456789"},
		{BootstrapTokenString{ID: "h", Secret: "b"}, "h.b"},
	}
	for _, rt := range tests {
		t.Run(rt.bts.ID, func(t *testing.T) {
			actual := rt.bts.String()
			if actual != rt.expected {
				t.Errorf(
					"failed BootstrapTokenString.String():\n\texpected: %s\n\t  actual: %s",
					rt.expected,
					actual,
				)
			}
		})
	}
}

func TestNewBootstrapTokenString(t *testing.T) {
	var tests = []struct {
		token         string
		expectedError bool
		bts           *BootstrapTokenString
	}{
		{token: "", expectedError: true, bts: nil},
		{token: ".", expectedError: true, bts: nil},
		{token: "1234567890123456789012", expectedError: true, bts: nil},   // invalid parcel size
		{token: "12345.1234567890123456", expectedError: true, bts: nil},   // invalid parcel size
		{token: ".1234567890123456", expectedError: true, bts: nil},        // invalid parcel size
		{token: "123456.", expectedError: true, bts: nil},                  // invalid parcel size
		{token: "123456:1234567890.123456", expectedError: true, bts: nil}, // invalid separation
		{token: "abcdef:1234567890123456", expectedError: true, bts: nil},  // invalid separation
		{token: "Abcdef.1234567890123456", expectedError: true, bts: nil},  // invalid token id
		{token: "123456.AABBCCDDEEFFGGHH", expectedError: true, bts: nil},  // invalid token secret
		{token: "123456.AABBCCD-EEFFGGHH", expectedError: true, bts: nil},  // invalid character
		{token: "abc*ef.1234567890123456", expectedError: true, bts: nil},  // invalid character
		{token: "abcdef.1234567890123456", expectedError: false, bts: &BootstrapTokenString{ID: "abcdef", Secret: "1234567890123456"}},
		{token: "123456.aabbccddeeffgghh", expectedError: false, bts: &BootstrapTokenString{ID: "123456", Secret: "aabbccddeeffgghh"}},
		{token: "abcdef.abcdef0123456789", expectedError: false, bts: &BootstrapTokenString{ID: "abcdef", Secret: "abcdef0123456789"}},
		{token: "123456.1234560123456789", expectedError: false, bts: &BootstrapTokenString{ID: "123456", Secret: "1234560123456789"}},
	}
	for _, rt := range tests {
		t.Run(rt.token, func(t *testing.T) {
			actual, err := NewBootstrapTokenString(rt.token)
			if (err != nil) != rt.expectedError {
				t.Errorf(
					"failed NewBootstrapTokenString for the token %q\n\texpected error: %t\n\t  actual error: %v",
					rt.token,
					rt.expectedError,
					err,
				)
			} else if !reflect.DeepEqual(actual, rt.bts) {
				t.Errorf(
					"failed NewBootstrapTokenString for the token %q\n\texpected: %v\n\t  actual: %v",
					rt.token,
					rt.bts,
					actual,
				)
			}
		})
	}
}

func TestNewBootstrapTokenStringFromIDAndSecret(t *testing.T) {
	var tests = []struct {
		id, secret    string
		expectedError bool
		bts           *BootstrapTokenString
	}{
		{id: "", secret: "", expectedError: true, bts: nil},
		{id: "1234567890123456789012", secret: "", expectedError: true, bts: nil}, // invalid parcel size
		{id: "12345", secret: "1234567890123456", expectedError: true, bts: nil},  // invalid parcel size
		{id: "", secret: "1234567890123456", expectedError: true, bts: nil},       // invalid parcel size
		{id: "123456", secret: "", expectedError: true, bts: nil},                 // invalid parcel size
		{id: "Abcdef", secret: "1234567890123456", expectedError: true, bts: nil}, // invalid token id
		{id: "123456", secret: "AABBCCDDEEFFGGHH", expectedError: true, bts: nil}, // invalid token secret
		{id: "123456", secret: "AABBCCD-EEFFGGHH", expectedError: true, bts: nil}, // invalid character
		{id: "abc*ef", secret: "1234567890123456", expectedError: true, bts: nil}, // invalid character
		{id: "abcdef", secret: "1234567890123456", expectedError: false, bts: &BootstrapTokenString{ID: "abcdef", Secret: "1234567890123456"}},
		{id: "123456", secret: "aabbccddeeffgghh", expectedError: false, bts: &BootstrapTokenString{ID: "123456", Secret: "aabbccddeeffgghh"}},
		{id: "abcdef", secret: "abcdef0123456789", expectedError: false, bts: &BootstrapTokenString{ID: "abcdef", Secret: "abcdef0123456789"}},
		{id: "123456", secret: "1234560123456789", expectedError: false, bts: &BootstrapTokenString{ID: "123456", Secret: "1234560123456789"}},
	}
	for _, rt := range tests {
		t.Run(rt.id, func(t *testing.T) {
			actual, err := NewBootstrapTokenStringFromIDAndSecret(rt.id, rt.secret)
			if (err != nil) != rt.expectedError {
				t.Errorf(
					"failed NewBootstrapTokenStringFromIDAndSecret for the token with id %q and secret %q\n\texpected error: %t\n\t  actual error: %v",
					rt.id,
					rt.secret,
					rt.expectedError,
					err,
				)
			} else if !reflect.DeepEqual(actual, rt.bts) {
				t.Errorf(
					"failed NewBootstrapTokenStringFromIDAndSecret for the token with id %q and secret %q\n\texpected: %v\n\t  actual: %v",
					rt.id,
					rt.secret,
					rt.bts,
					actual,
				)
			}
		})
	}
}
