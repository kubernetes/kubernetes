/*
Copyright 2021 The Kubernetes Authors.

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

package v1

import (
	"encoding/json"
	"reflect"
	"testing"
	"time"

	"github.com/pkg/errors"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	bootstrapapi "k8s.io/cluster-bootstrap/token/api"
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

// This timestamp is used as the reference value when computing expiration dates based on TTLs in these unit tests
var refTime = time.Date(1970, time.January, 1, 1, 1, 1, 0, time.UTC)

func TestBootstrapTokenToSecret(t *testing.T) {
	var tests = []struct {
		bt     *BootstrapToken
		secret *v1.Secret
	}{
		{
			&BootstrapToken{ // all together
				Token:       &BootstrapTokenString{ID: "abcdef", Secret: "abcdef0123456789"},
				Description: "foo",
				Expires: &metav1.Time{
					Time: refTime,
				},
				Usages: []string{"signing", "authentication"},
				Groups: []string{"system:bootstrappers", "system:bootstrappers:foo"},
			},
			&v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bootstrap-token-abcdef",
					Namespace: "kube-system",
				},
				Type: bootstrapapi.SecretTypeBootstrapToken,
				Data: map[string][]byte{
					"token-id":                       []byte("abcdef"),
					"token-secret":                   []byte("abcdef0123456789"),
					"description":                    []byte("foo"),
					"expiration":                     []byte(refTime.Format(time.RFC3339)),
					"usage-bootstrap-signing":        []byte("true"),
					"usage-bootstrap-authentication": []byte("true"),
					"auth-extra-groups":              []byte("system:bootstrappers,system:bootstrappers:foo"),
				},
			},
		},
	}
	for _, rt := range tests {
		t.Run(rt.bt.Token.ID, func(t *testing.T) {
			actual := BootstrapTokenToSecret(rt.bt)
			if !reflect.DeepEqual(actual, rt.secret) {
				t.Errorf(
					"failed BootstrapTokenToSecret():\n\texpected: %v\n\t  actual: %v",
					rt.secret,
					actual,
				)
			}
		})
	}
}

func TestBootstrapTokenToSecretRoundtrip(t *testing.T) {
	var tests = []struct {
		bt *BootstrapToken
	}{
		{
			&BootstrapToken{
				Token:       &BootstrapTokenString{ID: "abcdef", Secret: "abcdef0123456789"},
				Description: "foo",
				Expires: &metav1.Time{
					Time: refTime,
				},
				Usages: []string{"authentication", "signing"},
				Groups: []string{"system:bootstrappers", "system:bootstrappers:foo"},
			},
		},
	}
	for _, rt := range tests {
		t.Run(rt.bt.Token.ID, func(t *testing.T) {
			actual, err := BootstrapTokenFromSecret(BootstrapTokenToSecret(rt.bt))
			if err != nil {
				t.Errorf("failed BootstrapToken to Secret roundtrip with error: %v", err)
			}
			if !reflect.DeepEqual(actual, rt.bt) {
				t.Errorf(
					"failed BootstrapToken to Secret roundtrip:\n\texpected: %v\n\t  actual: %v",
					rt.bt,
					actual,
				)
			}
		})
	}
}

func TestEncodeTokenSecretData(t *testing.T) {
	var tests = []struct {
		name string
		bt   *BootstrapToken
		data map[string][]byte
	}{
		{
			"the minimum amount of information needed to be specified",
			&BootstrapToken{
				Token: &BootstrapTokenString{ID: "abcdef", Secret: "abcdef0123456789"},
			},
			map[string][]byte{
				"token-id":     []byte("abcdef"),
				"token-secret": []byte("abcdef0123456789"),
			},
		},
		{
			"adds description",
			&BootstrapToken{
				Token:       &BootstrapTokenString{ID: "abcdef", Secret: "abcdef0123456789"},
				Description: "foo",
			},
			map[string][]byte{
				"token-id":     []byte("abcdef"),
				"token-secret": []byte("abcdef0123456789"),
				"description":  []byte("foo"),
			},
		},
		{
			"adds ttl",
			&BootstrapToken{
				Token: &BootstrapTokenString{ID: "abcdef", Secret: "abcdef0123456789"},
				TTL: &metav1.Duration{
					Duration: mustParseDuration("2h", t),
				},
			},
			map[string][]byte{
				"token-id":     []byte("abcdef"),
				"token-secret": []byte("abcdef0123456789"),
				"expiration":   []byte(refTime.Add(mustParseDuration("2h", t)).Format(time.RFC3339)),
			},
		},
		{
			"adds expiration",
			&BootstrapToken{
				Token: &BootstrapTokenString{ID: "abcdef", Secret: "abcdef0123456789"},
				Expires: &metav1.Time{
					Time: refTime,
				},
			},
			map[string][]byte{
				"token-id":     []byte("abcdef"),
				"token-secret": []byte("abcdef0123456789"),
				"expiration":   []byte(refTime.Format(time.RFC3339)),
			},
		},
		{
			"adds ttl and expiration, should favor expiration",
			&BootstrapToken{
				Token: &BootstrapTokenString{ID: "abcdef", Secret: "abcdef0123456789"},
				TTL: &metav1.Duration{
					Duration: mustParseDuration("2h", t),
				},
				Expires: &metav1.Time{
					Time: refTime,
				},
			},
			map[string][]byte{
				"token-id":     []byte("abcdef"),
				"token-secret": []byte("abcdef0123456789"),
				"expiration":   []byte(refTime.Format(time.RFC3339)),
			},
		},
		{
			"adds usages",
			&BootstrapToken{
				Token:  &BootstrapTokenString{ID: "abcdef", Secret: "abcdef0123456789"},
				Usages: []string{"authentication", "signing"},
			},
			map[string][]byte{
				"token-id":                       []byte("abcdef"),
				"token-secret":                   []byte("abcdef0123456789"),
				"usage-bootstrap-signing":        []byte("true"),
				"usage-bootstrap-authentication": []byte("true"),
			},
		},
		{
			"adds groups",
			&BootstrapToken{
				Token:  &BootstrapTokenString{ID: "abcdef", Secret: "abcdef0123456789"},
				Groups: []string{"system:bootstrappers", "system:bootstrappers:foo"},
			},
			map[string][]byte{
				"token-id":          []byte("abcdef"),
				"token-secret":      []byte("abcdef0123456789"),
				"auth-extra-groups": []byte("system:bootstrappers,system:bootstrappers:foo"),
			},
		},
		{
			"all together",
			&BootstrapToken{
				Token:       &BootstrapTokenString{ID: "abcdef", Secret: "abcdef0123456789"},
				Description: "foo",
				TTL: &metav1.Duration{
					Duration: mustParseDuration("2h", t),
				},
				Expires: &metav1.Time{
					Time: refTime,
				},
				Usages: []string{"authentication", "signing"},
				Groups: []string{"system:bootstrappers", "system:bootstrappers:foo"},
			},
			map[string][]byte{
				"token-id":                       []byte("abcdef"),
				"token-secret":                   []byte("abcdef0123456789"),
				"description":                    []byte("foo"),
				"expiration":                     []byte(refTime.Format(time.RFC3339)),
				"usage-bootstrap-signing":        []byte("true"),
				"usage-bootstrap-authentication": []byte("true"),
				"auth-extra-groups":              []byte("system:bootstrappers,system:bootstrappers:foo"),
			},
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			actual := encodeTokenSecretData(rt.bt, refTime)
			if !reflect.DeepEqual(actual, rt.data) {
				t.Errorf(
					"failed encodeTokenSecretData:\n\texpected: %v\n\t  actual: %v",
					rt.data,
					actual,
				)
			}
		})
	}
}

func mustParseDuration(durationStr string, t *testing.T) time.Duration {
	d, err := time.ParseDuration(durationStr)
	if err != nil {
		t.Fatalf("couldn't parse duration %q: %v", durationStr, err)
	}
	return d
}

func TestBootstrapTokenFromSecret(t *testing.T) {
	var tests = []struct {
		desc          string
		name          string
		data          map[string][]byte
		bt            *BootstrapToken
		expectedError bool
	}{
		{
			"minimum information",
			"bootstrap-token-abcdef",
			map[string][]byte{
				"token-id":     []byte("abcdef"),
				"token-secret": []byte("abcdef0123456789"),
			},
			&BootstrapToken{
				Token: &BootstrapTokenString{ID: "abcdef", Secret: "abcdef0123456789"},
			},
			false,
		},
		{
			"invalid token id",
			"bootstrap-token-abcdef",
			map[string][]byte{
				"token-id":     []byte("abcdeF"),
				"token-secret": []byte("abcdef0123456789"),
			},
			nil,
			true,
		},
		{
			"invalid secret naming",
			"foo",
			map[string][]byte{
				"token-id":     []byte("abcdef"),
				"token-secret": []byte("abcdef0123456789"),
			},
			nil,
			true,
		},
		{
			"invalid token secret",
			"bootstrap-token-abcdef",
			map[string][]byte{
				"token-id":     []byte("abcdef"),
				"token-secret": []byte("ABCDEF0123456789"),
			},
			nil,
			true,
		},
		{
			"adds description",
			"bootstrap-token-abcdef",
			map[string][]byte{
				"token-id":     []byte("abcdef"),
				"token-secret": []byte("abcdef0123456789"),
				"description":  []byte("foo"),
			},
			&BootstrapToken{
				Token:       &BootstrapTokenString{ID: "abcdef", Secret: "abcdef0123456789"},
				Description: "foo",
			},
			false,
		},
		{
			"adds expiration",
			"bootstrap-token-abcdef",
			map[string][]byte{
				"token-id":     []byte("abcdef"),
				"token-secret": []byte("abcdef0123456789"),
				"expiration":   []byte(refTime.Format(time.RFC3339)),
			},
			&BootstrapToken{
				Token: &BootstrapTokenString{ID: "abcdef", Secret: "abcdef0123456789"},
				Expires: &metav1.Time{
					Time: refTime,
				},
			},
			false,
		},
		{
			"invalid expiration",
			"bootstrap-token-abcdef",
			map[string][]byte{
				"token-id":     []byte("abcdef"),
				"token-secret": []byte("abcdef0123456789"),
				"expiration":   []byte("invalid date"),
			},
			nil,
			true,
		},
		{
			"adds usages",
			"bootstrap-token-abcdef",
			map[string][]byte{
				"token-id":                       []byte("abcdef"),
				"token-secret":                   []byte("abcdef0123456789"),
				"usage-bootstrap-signing":        []byte("true"),
				"usage-bootstrap-authentication": []byte("true"),
			},
			&BootstrapToken{
				Token:  &BootstrapTokenString{ID: "abcdef", Secret: "abcdef0123456789"},
				Usages: []string{"authentication", "signing"},
			},
			false,
		},
		{
			"should ignore usages that aren't set to true",
			"bootstrap-token-abcdef",
			map[string][]byte{
				"token-id":                       []byte("abcdef"),
				"token-secret":                   []byte("abcdef0123456789"),
				"usage-bootstrap-signing":        []byte("true"),
				"usage-bootstrap-authentication": []byte("true"),
				"usage-bootstrap-foo":            []byte("false"),
				"usage-bootstrap-bar":            []byte(""),
			},
			&BootstrapToken{
				Token:  &BootstrapTokenString{ID: "abcdef", Secret: "abcdef0123456789"},
				Usages: []string{"authentication", "signing"},
			},
			false,
		},
		{
			"adds groups",
			"bootstrap-token-abcdef",
			map[string][]byte{
				"token-id":          []byte("abcdef"),
				"token-secret":      []byte("abcdef0123456789"),
				"auth-extra-groups": []byte("system:bootstrappers,system:bootstrappers:foo"),
			},
			&BootstrapToken{
				Token:  &BootstrapTokenString{ID: "abcdef", Secret: "abcdef0123456789"},
				Groups: []string{"system:bootstrappers", "system:bootstrappers:foo"},
			},
			false,
		},
		{
			"all fields set",
			"bootstrap-token-abcdef",
			map[string][]byte{
				"token-id":                       []byte("abcdef"),
				"token-secret":                   []byte("abcdef0123456789"),
				"description":                    []byte("foo"),
				"expiration":                     []byte(refTime.Format(time.RFC3339)),
				"usage-bootstrap-signing":        []byte("true"),
				"usage-bootstrap-authentication": []byte("true"),
				"auth-extra-groups":              []byte("system:bootstrappers,system:bootstrappers:foo"),
			},
			&BootstrapToken{
				Token:       &BootstrapTokenString{ID: "abcdef", Secret: "abcdef0123456789"},
				Description: "foo",
				Expires: &metav1.Time{
					Time: refTime,
				},
				Usages: []string{"authentication", "signing"},
				Groups: []string{"system:bootstrappers", "system:bootstrappers:foo"},
			},
			false,
		},
	}
	for _, rt := range tests {
		t.Run(rt.desc, func(t *testing.T) {
			actual, err := BootstrapTokenFromSecret(&v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      rt.name,
					Namespace: "kube-system",
				},
				Type: bootstrapapi.SecretTypeBootstrapToken,
				Data: rt.data,
			})
			if (err != nil) != rt.expectedError {
				t.Errorf(
					"failed BootstrapTokenFromSecret\n\texpected error: %t\n\t  actual error: %v",
					rt.expectedError,
					err,
				)
			} else {
				if actual == nil && rt.bt == nil {
					// if both pointers are nil, it's okay, just continue
					return
				}
				// If one of the pointers is defined but the other isn't, throw error. If both pointers are defined but unequal, throw error
				if (actual == nil && rt.bt != nil) || (actual != nil && rt.bt == nil) || !reflect.DeepEqual(*actual, *rt.bt) {
					t.Errorf(
						"failed BootstrapTokenFromSecret\n\texpected: %s\n\t  actual: %s",
						jsonMarshal(rt.bt),
						jsonMarshal(actual),
					)
				}
			}
		})
	}
}

func jsonMarshal(bt *BootstrapToken) string {
	b, _ := json.Marshal(*bt)
	return string(b)
}
