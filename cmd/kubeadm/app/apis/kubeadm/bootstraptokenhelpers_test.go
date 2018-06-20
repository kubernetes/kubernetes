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

package kubeadm

import (
	"encoding/json"
	"reflect"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// This timestamp is used as the reference value when computing expiration dates based on TTLs in these unit tests
var refTime = time.Date(1970, time.January, 1, 1, 1, 1, 0, time.UTC)

func TestToSecret(t *testing.T) {

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
				Type: v1.SecretType("bootstrap.kubernetes.io/token"),
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
		actual := rt.bt.ToSecret()
		if !reflect.DeepEqual(actual, rt.secret) {
			t.Errorf(
				"failed BootstrapToken.ToSecret():\n\texpected: %v\n\t  actual: %v",
				rt.secret,
				actual,
			)
		}
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
		actual, err := BootstrapTokenFromSecret(rt.bt.ToSecret())
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
	}
}

func TestEncodeTokenSecretData(t *testing.T) {
	var tests = []struct {
		bt   *BootstrapToken
		data map[string][]byte
	}{
		{
			&BootstrapToken{ // the minimum amount of information needed to be specified
				Token: &BootstrapTokenString{ID: "abcdef", Secret: "abcdef0123456789"},
			},
			map[string][]byte{
				"token-id":     []byte("abcdef"),
				"token-secret": []byte("abcdef0123456789"),
			},
		},
		{
			&BootstrapToken{ // adds description
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
			&BootstrapToken{ // adds ttl
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
			&BootstrapToken{ // adds expiration
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
			&BootstrapToken{ // adds ttl and expiration, should favor expiration
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
			&BootstrapToken{ // adds usages
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
			&BootstrapToken{ // adds groups
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
			&BootstrapToken{ // all together
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
		actual := encodeTokenSecretData(rt.bt, refTime)
		if !reflect.DeepEqual(actual, rt.data) {
			t.Errorf(
				"failed encodeTokenSecretData:\n\texpected: %v\n\t  actual: %v",
				rt.data,
				actual,
			)
		}
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
		name          string
		data          map[string][]byte
		bt            *BootstrapToken
		expectedError bool
	}{
		{ // minimum information
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
		{ // invalid token id
			"bootstrap-token-abcdef",
			map[string][]byte{
				"token-id":     []byte("abcdeF"),
				"token-secret": []byte("abcdef0123456789"),
			},
			nil,
			true,
		},
		{ // invalid secret naming
			"foo",
			map[string][]byte{
				"token-id":     []byte("abcdef"),
				"token-secret": []byte("abcdef0123456789"),
			},
			nil,
			true,
		},
		{ // invalid token secret
			"bootstrap-token-abcdef",
			map[string][]byte{
				"token-id":     []byte("abcdef"),
				"token-secret": []byte("ABCDEF0123456789"),
			},
			nil,
			true,
		},
		{ // adds description
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
		{ // adds expiration
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
		{ // invalid expiration
			"bootstrap-token-abcdef",
			map[string][]byte{
				"token-id":     []byte("abcdef"),
				"token-secret": []byte("abcdef0123456789"),
				"expiration":   []byte("invalid date"),
			},
			nil,
			true,
		},
		{ // adds usages
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
		{ // should ignore usages that aren't set to true
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
		{ // adds groups
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
		{ // all fields set
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
		actual, err := BootstrapTokenFromSecret(&v1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      rt.name,
				Namespace: "kube-system",
			},
			Type: v1.SecretType("bootstrap.kubernetes.io/token"),
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
				continue
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
	}
}

func jsonMarshal(bt *BootstrapToken) string {
	b, _ := json.Marshal(*bt)
	return string(b)
}

func TestGetSecretString(t *testing.T) {
	var tests = []struct {
		secret      *v1.Secret
		key         string
		expectedVal string
	}{
		{
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Data: map[string][]byte{
					"foo": []byte("bar"),
				},
			},
			key:         "foo",
			expectedVal: "bar",
		},
		{
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Data: map[string][]byte{
					"foo": []byte("bar"),
				},
			},
			key:         "baz",
			expectedVal: "",
		},
		{
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			},
			key:         "foo",
			expectedVal: "",
		},
	}
	for _, rt := range tests {
		actual := getSecretString(rt.secret, rt.key)
		if actual != rt.expectedVal {
			t.Errorf(
				"failed getSecretString:\n\texpected: %s\n\t  actual: %s",
				rt.expectedVal,
				actual,
			)
		}
	}
}
