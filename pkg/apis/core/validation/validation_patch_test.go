/*
Copyright 2024 The Kubernetes Authors.

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

package validation

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/core"
)

func TestOpenShiftValidateSecretUpdate(t *testing.T) {
	newSecretFn := func(ns, name string, secretType core.SecretType) *core.Secret {
		return &core.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:            name,
				Namespace:       ns,
				ResourceVersion: "1",
			},
			Type: secretType,
			Data: map[string][]byte{
				"tls.key": []byte("foo"),
				"tls.crt": []byte("bar"),
			},
		}
	}
	invalidTypeErrFn := func(secretType core.SecretType) field.ErrorList {
		return field.ErrorList{
			field.Invalid(field.NewPath("type"), secretType, "field is immutable"),
		}
	}
	tlsKeyRequiredErrFn := func() field.ErrorList {
		return field.ErrorList{
			field.Required(field.NewPath("data").Key(core.TLSCertKey), ""),
			field.Required(field.NewPath("data").Key(core.TLSPrivateKeyKey), ""),
		}
	}

	t.Run("verify whitelist", func(t *testing.T) {
		for key := range whitelist {
			ns, name := key, "foo"

			// exercise a valid type mutation: "SecretTypeTLS" -> "kubernetes.io/tls"
			oldSecret, newSecret := newSecretFn(ns, name, "SecretTypeTLS"), newSecretFn(ns, name, core.SecretTypeTLS)
			if errs := ValidateSecretUpdate(newSecret, oldSecret); len(errs) > 0 {
				t.Errorf("unexpected error: %v", errs)
			}

			// the reverse should not be allowed
			errExpected := invalidTypeErrFn("SecretTypeTLS")
			oldSecret, newSecret = newSecretFn(ns, name, core.SecretTypeTLS), newSecretFn(ns, name, "SecretTypeTLS")
			if errGot := ValidateSecretUpdate(newSecret, oldSecret); !cmp.Equal(errExpected, errGot) {
				t.Errorf("expected error: %v, diff: %s", errExpected, cmp.Diff(errExpected, errGot))
			}

			// no type change, no validation failure expected
			oldSecret, newSecret = newSecretFn(ns, name, core.SecretTypeTLS), newSecretFn(ns, name, core.SecretTypeTLS)
			if errs := ValidateSecretUpdate(newSecret, oldSecret); len(errs) > 0 {
				t.Errorf("unexpected error: %v", errs)
			}

			// exercise an invalid type mutation, we expect validation failure
			errExpected = invalidTypeErrFn(core.SecretTypeOpaque)
			oldSecret, newSecret = newSecretFn(ns, name, "SecretTypeTLS"), newSecretFn(ns, name, core.SecretTypeOpaque)
			if errGot := ValidateSecretUpdate(newSecret, oldSecret); !cmp.Equal(errExpected, errGot) {
				t.Errorf("expected error: %v, diff: %s", errExpected, cmp.Diff(errExpected, errGot))
			}

			// verify that kbernetes.io/tls validation ar enforced
			errExpected = tlsKeyRequiredErrFn()
			oldSecret, newSecret = newSecretFn(ns, name, "SecretTypeTLS"), newSecretFn(ns, name, core.SecretTypeTLS)
			newSecret.Data = nil
			if errGot := ValidateSecretUpdate(newSecret, oldSecret); !cmp.Equal(errExpected, errGot) {
				t.Errorf("expected error: %v, diff: %s", errExpected, cmp.Diff(errExpected, errGot))
			}
		}
	})

	// we must not break secrets that are not in the whitelist
	tests := []struct {
		name        string
		oldSecret   *core.Secret
		newSecret   *core.Secret
		errExpected field.ErrorList
	}{
		{
			name:        "secret is not whitelisted, valid type transition, update not allowed",
			oldSecret:   newSecretFn("foo", "bar", "SecretTypeTLS"),
			newSecret:   newSecretFn("foo", "bar", core.SecretTypeTLS),
			errExpected: invalidTypeErrFn(core.SecretTypeTLS),
		},
		{
			name:        "secret is not whitelisted, invalid type transition, update not allowed",
			oldSecret:   newSecretFn("foo", "bar", "SecretTypeTLS"),
			newSecret:   newSecretFn("foo", "bar", core.SecretTypeOpaque),
			errExpected: invalidTypeErrFn(core.SecretTypeOpaque),
		},
		{
			name:      "secret is not whitelisted, no type transition, update allowed",
			oldSecret: newSecretFn("foo", "bar", core.SecretTypeTLS),
			newSecret: newSecretFn("foo", "bar", core.SecretTypeTLS),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, ok := whitelist[test.oldSecret.Namespace]; ok {
				t.Errorf("misconfigured test: secret is in whitelist: %s", test.oldSecret.Namespace)
				return
			}

			errGot := ValidateSecretUpdate(test.newSecret, test.oldSecret)
			if !cmp.Equal(test.errExpected, errGot) {
				t.Errorf("expected error: %v, diff: %s", test.errExpected, cmp.Diff(test.errExpected, errGot))
			}
		})
	}
}
