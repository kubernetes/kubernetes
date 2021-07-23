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

package validation

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/pod-security-admission/admission/api"
)

type (
	test struct {
		configuration   api.PodSecurityConfiguration
		expectedErrList field.ErrorList
	}
)

const (
	invalidValueUppercase = "TEST"
	invalidValueChars     = "_&$%#"
	invalidValueTooLong   = "testtesttesttesttesttesttesttesttesttesttesttesttesttesttesttest"
	invalidValueEmpty     = ""
	validValue            = "testing"
)

func TestValidatePodSecurityConfiguration(t *testing.T) {
	tests := []test{
		// defaults
		{
			expectedErrList: field.ErrorList{
				field.Invalid(exemptionsPath("namespaces", 0), invalidValueEmpty, "..."),
				field.Invalid(exemptionsPath("namespaces", 1), invalidValueChars, "..."),
				field.Invalid(exemptionsPath("namespaces", 2), invalidValueUppercase, "..."),
				field.Invalid(exemptionsPath("namespaces", 3), invalidValueTooLong, "..."),
				field.Duplicate(exemptionsPath("namespaces", 5), validValue),
				field.Invalid(exemptionsPath("runtimeClasses", 0), invalidValueEmpty, "..."),
				field.Invalid(exemptionsPath("runtimeClasses", 1), invalidValueChars, "..."),
				field.Invalid(exemptionsPath("runtimeClasses", 2), invalidValueUppercase, "..."),
				field.Duplicate(exemptionsPath("runtimeClasses", 4), validValue),
				field.Invalid(exemptionsPath("usernames", 0), invalidValueEmpty, "..."),
				field.Duplicate(exemptionsPath("usernames", 2), validValue),
			},
			configuration: api.PodSecurityConfiguration{
				Defaults: api.PodSecurityDefaults{
					Enforce:        "privileged",
					EnforceVersion: "v1.22",
					Audit:          "baseline",
					AuditVersion:   "v1.25",
					Warn:           "restricted",
					WarnVersion:    "latest",
				},
				Exemptions: api.PodSecurityExemptions{
					Namespaces: []string{
						invalidValueEmpty,
						invalidValueChars,
						invalidValueUppercase,
						invalidValueTooLong,
						validValue,
						validValue,
					},
					RuntimeClasses: []string{
						invalidValueEmpty,
						invalidValueChars,
						invalidValueUppercase,
						validValue,
						validValue,
					},
					Usernames: []string{
						invalidValueEmpty,
						validValue,
						validValue,
					},
				},
			},
		},
		{
			expectedErrList: field.ErrorList{
				field.Invalid(defaultsPath("enforce"), "baslein", "..."),
				field.Invalid(defaultsPath("enforce-version"), "v.122", "..."),
				field.Invalid(defaultsPath("warn"), "", "..."),
				field.Invalid(defaultsPath("warn-version"), "", "..."),
				field.Invalid(defaultsPath("audit"), "lorum", "..."),
				field.Invalid(defaultsPath("audit-version"), "ipsum", "..."),
			},
			configuration: api.PodSecurityConfiguration{
				Defaults: api.PodSecurityDefaults{
					Enforce:        "baslein",
					EnforceVersion: "v.122",
					Audit:          "lorum",
					AuditVersion:   "ipsum",
					Warn:           "",
					WarnVersion:    "",
				},
				Exemptions: api.PodSecurityExemptions{},
			},
		},
	}

	for _, test := range tests {
		errList := ValidatePodSecurityConfiguration(&test.configuration)
		if len(errList) != len(test.expectedErrList) {
			t.Errorf("expected %d errs, got %d", len(test.expectedErrList), len(errList))
		}

		for i, expected := range test.expectedErrList {
			if expected.Type.String() != errList[i].Type.String() {
				t.Errorf("expected err type %s, got %s",
					expected.Type.String(),
					errList[i].Type.String())
			}
			if expected.BadValue != errList[i].BadValue {
				t.Errorf("expected bad value '%s', got '%s'",
					expected.BadValue,
					errList[i].BadValue)
			}
		}
	}
}

// defaultsPath returns the appropriate defaults path
func defaultsPath(child string) *field.Path {
	return field.NewPath("defaults", child)
}

// exemptionsPath returns the appropriate defaults path
func exemptionsPath(child string, i int) *field.Path {
	return field.NewPath("exemptions", child).Index(i)
}
