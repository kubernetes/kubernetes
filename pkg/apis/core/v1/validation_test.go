/*
Copyright The Kubernetes Authors.

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
	"context"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestValidateForbiddenReservedSuffixes(t *testing.T) {
	reservedSuffixes := []string{".k8s.io", ".kubernetes.io"}
	goodValues := []string{
		"",
		"dev-kubernetes.io",
		"dev-k8s.io",
		"dev-k8s.io",
		"dev-k8s.io/path",
		"foo.dev-k8s.io",
		"this.is.a.really.long.fqdn",
		"this.is.a.really.long.fqdn/with/a/path",
		"example.co.uk",
		"10.0.0.1", // DNS labels can start with numbers and there is no requirement for letters.
		"hyphens-are-good.example.com",
		strings.Repeat("a", 63) + ".example.com",
		strings.Repeat("a", 63) + "." + strings.Repeat("b", 63) + "." + strings.Repeat("c", 63) + "." + strings.Repeat("d", 49) + ".example.com/bar",
	}
	for _, val := range goodValues {
		if err := ValidateForbiddenReservedDomainSuffixes(field.NewPath(""), val, reservedSuffixes).ToAggregate(); err != nil {
			t.Errorf("expected no errors for %q: %v", val, err)
		}
	}

	badValues := []string{
		"k8s.io",
		"k8s.io.",
		"k8s.io/path",
		"k8s.io./path",
		".k8s.io",
		"Dev.k8s.io",
		"dev.k8s.io...",
		"dev.k8s.io",
		"dev.K8s.io",
		"kubernetes.io",
		".kubernetes.io",
		"...kubernetes.io.",
		"Dev.kubernetes.io",
		"dev.kubernetes.io.",
		"dev.kubernetes.io. ",
		"dev.kubernetes.io",
		" dev.kubernetes.io",
		"dev.kubernetes.io ",
		"dev.Kubernetes.io",
		"dev.Kubernetes.io/",
		"dev.Kubernetes.io/path",
		"dev.Kubernetes.io/with/a/long/path",
		strings.Repeat("a", 64) + ".k8s.io",
		strings.Repeat("a", 63) + "." + strings.Repeat("b", 63) + "." + strings.Repeat("c", 63) + "." + strings.Repeat("d", 55) + ".k8s.io",
	}
	for _, val := range badValues {
		if err := ValidateForbiddenReservedDomainSuffixes(field.NewPath(""), val, reservedSuffixes).ToAggregate(); err == nil {
			t.Errorf("expected errors for %q", val)
		}
	}
}

func TestValidateCustom_SecurityProfileOCI_Ref(t *testing.T) {
	const digest = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	validValues := []string{
		"registry.example.com/profile@" + digest,
		"localhost:5000/profile@" + digest,
		"docker.io/library/profile@" + digest,
		"registry.example.com/profile@sha512:" + strings.Repeat("0123456789abcdef", 8),
		"registry.example.com/profile@sha384:" + strings.Repeat("0123456789abcdef", 6),
	}
	for _, value := range validValues {
		if errs := ValidateCustom_SecurityProfileOCI_Ref(context.Background(), operation.Operation{}, field.NewPath("ref"), &value, nil); len(errs) != 0 {
			t.Errorf("expected no errors for %q: %v", value, errs)
		}
	}

	invalidValues := map[string]string{
		"short name":                    "profile@" + digest,
		"repository without registry":   "profiles/seccomp@" + digest,
		"docker.io without library":     "docker.io/profile@" + digest,
		"tag":                           "registry.example.com/profile:v1",
		"tag and digest":                "registry.example.com/profile:v1@" + digest,
		"neither tag nor digest":        "registry.example.com/profile",
		"digest of the wrong length":    "registry.example.com/profile@sha256:0123",
		"unregistered digest algorithm": "registry.example.com/profile@md5:" + strings.Repeat("0", 32),
		// Longer than the maximum in bytes but not in characters, so maxLength
		// does not reject it and the format check must.
		"multibyte within the maximum": "registry.example.com/" + strings.Repeat("ä", 900) + "@" + digest,
	}
	for name, value := range invalidValues {
		errs := ValidateCustom_SecurityProfileOCI_Ref(context.Background(), operation.Operation{}, field.NewPath("ref"), &value, nil)
		if len(errs) != 1 || errs[0].Type != field.ErrorTypeInvalid {
			t.Errorf("%s: expected one invalid error for %q, got %v", name, value, errs)
		}
	}

	// Unset, empty, and oversized values are left to +k8s:required and +k8s:maxLength.
	empty := ""
	tooLong := strings.Repeat("a", securityProfileOCIRefMaxLength+1)
	for _, value := range []*string{nil, &empty, &tooLong} {
		if errs := ValidateCustom_SecurityProfileOCI_Ref(context.Background(), operation.Operation{}, field.NewPath("ref"), value, nil); len(errs) != 0 {
			t.Errorf("expected no errors for %v: %v", value, errs)
		}
	}
}

func TestValidateCustom_SecurityProfileOCIBase_LocalhostProfile(t *testing.T) {
	// A backslash is not a path separator; "profiles\..\base.json" is one
	// file name below the seccomp root on the node.
	for _, value := range []string{"profile.json", "profiles/agentic.json", "./profile.json", "profiles\\..\\base.json"} {
		if errs := ValidateCustom_SecurityProfileOCIBase_LocalhostProfile(context.Background(), operation.Operation{}, field.NewPath("localhostProfile"), &value, nil); len(errs) != 0 {
			t.Errorf("expected no errors for %q: %v", value, errs)
		}
	}

	invalidValues := map[string]struct {
		value   string
		errType field.ErrorType
	}{
		"empty":         {"", field.ErrorTypeRequired},
		"absolute path": {"/etc/profile.json", field.ErrorTypeInvalid},
		"backsteps":     {"../profile.json", field.ErrorTypeInvalid},
	}
	for name, tc := range invalidValues {
		errs := ValidateCustom_SecurityProfileOCIBase_LocalhostProfile(context.Background(), operation.Operation{}, field.NewPath("localhostProfile"), &tc.value, nil)
		if len(errs) != 1 || errs[0].Type != tc.errType {
			t.Errorf("%s: expected one %s error for %q, got %v", name, tc.errType, tc.value, errs)
		}
	}

	// Unset and oversized values are left to the union and +k8s:maxLength.
	tooLong := strings.Repeat("a", securityProfileOCIBaseLocalhostProfileMaxLength+1)
	for _, value := range []*string{nil, &tooLong} {
		if errs := ValidateCustom_SecurityProfileOCIBase_LocalhostProfile(context.Background(), operation.Operation{}, field.NewPath("localhostProfile"), value, nil); len(errs) != 0 {
			t.Errorf("expected no errors for %v: %v", value, errs)
		}
	}
}
