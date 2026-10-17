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
	"strings"
	"testing"

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
