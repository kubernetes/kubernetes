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

package admission

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAddAnnotation(t *testing.T) {
	attr := &attributesRecord{}

	// test AddAnnotation
	attr.AddAnnotation("podsecuritypolicy.admission.k8s.io/validate-policy", "privileged")
	attr.AddAnnotation("podsecuritypolicy.admission.k8s.io/admit-policy", "privileged")
	annotations := attr.getAnnotations()
	assert.Equal(t, annotations["podsecuritypolicy.admission.k8s.io/validate-policy"], "privileged")

	// test overwrite
	assert.Error(t, attr.AddAnnotation("podsecuritypolicy.admission.k8s.io/validate-policy", "privileged-overwrite"),
		"admission annotations should not be allowd to be overwritten")
	annotations = attr.getAnnotations()
	assert.Equal(t, annotations["podsecuritypolicy.admission.k8s.io/validate-policy"], "privileged", "admission annotations should not be overwritten")

	// test invalid plugin names
	var testCases map[string]string = map[string]string{
		"invalid dns subdomain": "INVALID-DNS-Subdomain/policy",
		"no plugin name":        "policy",
		"no key name":           "podsecuritypolicy.admission.k8s.io",
		"empty key":             "",
	}
	for name, invalidKey := range testCases {
		err := attr.AddAnnotation(invalidKey, "value-foo")
		assert.Error(t, err)
		annotations = attr.getAnnotations()
		assert.Equal(t, annotations[invalidKey], "", name+": invalid pluginName is not allowed ")
	}

	// test all saved annotations
	assert.Equal(
		t,
		annotations,
		map[string]string{
			"podsecuritypolicy.admission.k8s.io/validate-policy": "privileged",
			"podsecuritypolicy.admission.k8s.io/admit-policy":    "privileged",
		},
		"unexpected final annotations",
	)
}
