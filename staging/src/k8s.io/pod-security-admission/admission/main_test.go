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

package admission

import (
	"fmt"
	"os"
	"reflect"
	"testing"

	admissionv1 "k8s.io/api/admission/v1"
)

func TestMain(m *testing.M) {
	sharedResponses := map[string]*admissionv1.AdmissionResponse{
		"sharedAllowedResponse":                        sharedAllowedResponse,
		"sharedAllowedPrivilegedResponse":              sharedAllowedPrivilegedResponse,
		"sharedAllowedByUserExemptionResponse":         sharedAllowedByUserExemptionResponse,
		"sharedAllowedByNamespaceExemptionResponse":    sharedAllowedByNamespaceExemptionResponse,
		"sharedAllowedByRuntimeClassExemptionResponse": sharedAllowedByRuntimeClassExemptionResponse,
	}
	sharedResponseCopies := map[string]*admissionv1.AdmissionResponse{}
	for name, response := range sharedResponses {
		sharedResponseCopies[name] = response.DeepCopy()
	}

	rc := m.Run()

	for name := range sharedResponses {
		if !reflect.DeepEqual(sharedResponseCopies[name], sharedResponses[name]) {
			fmt.Fprintf(os.Stderr, "%s mutated\n", name)
			rc = 1
		}
	}

	os.Exit(rc)
}
