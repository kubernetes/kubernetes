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

package leasecandidate

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	coordination "k8s.io/kubernetes/pkg/apis/coordination"
	registry "k8s.io/kubernetes/pkg/registry/coordination/leasecandidate"
	"k8s.io/kubernetes/test/declarative_validation/meta"
)

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithNamespace(genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "coordination.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "leasecandidates",
		IsResourceRequest: true,
		Verb:              "create",
	}), metav1.NamespaceDefault)

	testCases := map[string]struct {
		input        coordination.LeaseCandidate
		expectedErrs field.ErrorList
	}{
		"spec.binaryVersion: empty = invalid": {
			input: mkValidLeaseCandidate(tweakBinaryVersion("")),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "binaryVersion"), "").MarkAlpha(),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, registry.Strategy, tc.expectedErrs)
		})
	}

	obj := mkValidLeaseCandidate()
	meta.RunObjectMetaTestCases(t, ctx, &obj, registry.Strategy)
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithNamespace(genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "coordination.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "leasecandidates",
		Name:              "valid-obj",
		IsResourceRequest: true,
		Verb:              "update",
	}), metav1.NamespaceDefault)

	testCases := map[string]struct {
		old, update  coordination.LeaseCandidate
		expectedErrs field.ErrorList
	}{
		"spec.binaryVersion: set to empty = invalid": {
			old:    mkValidLeaseCandidate(),
			update: mkValidLeaseCandidate(tweakBinaryVersion("")),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "binaryVersion"), "").MarkAlpha(),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			tc.old.ResourceVersion = "1"
			tc.update.ResourceVersion = "1"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, registry.Strategy, tc.expectedErrs)
		})
	}

	updateObj := mkValidLeaseCandidate()
	meta.RunObjectMetaUpdateTestCases(t, ctx, &updateObj, registry.Strategy)
}

func mkValidLeaseCandidate(tweaks ...func(lc *coordination.LeaseCandidate)) coordination.LeaseCandidate {
	lc := coordination.LeaseCandidate{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valid-obj",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: coordination.LeaseCandidateSpec{
			LeaseName:        "lease",
			BinaryVersion:    "1.0.0",
			EmulationVersion: "1.0.0",
			Strategy:         coordination.OldestEmulationVersion,
		},
	}
	for _, tweak := range tweaks {
		tweak(&lc)
	}
	return lc
}

func tweakBinaryVersion(binaryVersion string) func(*coordination.LeaseCandidate) {
	return func(lc *coordination.LeaseCandidate) {
		lc.Spec.BinaryVersion = binaryVersion
	}
}
