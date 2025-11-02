/*
Copyright 2025 The Kubernetes Authors.

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

package lease

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	coordination "k8s.io/kubernetes/pkg/apis/coordination"
	"k8s.io/utils/ptr"
)

var apiVersions = []string{"v1", "v1alpha2", "v1beta1"}

func TestDeclarativeValidateForDeclarative(t *testing.T) {
	for _, apiVersion := range apiVersions {
		testDeclarativeValidateForDeclarative(t, apiVersion)
	}
}

func testDeclarativeValidateForDeclarative(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "coordination.k8s.io",
		APIVersion: apiVersion,
	})

	testCases := map[string]struct {
		input        coordination.Lease
		expectedErrs field.ErrorList
	}{
		// baseline
		"valid lease": {
			input: mkValidLease(),
		},
		// spec.leaseDurationSeconds
		"leaseDurationSeconds: nil": {
			input: mkValidLease(func(lease *coordination.Lease) {
				lease.Spec.LeaseDurationSeconds = nil
			}),
		},
		"leaseDurationSeconds: 0": {
			input: mkValidLease(setLeaseDurationSeconds(0)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec.leaseDurationSeconds"), ptr.To[int32](0), "").WithOrigin("minimum"),
			},
		},
		"leaseDurationSeconds: positive": {
			input: mkValidLease(setLeaseDurationSeconds(60)),
		},
		"leaseDurationSeconds: negative": {
			input: mkValidLease(setLeaseDurationSeconds(-1)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec.leaseDurationSeconds"), ptr.To[int32](-1), "").WithOrigin("minimum"),
			},
		},
		// spec.leaseTransitions
		"leaseTransitions: nil": {
			input: mkValidLease(func(lease *coordination.Lease) {
				lease.Spec.LeaseTransitions = nil
			}),
		},
		"leaseTransitions: 0": {
			input: mkValidLease(setLeaseTransitions(0)),
		},
		"leaseTransitions: positive": {
			input: mkValidLease(setLeaseTransitions(5)),
		},
		"leaseTransitions: negative": {
			input: mkValidLease(setLeaseTransitions(-1)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec.leaseTransitions"), ptr.To[int32](-1), "").WithOrigin("minimum"),
			},
		},
	}

	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func TestValidateUpdateForDeclarative(t *testing.T) {
	for _, apiVersion := range apiVersions {
		testValidateUpdateForDeclarative(t, apiVersion)
	}
}
func testValidateUpdateForDeclarative(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "coordination.k8s.io",
		APIVersion: apiVersion,
	})

	testCases := map[string]struct {
		old          coordination.Lease
		update       coordination.Lease
		expectedErrs field.ErrorList
	}{
		// baseline
		"no change": {
			old:    mkValidLease(),
			update: mkValidLease(),
		},
		// spec.leaseDurationSeconds
		"leaseDurationSeconds: valid update": {
			old:    mkValidLease(setLeaseDurationSeconds(60)),
			update: mkValidLease(setLeaseDurationSeconds(120)),
		},
		"leaseDurationSeconds: 0": {
			old:    mkValidLease(setLeaseDurationSeconds(60)),
			update: mkValidLease(setLeaseDurationSeconds(0)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec.leaseDurationSeconds"), ptr.To[int32](0), "").WithOrigin("minimum"),
			},
		},
		"leaseDurationSeconds: negative": {
			old:    mkValidLease(setLeaseDurationSeconds(60)),
			update: mkValidLease(setLeaseDurationSeconds(-5)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec.leaseDurationSeconds"), ptr.To[int32](-5), "").WithOrigin("minimum"),
			},
		},
		// spec.leaseTransitions
		"leaseTransitions: valid update": {
			old:    mkValidLease(setLeaseTransitions(5)),
			update: mkValidLease(setLeaseTransitions(10)),
		},
		"leaseTransitions: 0": {
			old:    mkValidLease(setLeaseTransitions(5)),
			update: mkValidLease(setLeaseTransitions(0)),
		},
		"leaseTransitions: negative": {
			old:    mkValidLease(setLeaseTransitions(5)),
			update: mkValidLease(setLeaseTransitions(-1)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec.leaseTransitions"), ptr.To[int32](-1), "").WithOrigin("minimum"),
			},
		},
	}

	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			tc.old.ObjectMeta.ResourceVersion = "1"
			tc.update.ObjectMeta.ResourceVersion = "1"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

// mkValidLease produces a Lease which passes validation with no tweaks.
func mkValidLease(tweaks ...func(lease *coordination.Lease)) coordination.Lease {
	lease := coordination.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-lease",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: coordination.LeaseSpec{
			LeaseDurationSeconds: ptr.To[int32](60),
			AcquireTime:          &metav1.MicroTime{Time: metav1.Now().Time},
			RenewTime:            &metav1.MicroTime{Time: metav1.Now().Time},
			LeaseTransitions:     ptr.To[int32](0),
		},
	}
	for _, tweak := range tweaks {
		tweak(&lease)
	}
	return lease
}

func setLeaseDurationSeconds(val int32) func(lease *coordination.Lease) {
	return func(lease *coordination.Lease) {
		lease.Spec.LeaseDurationSeconds = ptr.To(val)
	}
}

func setLeaseTransitions(val int32) func(lease *coordination.Lease) {
	return func(lease *coordination.Lease) {
		lease.Spec.LeaseTransitions = ptr.To(val)
	}
}
