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

package replicationcontroller

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/utils/ptr"
)

func TestDeclarativeValidateForDeclarative(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "",
		APIVersion: "v1",
	})
	testCases := map[string]struct {
		input        api.ReplicationController
		expectedErrs field.ErrorList
	}{
		// baseline
		"empty resource": {
			input: mkValidReplicationController(),
		},
		// spec.replicas
		"nil replicas": {
			input: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.Spec.Replicas = nil
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec.replicas"), ""),
			},
		},
		"0 replicas": {
			input: mkValidReplicationController(setSpecReplicas(0)),
		},
		"1 replicas": {
			input: mkValidReplicationController(setSpecReplicas(1)),
		},
		"positive replicas": {
			input: mkValidReplicationController(setSpecReplicas(100)),
		},
		"negative replicas": {
			input: mkValidReplicationController(setSpecReplicas(-1)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec.replicas"), nil, "").WithOrigin("minimum"),
			},
		},
		// spec.minReadySeconds
		"0 minReadySeconds": {
			input: mkValidReplicationController(setSpecMinReadySeconds(0)),
		},
		"1 minReadySeconds": {
			input: mkValidReplicationController(setSpecMinReadySeconds(1)),
		},
		"positive minReadySeconds": {
			input: mkValidReplicationController(setSpecMinReadySeconds(100)),
		},
		"negative minReadySeconds": {
			input: mkValidReplicationController(setSpecMinReadySeconds(-1)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec.minReadySeconds"), nil, "").WithOrigin("minimum"),
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
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "",
		APIVersion: "v1",
	})
	testCases := map[string]struct {
		old          api.ReplicationController
		update       api.ReplicationController
		expectedErrs field.ErrorList
	}{
		// baseline
		"baseline": {
			old:    mkValidReplicationController(),
			update: mkValidReplicationController(),
		},
		// spec.replicas
		"nil replicas": {
			old: mkValidReplicationController(),
			update: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.Spec.Replicas = nil
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec.replicas"), ""),
			},
		},
		"0 replicas": {
			old:    mkValidReplicationController(),
			update: mkValidReplicationController(setSpecReplicas(0)),
		},
		"1 replicas": {
			old:    mkValidReplicationController(),
			update: mkValidReplicationController(setSpecReplicas(1)),
		},
		"positive replicas": {
			old:    mkValidReplicationController(),
			update: mkValidReplicationController(setSpecReplicas(100)),
		},
		"negative replicas": {
			old:    mkValidReplicationController(),
			update: mkValidReplicationController(setSpecReplicas(-1)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec.replicas"), nil, "").WithOrigin("minimum"),
			},
		},
		// spec.minReadySeconds
		"0 minReadySeconds": {
			old:    mkValidReplicationController(),
			update: mkValidReplicationController(setSpecMinReadySeconds(0)),
		},
		"1 minReadySeconds": {
			old:    mkValidReplicationController(),
			update: mkValidReplicationController(setSpecMinReadySeconds(1)),
		},
		"positive minReadySeconds": {
			old:    mkValidReplicationController(),
			update: mkValidReplicationController(setSpecMinReadySeconds(3)),
		},
		"negative minReadySeconds": {
			old:    mkValidReplicationController(),
			update: mkValidReplicationController(setSpecMinReadySeconds(-1)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec.minReadySeconds"), nil, "").WithOrigin("minimum"),
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

// mkValidReplicationController produces a ReplicationController which passes
// validation with no tweaks.
func mkValidReplicationController(tweaks ...func(rc *api.ReplicationController)) api.ReplicationController {
	rc := api.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
		Spec: api.ReplicationControllerSpec{
			Replicas: ptr.To[int32](1),
			Selector: map[string]string{"a": "b"},
			Template: &api.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"a": "b"},
				},
				Spec: podtest.MakePodSpec(),
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(&rc)
	}
	return rc
}

func setSpecReplicas(val int32) func(rc *api.ReplicationController) {
	return func(rc *api.ReplicationController) {
		rc.Spec.Replicas = ptr.To(val)
	}
}

func setSpecMinReadySeconds(val int32) func(rc *api.ReplicationController) {
	return func(rc *api.ReplicationController) {
		rc.Spec.MinReadySeconds = val
	}
}
