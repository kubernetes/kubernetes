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

package pod

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	registry "k8s.io/kubernetes/pkg/registry/core/pod"
)

func TestDeclarativeValidateRestoreStatus(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	statusPath := field.NewPath("status", "restoreStatus")
	statePath := statusPath.Child("restoreState")
	status := func(state api.PodRestoreState) *api.PodRestoreStatus {
		return &api.PodRestoreStatus{RestoreState: state}
	}
	terminalErr := field.Invalid(statePath, nil, "").MarkFromImperative()
	tests := map[string]struct {
		oldStatus    *api.PodRestoreStatus
		newStatus    *api.PodRestoreStatus
		expectedErrs field.ErrorList
	}{
		"unset":              {},
		"start restore":      {newStatus: status(api.PodRestoreStateInProgress)},
		"initial completion": {newStatus: status(api.PodRestoreStateCompleted)},
		"initial failure":    {newStatus: status(api.PodRestoreStateFailed)},
		"empty state": {
			newStatus:    status(""),
			expectedErrs: field.ErrorList{field.Required(statePath, "")},
		},
		"unknown state": {
			newStatus:    status("Unknown"),
			expectedErrs: field.ErrorList{field.NotSupported(statePath, nil, []string{})},
		},
		"restore completes":           {oldStatus: status(api.PodRestoreStateInProgress), newStatus: status(api.PodRestoreStateCompleted)},
		"restore fails":               {oldStatus: status(api.PodRestoreStateInProgress), newStatus: status(api.PodRestoreStateFailed)},
		"restore remains in progress": {oldStatus: status(api.PodRestoreStateInProgress), newStatus: status(api.PodRestoreStateInProgress)},
		"completed is unchanged":      {oldStatus: status(api.PodRestoreStateCompleted), newStatus: status(api.PodRestoreStateCompleted)},
		"failed is unchanged":         {oldStatus: status(api.PodRestoreStateFailed), newStatus: status(api.PodRestoreStateFailed)},
		"terminal message can change": {
			oldStatus: status(api.PodRestoreStateFailed),
			newStatus: &api.PodRestoreStatus{RestoreState: api.PodRestoreStateFailed, Reason: "RestoreInterrupted", Message: "kubelet restarted"},
		},
		"completed cannot restart": {
			oldStatus: status(api.PodRestoreStateCompleted), newStatus: status(api.PodRestoreStateInProgress),
			expectedErrs: field.ErrorList{terminalErr},
		},
		"completed cannot fail": {
			oldStatus: status(api.PodRestoreStateCompleted), newStatus: status(api.PodRestoreStateFailed),
			expectedErrs: field.ErrorList{terminalErr},
		},
		"failed cannot restart": {
			oldStatus: status(api.PodRestoreStateFailed), newStatus: status(api.PodRestoreStateInProgress),
			expectedErrs: field.ErrorList{terminalErr},
		},
		"failed cannot complete": {
			oldStatus: status(api.PodRestoreStateFailed), newStatus: status(api.PodRestoreStateCompleted),
			expectedErrs: field.ErrorList{terminalErr},
		},
		"completed cannot become empty": {
			oldStatus: status(api.PodRestoreStateCompleted), newStatus: status(""),
			expectedErrs: field.ErrorList{terminalErr, field.Required(statePath, "")},
		},
		"completed cannot become unknown": {
			oldStatus: status(api.PodRestoreStateCompleted), newStatus: status("Unknown"),
			expectedErrs: field.ErrorList{terminalErr, field.NotSupported(statePath, nil, []string{})},
		},
		"in progress cannot be cleared": {
			oldStatus:    status(api.PodRestoreStateInProgress),
			expectedErrs: field.ErrorList{field.Invalid(statusPath, nil, "").WithOrigin("update")},
		},
		"completed cannot be cleared": {
			oldStatus:    status(api.PodRestoreStateCompleted),
			expectedErrs: field.ErrorList{field.Invalid(statusPath, nil, "").WithOrigin("update")},
		},
		"failed cannot be cleared": {
			oldStatus:    status(api.PodRestoreStateFailed),
			expectedErrs: field.ErrorList{field.Invalid(statusPath, nil, "").WithOrigin("update")},
		},
		"legacy state is ratcheted": {
			oldStatus: status("Unknown"),
			newStatus: &api.PodRestoreStatus{RestoreState: "Unknown", Message: "unchanged legacy state"},
		},
	}
	for _, apiVersion := range apiVersions {
		ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
			APIGroup: "", APIVersion: apiVersion, Resource: "pods", Subresource: "status", Verb: "update", IsResourceRequest: true,
		})
		for name, tc := range tests {
			t.Run(apiVersion+"/"+name, func(t *testing.T) {
				oldPod := podtest.MakePod("pod", podtest.SetRestoreFrom("checkpoint"), podtest.SetResourceVersion("1"))
				oldPod.Status.RestoreStatus = tc.oldStatus
				newPod := oldPod.DeepCopy()
				newPod.Status.RestoreStatus = tc.newStatus
				apitesting.VerifyUpdateValidationEquivalence(t, ctx, newPod, oldPod, registry.StatusStrategy, tc.expectedErrs, apitesting.WithSubResources("status"))
			})
		}
	}
}
