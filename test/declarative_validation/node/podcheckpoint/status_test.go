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

package podcheckpoint

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation/field"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/node"
	registry "k8s.io/kubernetes/pkg/registry/node/podcheckpoint"
	"k8s.io/kubernetes/test/declarative_validation/meta"
	"k8s.io/utils/ptr"
)

func TestDeclarativeValidateStatusUpdate(t *testing.T) {
	statusPath := field.NewPath("status")
	locationPath := statusPath.Child("checkpointLocation")
	containersPath := statusPath.Child("checkpointedContainers")
	templatePath := statusPath.Child("checkpointedPodTemplate")
	location := func(path string) *node.CheckpointSource {
		return &node.CheckpointSource{Type: node.CheckpointSourceTypeNodeLocal, NodeLocal: &node.NodeLocalCheckpointSource{Path: path}}
	}
	template := func(image, label string) *core.PodTemplateSpec {
		return &core.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"app": label}},
			Spec:       core.PodSpec{Containers: []core.Container{{Name: "app", Image: image}}},
		}
	}
	tests := map[string]struct {
		oldStatus    node.PodCheckpointStatus
		newStatus    node.PodCheckpointStatus
		expectedErrs field.ErrorList
	}{
		"empty status": {},
		"template can be initially captured": {
			newStatus: node.PodCheckpointStatus{CheckpointedPodTemplate: template("image:v1", "app")},
		},
		"unchanged template allows other status updates": {
			oldStatus: node.PodCheckpointStatus{CheckpointedPodTemplate: template("image:v1", "app")},
			newStatus: node.PodCheckpointStatus{CheckpointedPodTemplate: template("image:v1", "app"), NodeName: ptr.To("node")},
		},
		"template spec cannot change": {
			oldStatus:    node.PodCheckpointStatus{CheckpointedPodTemplate: template("image:v1", "app")},
			newStatus:    node.PodCheckpointStatus{CheckpointedPodTemplate: template("image:v2", "app")},
			expectedErrs: field.ErrorList{field.Invalid(templatePath, nil, "").WithOrigin("update")},
		},
		"template metadata cannot change": {
			oldStatus:    node.PodCheckpointStatus{CheckpointedPodTemplate: template("image:v1", "app")},
			newStatus:    node.PodCheckpointStatus{CheckpointedPodTemplate: template("image:v1", "other")},
			expectedErrs: field.ErrorList{field.Invalid(templatePath, nil, "").WithOrigin("update")},
		},
		"template cannot be cleared": {
			oldStatus:    node.PodCheckpointStatus{CheckpointedPodTemplate: template("image:v1", "app")},
			expectedErrs: field.ErrorList{field.Invalid(templatePath, nil, "").WithOrigin("update")},
		},
		"template cannot be replaced with an empty template": {
			oldStatus:    node.PodCheckpointStatus{CheckpointedPodTemplate: template("image:v1", "app")},
			newStatus:    node.PodCheckpointStatus{CheckpointedPodTemplate: &core.PodTemplateSpec{}},
			expectedErrs: field.ErrorList{field.Invalid(templatePath, nil, "").WithOrigin("update")},
		},
		"empty template cannot be cleared": {
			oldStatus:    node.PodCheckpointStatus{CheckpointedPodTemplate: &core.PodTemplateSpec{}},
			expectedErrs: field.ErrorList{field.Invalid(templatePath, nil, "").WithOrigin("update")},
		},
		"empty template cannot be replaced": {
			oldStatus:    node.PodCheckpointStatus{CheckpointedPodTemplate: &core.PodTemplateSpec{}},
			newStatus:    node.PodCheckpointStatus{CheckpointedPodTemplate: template("image:v1", "app")},
			expectedErrs: field.ErrorList{field.Invalid(templatePath, nil, "").WithOrigin("update")},
		},
		"valid node and source UID": {
			newStatus: node.PodCheckpointStatus{NodeName: ptr.To("node.example.com"), SourcePodUID: ptr.To(types.UID("opaque-uid"))},
		},
		"invalid node name": {
			newStatus:    node.PodCheckpointStatus{NodeName: ptr.To("Invalid Node")},
			expectedErrs: field.ErrorList{field.Invalid(statusPath.Child("nodeName"), nil, "").WithOrigin("format=k8s-long-name")},
		},
		"empty source UID": {
			newStatus:    node.PodCheckpointStatus{SourcePodUID: ptr.To(types.UID(""))},
			expectedErrs: field.ErrorList{field.TooShort(statusPath.Child("sourcePodUID"), "", 1).WithOrigin("minLength")},
		},
		"valid node-local location": {
			newStatus: node.PodCheckpointStatus{CheckpointLocation: location("namespace/pod/checkpoint")},
		},
		"empty discriminator": {
			newStatus:    node.PodCheckpointStatus{CheckpointLocation: &node.CheckpointSource{}},
			expectedErrs: field.ErrorList{field.Required(locationPath.Child("type"), "")},
		},
		"unknown backend": {
			newStatus:    node.PodCheckpointStatus{CheckpointLocation: &node.CheckpointSource{Type: "Unknown"}},
			expectedErrs: field.ErrorList{field.NotSupported(locationPath.Child("type"), nil, []string{})},
		},
		"missing node-local member": {
			newStatus:    node.PodCheckpointStatus{CheckpointLocation: &node.CheckpointSource{Type: node.CheckpointSourceTypeNodeLocal}},
			expectedErrs: field.ErrorList{field.Invalid(locationPath.Child("nodeLocal"), nil, "").WithOrigin("union")},
		},
		"member does not match discriminator": {
			newStatus: node.PodCheckpointStatus{CheckpointLocation: &node.CheckpointSource{Type: "Unknown", NodeLocal: &node.NodeLocalCheckpointSource{Path: "checkpoint"}}},
			expectedErrs: field.ErrorList{
				field.NotSupported(locationPath.Child("type"), nil, []string{}),
				field.Invalid(locationPath.Child("nodeLocal"), nil, "").WithOrigin("union"),
			},
		},
		"empty path": {
			newStatus:    node.PodCheckpointStatus{CheckpointLocation: location("")},
			expectedErrs: field.ErrorList{field.Required(locationPath.Child("nodeLocal", "path"), "")},
		},
		"valid containers": {
			newStatus: node.PodCheckpointStatus{CheckpointedContainers: []node.PodCheckpointContainerStatus{{Name: "app"}, {Name: "sidecar"}}},
		},
		"empty container name": {
			newStatus:    node.PodCheckpointStatus{CheckpointedContainers: []node.PodCheckpointContainerStatus{{}}},
			expectedErrs: field.ErrorList{field.Required(containersPath.Index(0).Child("name"), "")},
		},
		"invalid container name": {
			newStatus:    node.PodCheckpointStatus{CheckpointedContainers: []node.PodCheckpointContainerStatus{{Name: "Invalid_Name"}}},
			expectedErrs: field.ErrorList{field.Invalid(containersPath.Index(0).Child("name"), nil, "").WithOrigin("format=k8s-short-name")},
		},
		"duplicate container names": {
			newStatus:    node.PodCheckpointStatus{CheckpointedContainers: []node.PodCheckpointContainerStatus{{Name: "app"}, {Name: "app"}}},
			expectedErrs: field.ErrorList{field.Duplicate(containersPath.Index(1), nil)},
		},
		"legacy location is ratcheted": {
			oldStatus: node.PodCheckpointStatus{CheckpointLocation: &node.CheckpointSource{Type: "Unknown"}},
			newStatus: node.PodCheckpointStatus{CheckpointLocation: &node.CheckpointSource{Type: "Unknown"}, NodeName: ptr.To("node")},
		},
		"unchanged invalid container is ratcheted after reorder": {
			oldStatus: node.PodCheckpointStatus{CheckpointedContainers: []node.PodCheckpointContainerStatus{{Name: "Invalid_Name"}, {Name: "app"}}},
			newStatus: node.PodCheckpointStatus{CheckpointedContainers: []node.PodCheckpointContainerStatus{{Name: "app"}, {Name: "Invalid_Name"}}},
		},
		"new invalid container is rejected after reorder": {
			oldStatus:    node.PodCheckpointStatus{CheckpointedContainers: []node.PodCheckpointContainerStatus{{Name: "sidecar"}, {Name: "app"}}},
			newStatus:    node.PodCheckpointStatus{CheckpointedContainers: []node.PodCheckpointContainerStatus{{Name: "app"}, {Name: "Invalid_Name"}}},
			expectedErrs: field.ErrorList{field.Invalid(containersPath.Index(1).Child("name"), nil, "").WithOrigin("format=k8s-short-name")},
		},
	}
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := requestContext(apiVersion, "update", "status")
			for name, tc := range tests {
				t.Run(name, func(t *testing.T) {
					oldPC := mkPodCheckpoint()
					oldPC.ResourceVersion = "1"
					oldPC.Status = tc.oldStatus
					newPC := oldPC.DeepCopy()
					newPC.Status = tc.newStatus
					apitesting.VerifyUpdateValidationEquivalence(t, ctx, newPC, oldPC, registry.StatusStrategy, tc.expectedErrs, apitesting.WithSubResources("status"))
				})
			}
			meta.RunConditionTestCases(t, ctx, statusPath.Child("conditions"), mkPodCheckpoint(), registry.StatusStrategy, func(pc *node.PodCheckpoint, conditions []metav1.Condition) {
				pc.Status.Conditions = conditions
			})
		})
	}
}
