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
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation/field"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/node"
	nodev1alpha1 "k8s.io/kubernetes/pkg/apis/node/v1alpha1"
	registry "k8s.io/kubernetes/pkg/registry/node/podcheckpoint"
	"k8s.io/kubernetes/test/declarative_validation/meta"
	"k8s.io/utils/ptr"
)

// mkTemplate returns a captured template that passes Pod template validation.
func mkTemplate(image, label string, tweaks ...podtest.TweakContainer) *core.PodTemplateSpec {
	return &core.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"app": label}},
		Spec:       podtest.MakePodSpec(podtest.SetContainers(podtest.MakeContainer("app", append(tweaks, podtest.SetContainerImage(image))...))),
	}
}

func TestDeclarativeValidateStatusUpdate(t *testing.T) {
	statusPath := field.NewPath("status")
	locationPath := statusPath.Child("checkpointLocation")
	containersPath := statusPath.Child("checkpointedContainers")
	templatePath := statusPath.Child("checkpointedPodTemplate")
	location := func(path string) *node.CheckpointSource {
		return &node.CheckpointSource{Type: node.CheckpointSourceTypeNodeLocal, NodeLocal: &node.NodeLocalCheckpointSource{Path: path}}
	}
	oversizedTemplate := mkTemplate("image:v1", "app", func(c *core.Container) {
		c.Env = []core.EnvVar{{Name: "DATA", Value: strings.Repeat("x", nodev1alpha1.MaxCheckpointedPodTemplateBytes)}}
	})
	largeTemplate := mkTemplate("image:v1", "app", func(c *core.Container) {
		c.Env = []core.EnvVar{{Name: "DATA", Value: strings.Repeat("x", 768*1024)}}
	})
	tests := map[string]struct {
		oldStatus    node.PodCheckpointStatus
		newStatus    node.PodCheckpointStatus
		expectedErrs field.ErrorList
	}{
		"empty status": {},
		"template can be initially captured": {
			newStatus: node.PodCheckpointStatus{CheckpointedPodTemplate: mkTemplate("image:v1", "app")},
		},
		"template larger than 512 KiB can be captured": {
			newStatus: node.PodCheckpointStatus{CheckpointedPodTemplate: largeTemplate},
		},
		"unchanged template allows other status updates": {
			oldStatus: node.PodCheckpointStatus{CheckpointedPodTemplate: mkTemplate("image:v1", "app")},
			newStatus: node.PodCheckpointStatus{CheckpointedPodTemplate: mkTemplate("image:v1", "app"), NodeName: new("node")},
		},
		"template spec cannot change": {
			oldStatus:    node.PodCheckpointStatus{CheckpointedPodTemplate: mkTemplate("image:v1", "app")},
			newStatus:    node.PodCheckpointStatus{CheckpointedPodTemplate: mkTemplate("image:v2", "app")},
			expectedErrs: field.ErrorList{field.Invalid(templatePath, nil, "").WithOrigin("update")},
		},
		"template metadata cannot change": {
			oldStatus:    node.PodCheckpointStatus{CheckpointedPodTemplate: mkTemplate("image:v1", "app")},
			newStatus:    node.PodCheckpointStatus{CheckpointedPodTemplate: mkTemplate("image:v1", "other")},
			expectedErrs: field.ErrorList{field.Invalid(templatePath, nil, "").WithOrigin("update")},
		},
		"template cannot be cleared": {
			oldStatus:    node.PodCheckpointStatus{CheckpointedPodTemplate: mkTemplate("image:v1", "app")},
			expectedErrs: field.ErrorList{field.Invalid(templatePath, nil, "").WithOrigin("update")},
		},
		"template cannot be replaced with an empty template": {
			oldStatus:    node.PodCheckpointStatus{CheckpointedPodTemplate: mkTemplate("image:v1", "app")},
			newStatus:    node.PodCheckpointStatus{CheckpointedPodTemplate: &core.PodTemplateSpec{}},
			expectedErrs: field.ErrorList{field.Invalid(templatePath, nil, "").WithOrigin("update")},
		},
		"empty template cannot be cleared": {
			oldStatus:    node.PodCheckpointStatus{CheckpointedPodTemplate: &core.PodTemplateSpec{}},
			expectedErrs: field.ErrorList{field.Invalid(templatePath, nil, "").WithOrigin("update")},
		},
		"empty template cannot be replaced": {
			oldStatus:    node.PodCheckpointStatus{CheckpointedPodTemplate: &core.PodTemplateSpec{}},
			newStatus:    node.PodCheckpointStatus{CheckpointedPodTemplate: mkTemplate("image:v1", "app")},
			expectedErrs: field.ErrorList{field.Invalid(templatePath, nil, "").WithOrigin("update")},
		},
		"invalid template cannot be captured": {
			newStatus:    node.PodCheckpointStatus{CheckpointedPodTemplate: mkTemplate("", "app")},
			expectedErrs: field.ErrorList{field.Required(templatePath.Child("spec", "containers").Index(0).Child("image"), "").MarkFromImperative()},
		},
		"invalid template labels cannot be captured": {
			newStatus:    node.PodCheckpointStatus{CheckpointedPodTemplate: mkTemplate("image:v1", "Not A Label Value")},
			expectedErrs: field.ErrorList{field.Invalid(templatePath.Child("labels"), nil, "").WithOrigin("format=k8s-label-value").MarkFromImperative()},
		},
		"oversized template cannot be captured": {
			newStatus:    node.PodCheckpointStatus{CheckpointedPodTemplate: oversizedTemplate},
			expectedErrs: field.ErrorList{field.TooLong(templatePath, "", nodev1alpha1.MaxCheckpointedPodTemplateBytes)},
		},
		// Template validation runs once, when the template is captured.
		"unchanged invalid template is not revalidated": {
			oldStatus: node.PodCheckpointStatus{CheckpointedPodTemplate: mkTemplate("", "app")},
			newStatus: node.PodCheckpointStatus{CheckpointedPodTemplate: mkTemplate("", "app"), NodeName: new("node")},
		},
		"valid node and source UID": {
			newStatus: node.PodCheckpointStatus{NodeName: new("node.example.com"), SourcePodUID: ptr.To(types.UID("opaque-uid"))},
		},
		"unchanged node and source UID": {
			oldStatus: node.PodCheckpointStatus{NodeName: new("node"), SourcePodUID: ptr.To(types.UID("source-uid"))},
			newStatus: node.PodCheckpointStatus{NodeName: new("node"), SourcePodUID: ptr.To(types.UID("source-uid")), CheckpointLocation: location("checkpoint")},
		},
		"node cannot be reassigned": {
			oldStatus:    node.PodCheckpointStatus{NodeName: new("node-a")},
			newStatus:    node.PodCheckpointStatus{NodeName: new("node-b")},
			expectedErrs: field.ErrorList{field.Invalid(statusPath.Child("nodeName"), nil, "").WithOrigin("update")},
		},
		"node cannot be cleared": {
			oldStatus:    node.PodCheckpointStatus{NodeName: new("node-a")},
			expectedErrs: field.ErrorList{field.Invalid(statusPath.Child("nodeName"), nil, "").WithOrigin("update")},
		},
		"source UID cannot be changed": {
			oldStatus:    node.PodCheckpointStatus{SourcePodUID: ptr.To(types.UID("source-uid"))},
			newStatus:    node.PodCheckpointStatus{SourcePodUID: ptr.To(types.UID("replacement-uid"))},
			expectedErrs: field.ErrorList{field.Invalid(statusPath.Child("sourcePodUID"), nil, "").WithOrigin("update")},
		},
		"source UID cannot be cleared": {
			oldStatus:    node.PodCheckpointStatus{SourcePodUID: ptr.To(types.UID("source-uid"))},
			expectedErrs: field.ErrorList{field.Invalid(statusPath.Child("sourcePodUID"), nil, "").WithOrigin("update")},
		},
		"invalid node name": {
			newStatus:    node.PodCheckpointStatus{NodeName: new("Invalid Node")},
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
			newStatus: node.PodCheckpointStatus{CheckpointLocation: &node.CheckpointSource{Type: "Unknown"}, NodeName: new("node")},
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
			for _, tc := range meta.GenerateConditionTestCases(statusPath.Child("conditions")) {
				t.Run("conditions: "+tc.Name, func(t *testing.T) {
					oldPC := mkPodCheckpoint()
					oldPC.ResourceVersion = "1"
					newPC := oldPC.DeepCopy()
					newPC.Status.Conditions = tc.Conditions
					// The new condition list is alpha; the condition fields retain their own stability.
					for i, err := range tc.ExpectedErrs {
						if err.Type == field.ErrorTypeDuplicate {
							tc.ExpectedErrs[i] = err.MarkAlpha()
						}
					}
					apitesting.VerifyUpdateValidationEquivalence(t, ctx, newPC, oldPC, registry.StatusStrategy, tc.ExpectedErrs, apitesting.WithSubResources("status"))
				})
			}
		})
	}
}
