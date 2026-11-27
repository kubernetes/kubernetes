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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation/field"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/node"
	nodevalidation "k8s.io/kubernetes/pkg/apis/node/validation"
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

// mkTemplateWithContainers returns a captured template with the named
// containers, so a status can satisfy the container/template consistency check.
func mkTemplateWithContainers(names ...string) *core.PodTemplateSpec {
	containers := make([]core.Container, 0, len(names))
	for _, name := range names {
		containers = append(containers, podtest.MakeContainer(name, podtest.SetContainerImage("image:v1")))
	}
	return &core.PodTemplateSpec{Spec: podtest.MakePodSpec(podtest.SetContainers(containers...))}
}

func terminalReadyCondition() metav1.Condition {
	return metav1.Condition{
		Type:               node.PodCheckpointConditionReady,
		Status:             metav1.ConditionTrue,
		Reason:             node.PodCheckpointReasonCompleted,
		LastTransitionTime: metav1.Now(),
	}
}

func TestDeclarativeValidateStatusUpdate(t *testing.T) {
	statusPath := field.NewPath("status")
	locationPath := statusPath.Child("checkpointLocation")
	completionTimePath := statusPath.Child("completionTime")
	containersPath := statusPath.Child("checkpointedContainers")
	templatePath := statusPath.Child("checkpointedPodTemplate")
	location := func(path string) *node.CheckpointSource {
		return &node.CheckpointSource{Type: node.CheckpointSourceTypeNodeLocal, NodeLocal: &node.NodeLocalCheckpointSource{Path: path}}
	}
	oversizedTemplate := mkTemplate("image:v1", "app", func(c *core.Container) {
		c.Env = []core.EnvVar{{Name: "DATA", Value: strings.Repeat("x", nodevalidation.MaxCheckpointedPodTemplateBytes)}}
	})
	largeTemplate := mkTemplate("image:v1", "app", func(c *core.Container) {
		c.Env = []core.EnvVar{{Name: "DATA", Value: strings.Repeat("x", 768*1024)}}
	})
	restartableTemplate := mkTemplateWithContainers("app")
	restartableTemplate.Spec.InitContainers = []core.Container{
		podtest.MakeContainer("sidecar", podtest.SetContainerRestartPolicy(core.ContainerRestartPolicyAlways)),
	}
	terminalReady := terminalReadyCondition()
	terminalFailure := terminalReady
	terminalFailure.Status = metav1.ConditionFalse
	terminalFailure.Reason = node.PodCheckpointReasonFailed
	terminalSourceReplaced := terminalFailure
	terminalSourceReplaced.Reason = node.PodCheckpointReasonSourcePodReplaced
	firstCompletionTime := metav1.NewTime(time.Unix(1, 0))
	secondCompletionTime := metav1.NewTime(time.Unix(2, 0))
	tests := map[string]struct {
		oldStatus    node.PodCheckpointStatus
		newStatus    node.PodCheckpointStatus
		newStatusNil bool
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
			expectedErrs: field.ErrorList{field.TooLong(templatePath, "", nodevalidation.MaxCheckpointedPodTemplateBytes).MarkFromImperative()},
		},
		// Template validation runs once, when the template is captured.
		"unchanged invalid template is not revalidated": {
			oldStatus: node.PodCheckpointStatus{CheckpointedPodTemplate: mkTemplate("", "app")},
			newStatus: node.PodCheckpointStatus{CheckpointedPodTemplate: mkTemplate("", "app"), NodeName: new("node")},
		},
		"unchanged oversized template is not revalidated": {
			oldStatus: node.PodCheckpointStatus{CheckpointedPodTemplate: oversizedTemplate},
			newStatus: node.PodCheckpointStatus{CheckpointedPodTemplate: oversizedTemplate, NodeName: new("node")},
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
		"checkpoint location cannot be changed": {
			oldStatus:    node.PodCheckpointStatus{CheckpointLocation: location("node-a/checkpoint")},
			newStatus:    node.PodCheckpointStatus{CheckpointLocation: location("node-b/checkpoint")},
			expectedErrs: field.ErrorList{field.Invalid(locationPath, nil, "").WithOrigin("update")},
		},
		"checkpoint location cannot be cleared": {
			oldStatus:    node.PodCheckpointStatus{CheckpointLocation: location("node-a/checkpoint")},
			expectedErrs: field.ErrorList{field.Invalid(locationPath, nil, "").WithOrigin("update")},
		},
		"completion time can be initially set": {
			newStatus: node.PodCheckpointStatus{
				Conditions:     []metav1.Condition{terminalReady},
				CompletionTime: &firstCompletionTime,
			},
		},
		"completion time cannot be changed": {
			oldStatus:    node.PodCheckpointStatus{CompletionTime: &firstCompletionTime},
			newStatus:    node.PodCheckpointStatus{CompletionTime: &secondCompletionTime},
			expectedErrs: field.ErrorList{field.Invalid(completionTimePath, nil, "").WithOrigin("update")},
		},
		"completion time cannot be cleared": {
			oldStatus:    node.PodCheckpointStatus{CompletionTime: &firstCompletionTime},
			expectedErrs: field.ErrorList{field.Invalid(completionTimePath, nil, "").WithOrigin("update")},
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
			newStatus: node.PodCheckpointStatus{
				CheckpointedContainers:  []node.PodCheckpointContainerStatus{{Name: "app"}, {Name: "sidecar"}},
				CheckpointedPodTemplate: mkTemplateWithContainers("app", "sidecar"),
			},
		},
		"restartable init containers are included": {
			newStatus: node.PodCheckpointStatus{
				CheckpointedContainers:  []node.PodCheckpointContainerStatus{{Name: "app"}, {Name: "sidecar"}},
				CheckpointedPodTemplate: restartableTemplate,
			},
		},
		"checkpointed containers require a captured template": {
			newStatus: node.PodCheckpointStatus{CheckpointedContainers: []node.PodCheckpointContainerStatus{{Name: "app"}}},
			expectedErrs: field.ErrorList{
				field.Forbidden(containersPath, "requires a captured Pod template").MarkFromImperative(),
			},
		},
		"checkpointed containers must match the captured template": {
			newStatus: node.PodCheckpointStatus{
				CheckpointedContainers:  []node.PodCheckpointContainerStatus{{Name: "unexpected"}},
				CheckpointedPodTemplate: mkTemplateWithContainers("app"),
			},
			expectedErrs: field.ErrorList{
				field.Invalid(containersPath.Index(0).Child("name"), nil, "").MarkFromImperative(),
			},
		},
		"empty container name": {
			newStatus: node.PodCheckpointStatus{
				CheckpointedContainers:  []node.PodCheckpointContainerStatus{{}},
				CheckpointedPodTemplate: mkTemplateWithContainers("app"),
			},
			expectedErrs: field.ErrorList{
				field.Required(containersPath.Index(0).Child("name"), ""),
				field.Invalid(containersPath.Index(0).Child("name"), nil, "").MarkFromImperative(),
			},
		},
		"invalid container name": {
			newStatus: node.PodCheckpointStatus{
				CheckpointedContainers:  []node.PodCheckpointContainerStatus{{Name: "Invalid_Name"}},
				CheckpointedPodTemplate: mkTemplateWithContainers("app"),
			},
			expectedErrs: field.ErrorList{
				field.Invalid(containersPath.Index(0).Child("name"), nil, "").WithOrigin("format=k8s-short-name"),
				field.Invalid(containersPath.Index(0).Child("name"), nil, "").MarkFromImperative(),
			},
		},
		"duplicate container names": {
			newStatus: node.PodCheckpointStatus{
				CheckpointedContainers:  []node.PodCheckpointContainerStatus{{Name: "app"}, {Name: "app"}},
				CheckpointedPodTemplate: mkTemplateWithContainers("app"),
			},
			expectedErrs: field.ErrorList{
				field.Duplicate(containersPath.Index(1), nil),
			},
		},
		"legacy location is ratcheted": {
			oldStatus: node.PodCheckpointStatus{CheckpointLocation: &node.CheckpointSource{Type: "Unknown"}},
			newStatus: node.PodCheckpointStatus{CheckpointLocation: &node.CheckpointSource{Type: "Unknown"}, NodeName: new("node")},
		},
		"unchanged invalid container is ratcheted after reorder": {
			oldStatus: node.PodCheckpointStatus{CheckpointedContainers: []node.PodCheckpointContainerStatus{{Name: "Invalid_Name"}, {Name: "app"}}},
			newStatus: node.PodCheckpointStatus{CheckpointedContainers: []node.PodCheckpointContainerStatus{{Name: "app"}, {Name: "Invalid_Name"}}},
		},
		"containers cannot be added once set": {
			oldStatus: node.PodCheckpointStatus{CheckpointedContainers: []node.PodCheckpointContainerStatus{{Name: "app"}}},
			newStatus: node.PodCheckpointStatus{
				CheckpointedContainers:  []node.PodCheckpointContainerStatus{{Name: "app"}, {Name: "Invalid_Name"}},
				CheckpointedPodTemplate: mkTemplateWithContainers("app"),
			},
			expectedErrs: field.ErrorList{
				field.Invalid(containersPath.Index(1).Child("name"), nil, "").WithOrigin("format=k8s-short-name"),
				field.Forbidden(containersPath, "field is immutable once set").MarkFromImperative(),
			},
		},
		"terminal outcome cannot be removed": {
			oldStatus:    node.PodCheckpointStatus{Conditions: []metav1.Condition{terminalReady}},
			expectedErrs: field.ErrorList{field.Forbidden(statusPath.Child("conditions"), "cannot change or remove a terminal Ready condition").MarkFromImperative()},
		},
		"failed outcome cannot be removed": {
			oldStatus:    node.PodCheckpointStatus{Conditions: []metav1.Condition{terminalFailure}},
			expectedErrs: field.ErrorList{field.Forbidden(statusPath.Child("conditions"), "cannot change or remove a terminal Ready condition").MarkFromImperative()},
		},
		"source-replaced outcome cannot be removed": {
			oldStatus:    node.PodCheckpointStatus{Conditions: []metav1.Condition{terminalSourceReplaced}},
			expectedErrs: field.ErrorList{field.Forbidden(statusPath.Child("conditions"), "cannot change or remove a terminal Ready condition").MarkFromImperative()},
		},
		"terminal diagnostics cannot change": {
			oldStatus: node.PodCheckpointStatus{Conditions: []metav1.Condition{terminalReady}},
			newStatus: node.PodCheckpointStatus{Conditions: []metav1.Condition{{
				Type:               terminalReady.Type,
				Status:             terminalReady.Status,
				Reason:             terminalReady.Reason,
				Message:            "additional detail",
				LastTransitionTime: terminalReady.LastTransitionTime,
			}}},
			expectedErrs: field.ErrorList{field.Forbidden(statusPath.Child("conditions"), "cannot change or remove a terminal Ready condition").MarkFromImperative()},
		},
		"terminal outcome cannot change": {
			oldStatus: node.PodCheckpointStatus{Conditions: []metav1.Condition{terminalReady}},
			newStatus: node.PodCheckpointStatus{Conditions: []metav1.Condition{{
				Type:               terminalReady.Type,
				Status:             metav1.ConditionFalse,
				Reason:             node.PodCheckpointReasonFailed,
				LastTransitionTime: terminalReady.LastTransitionTime,
			}}},
			expectedErrs: field.ErrorList{field.Forbidden(statusPath.Child("conditions"), "cannot change or remove a terminal Ready condition").MarkFromImperative()},
		},
		"terminal checkpoint data cannot change": {
			oldStatus: node.PodCheckpointStatus{
				Conditions:         []metav1.Condition{terminalReady},
				CheckpointLocation: location("node-a/checkpoint"),
			},
			newStatus: node.PodCheckpointStatus{
				Conditions:         []metav1.Condition{terminalReady},
				CheckpointLocation: location("node-b/checkpoint"),
			},
			expectedErrs: field.ErrorList{
				field.Invalid(locationPath, nil, "field cannot be modified once set").WithOrigin("update"),
				field.Forbidden(statusPath, "cannot change checkpoint data after a terminal outcome").MarkFromImperative(),
			},
		},
		"status cannot be cleared": {
			oldStatus: node.PodCheckpointStatus{
				CheckpointedContainers:  []node.PodCheckpointContainerStatus{{Name: "app"}},
				CheckpointedPodTemplate: mkTemplateWithContainers("app"),
			},
			newStatusNil: true,
			expectedErrs: field.ErrorList{field.Invalid(statusPath, nil, "field cannot be cleared once set").WithOrigin("update")},
		},
		"checkpointed containers cannot be removed": {
			oldStatus: node.PodCheckpointStatus{
				CheckpointedContainers:  []node.PodCheckpointContainerStatus{{Name: "app"}},
				CheckpointedPodTemplate: mkTemplateWithContainers("app"),
			},
			expectedErrs: field.ErrorList{
				field.Invalid(templatePath, nil, "field cannot be cleared once set").WithOrigin("update"),
				field.Forbidden(containersPath, "item may not be removed").WithOrigin("update"),
				field.Forbidden(containersPath, "field is immutable once set").MarkFromImperative(),
			},
		},
		"terminal checkpoint container order is immaterial": {
			oldStatus: node.PodCheckpointStatus{
				Conditions:             []metav1.Condition{terminalReady},
				CheckpointedContainers: []node.PodCheckpointContainerStatus{{Name: "app"}, {Name: "sidecar"}},
			},
			newStatus: node.PodCheckpointStatus{
				Conditions:             []metav1.Condition{terminalReady},
				CheckpointedContainers: []node.PodCheckpointContainerStatus{{Name: "sidecar"}, {Name: "app"}},
			},
		},
	}
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := requestContext(apiVersion, "update", "status")
			for name, tc := range tests {
				t.Run(name, func(t *testing.T) {
					oldPC := mkPodCheckpoint()
					oldPC.ResourceVersion = "1"
					oldPC.Status = &tc.oldStatus
					newPC := oldPC.DeepCopy()
					if tc.newStatusNil {
						newPC.Status = nil
					} else {
						newPC.Status = &tc.newStatus
					}
					apitesting.VerifyUpdateValidationEquivalence(t, ctx, newPC, oldPC, registry.StatusStrategy, tc.expectedErrs, apitesting.WithSubResources("status"))
				})
			}
			for _, tc := range meta.GenerateConditionTestCases(statusPath.Child("conditions")) {
				t.Run("conditions: "+tc.Name, func(t *testing.T) {
					oldPC := mkPodCheckpoint()
					oldPC.ResourceVersion = "1"
					oldPC.Status = &node.PodCheckpointStatus{}
					newPC := oldPC.DeepCopy()
					newPC.Status.Conditions = tc.Conditions
					apitesting.VerifyUpdateValidationEquivalence(t, ctx, newPC, oldPC, registry.StatusStrategy, tc.ExpectedErrs, apitesting.WithSubResources("status"))
				})
			}
		})
	}
}
