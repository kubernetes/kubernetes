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

package validation

import (
	"testing"

	"github.com/stretchr/testify/require"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	"k8s.io/kubernetes/pkg/apis/core"
	nodeapi "k8s.io/kubernetes/pkg/apis/node"
	"k8s.io/utils/ptr"
)

func newPodCheckpoint() *nodeapi.PodCheckpoint {
	return &nodeapi.PodCheckpoint{
		ObjectMeta: metav1.ObjectMeta{Name: "cp-1", Namespace: "ns"},
		Spec: nodeapi.PodCheckpointSpec{
			SourcePod: nodeapi.PodReference{Name: "my-app"},
		},
	}
}

func TestValidatePodCheckpointCapturedTemplate(t *testing.T) {
	valid := &core.PodTemplateSpec{Spec: podtest.MakePodSpec()}
	invalid := valid.DeepCopy()
	invalid.Spec.Containers[0].Image = ""
	for _, tc := range []struct {
		name                     string
		oldTemplate, newTemplate *core.PodTemplateSpec
		wantError                bool
	}{
		{name: "unset"},
		{name: "initial valid template", newTemplate: valid},
		{name: "initial invalid template", newTemplate: invalid, wantError: true},
		{name: "unchanged invalid template", oldTemplate: invalid, newTemplate: invalid},
	} {
		t.Run(tc.name, func(t *testing.T) {
			old := newPodCheckpoint()
			old.ResourceVersion = "1"
			old.Status.CheckpointedPodTemplate = tc.oldTemplate
			updated := old.DeepCopy()
			updated.Status.CheckpointedPodTemplate = tc.newTemplate
			updated.Status.CompletionTime = new(metav1.Now())
			errs := ValidatePodCheckpointStatusUpdate(updated, old)
			if got := len(errs) != 0; got != tc.wantError {
				t.Fatalf("validation errors = %v, want error %t", errs, tc.wantError)
			}
			if tc.wantError && errs[0].Field != "status.checkpointedPodTemplate.spec.containers[0].image" {
				t.Errorf("validation field = %q, want captured container image", errs[0].Field)
			}
		})
	}
}

func TestValidatePodCheckpointStatusUpdate(t *testing.T) {
	old := newPodCheckpoint()
	old.ResourceVersion = "1"

	valid := old.DeepCopy()
	valid.Status.Conditions = []metav1.Condition{{
		Type:               nodeapi.PodCheckpointConditionReady,
		Status:             metav1.ConditionTrue,
		Reason:             nodeapi.PodCheckpointReasonCompleted,
		LastTransitionTime: metav1.Now(),
	}}
	if errs := ValidatePodCheckpointStatusUpdate(valid, old); len(errs) != 0 {
		t.Errorf("expected no errors for valid status, got %v", errs)
	}

	bad := old.DeepCopy()
	bad.Status.Conditions = []metav1.Condition{{
		Type:   nodeapi.PodCheckpointConditionReady,
		Status: "Maybe", // not a valid ConditionStatus
		Reason: nodeapi.PodCheckpointReasonCompleted,
	}}
	if errs := ValidatePodCheckpointStatusUpdate(bad, old); len(errs) == 0 {
		t.Errorf("expected errors for invalid condition, got none")
	}
}

func TestPodCheckpointTerminalOutcome(t *testing.T) {
	for _, reason := range []string{nodeapi.PodCheckpointReasonCompleted, nodeapi.PodCheckpointReasonFailed, nodeapi.PodCheckpointReasonSourcePodReplaced} {
		t.Run(reason, func(t *testing.T) {
			old := newPodCheckpoint()
			old.ResourceVersion = "1"
			status := metav1.ConditionFalse
			if reason == nodeapi.PodCheckpointReasonCompleted {
				status = metav1.ConditionTrue
			}
			old.Status.Conditions = []metav1.Condition{{Type: nodeapi.PodCheckpointConditionReady, Status: status, Reason: reason, LastTransitionTime: metav1.Now()}}
			for _, tc := range []struct {
				name    string
				mutate  func(*nodeapi.PodCheckpoint)
				allowed bool
			}{
				{name: "unchanged", allowed: true, mutate: func(*nodeapi.PodCheckpoint) {}},
				{name: "diagnostic", allowed: true, mutate: func(p *nodeapi.PodCheckpoint) { p.Status.Conditions[0].Message = "more detail" }},
				{name: "other condition first", allowed: true, mutate: func(p *nodeapi.PodCheckpoint) {
					p.Status.Conditions = append([]metav1.Condition{{Type: "Example", Status: metav1.ConditionTrue, Reason: "Example", LastTransitionTime: metav1.Now()}}, p.Status.Conditions...)
				}},
				{name: "clear", mutate: func(p *nodeapi.PodCheckpoint) { p.Status.Conditions = nil }},
				{name: "change status", mutate: func(p *nodeapi.PodCheckpoint) { p.Status.Conditions[0].Status = metav1.ConditionUnknown }},
				{name: "change reason", mutate: func(p *nodeapi.PodCheckpoint) { p.Status.Conditions[0].Reason = "Retry" }},
				{name: "location", mutate: func(p *nodeapi.PodCheckpoint) { p.Status.CheckpointLocation = &nodeapi.CheckpointSource{} }},
				{name: "template", mutate: func(p *nodeapi.PodCheckpoint) { p.Status.CheckpointedPodTemplate = &core.PodTemplateSpec{} }},
				{name: "containers", mutate: func(p *nodeapi.PodCheckpoint) {
					p.Status.CheckpointedContainers = []nodeapi.PodCheckpointContainerStatus{{Name: "app"}}
				}},
				{name: "node", mutate: func(p *nodeapi.PodCheckpoint) { p.Status.NodeName = new("node") }},
				{name: "source UID", mutate: func(p *nodeapi.PodCheckpoint) { p.Status.SourcePodUID = ptr.To(types.UID("source")) }},
				{name: "completion", mutate: func(p *nodeapi.PodCheckpoint) { p.Status.CompletionTime = new(metav1.Now()) }},
			} {
				t.Run(tc.name, func(t *testing.T) {
					update := old.DeepCopy()
					tc.mutate(update)
					errs := ValidatePodCheckpointStatusUpdate(update, old)
					if tc.allowed {
						require.Empty(t, errs)
					} else {
						require.NotEmpty(t, errs)
					}
				})
			}
		})
	}
}

func TestPodCheckpointTerminalContainerOrder(t *testing.T) {
	old := newPodCheckpoint()
	old.ResourceVersion = "1"
	old.Status.Conditions = []metav1.Condition{{Type: nodeapi.PodCheckpointConditionReady, Status: metav1.ConditionTrue, Reason: nodeapi.PodCheckpointReasonCompleted, LastTransitionTime: metav1.Now()}}
	old.Status.CheckpointedContainers = []nodeapi.PodCheckpointContainerStatus{{Name: "app"}, {Name: "sidecar"}}
	update := old.DeepCopy()
	update.Status.CheckpointedContainers[0], update.Status.CheckpointedContainers[1] = update.Status.CheckpointedContainers[1], update.Status.CheckpointedContainers[0]
	originalOld, originalUpdate := old.DeepCopy(), update.DeepCopy()
	require.Empty(t, ValidatePodCheckpointStatusUpdate(update, old))
	require.Equal(t, originalOld, old)
	require.Equal(t, originalUpdate, update)
	update.Status.CheckpointedContainers[0].Name = "other"
	require.NotEmpty(t, ValidatePodCheckpointStatusUpdate(update, old))
}

func TestPodCheckpointedContainersMatchCapturedTemplate(t *testing.T) {
	template := &core.PodTemplateSpec{Spec: podtest.MakePodSpec()}
	template.Spec.InitContainers = append(template.Spec.InitContainers, core.Container{
		Name:          "sidecar",
		RestartPolicy: ptr.To(core.ContainerRestartPolicyAlways),
	})
	pc := newPodCheckpoint()
	pc.ResourceVersion = "1"
	pc.Status.CheckpointedPodTemplate = template
	pc.Status.CheckpointedContainers = []nodeapi.PodCheckpointContainerStatus{{Name: template.Spec.Containers[0].Name}, {Name: "sidecar"}}
	old := pc.DeepCopy()
	old.ResourceVersion = "1"
	old.Status.CheckpointedContainers = nil
	require.Empty(t, ValidatePodCheckpointStatusUpdate(pc, old))

	pc.Status.CheckpointedContainers = append(pc.Status.CheckpointedContainers, nodeapi.PodCheckpointContainerStatus{Name: "unexpected"})
	require.NotEmpty(t, ValidatePodCheckpointStatusUpdate(pc, old))
}
