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

package util

import (
	"maps"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/resource"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

func sourcePod() *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "my-app",
			Namespace:       "team-a",
			UID:             types.UID("abc-123"),
			ResourceVersion: "456",
			Generation:      7,
			Labels:          map[string]string{"app": "my-app"},
			Annotations:     map[string]string{"team": "a"},
			OwnerReferences: []metav1.OwnerReference{{Kind: "ReplicaSet", Name: "rs-1"}},
			ManagedFields:   []metav1.ManagedFieldsEntry{{Manager: "kubelet"}},
		},
		Spec: v1.PodSpec{
			NodeName:     "node-1",
			NodeSelector: map[string]string{"disktype": "ssd"},
			Containers:   []v1.Container{{Name: "main", Image: "my-app:latest"}},
		},
		Status: v1.PodStatus{Phase: v1.PodRunning},
	}
}

func TestSanitizePodTemplate_StripsNodeLocalAndIdentityFields(t *testing.T) {
	pod := sourcePod()
	tmpl := SanitizePodTemplate(pod)

	if tmpl == nil {
		t.Fatal("expected non-nil template")
	}

	// Identity / server-set metadata must be cleared.
	if tmpl.UID != "" {
		t.Errorf("UID not cleared: %q", tmpl.UID)
	}
	if tmpl.ResourceVersion != "" {
		t.Errorf("ResourceVersion not cleared: %q", tmpl.ResourceVersion)
	}
	if tmpl.Generation != 0 {
		t.Errorf("Generation not cleared: %d", tmpl.Generation)
	}
	if tmpl.Name != "" || tmpl.Namespace != "" {
		t.Errorf("Name/Namespace not cleared: %q/%q", tmpl.Name, tmpl.Namespace)
	}
	if len(tmpl.ManagedFields) != 0 {
		t.Errorf("ManagedFields not cleared: %+v", tmpl.ManagedFields)
	}

	// Node-local scheduling state must be cleared.
	if tmpl.Spec.NodeName != "" {
		t.Errorf("spec.NodeName not cleared: %q", tmpl.Spec.NodeName)
	}

	// User-meaningful metadata and spec must be preserved.
	if tmpl.Labels["app"] != "my-app" {
		t.Errorf("labels not preserved: %+v", tmpl.Labels)
	}
	if tmpl.Annotations["team"] != "a" {
		t.Errorf("annotations not preserved: %+v", tmpl.Annotations)
	}
	if len(tmpl.OwnerReferences) != 1 || tmpl.OwnerReferences[0].Name != "rs-1" {
		t.Errorf("ownerReferences not preserved: %+v", tmpl.OwnerReferences)
	}
	if len(tmpl.Spec.Containers) != 1 || tmpl.Spec.Containers[0].Image != "my-app:latest" {
		t.Errorf("containers not preserved: %+v", tmpl.Spec.Containers)
	}
	// Placement constraints remain part of the captured workload.
	if tmpl.Spec.NodeSelector["disktype"] != "ssd" {
		t.Errorf("nodeSelector not preserved: %+v", tmpl.Spec.NodeSelector)
	}
}

func TestSanitizePodTemplatePreservesNodeIdentityConstraints(t *testing.T) {
	pod := sourcePod()
	pod.Spec.NodeSelector = map[string]string{
		v1.LabelHostname: "node-1",
		"disktype":       "ssd",
	}
	pod.Spec.Affinity = &v1.Affinity{
		NodeAffinity: &v1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
				NodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{
							{Key: v1.LabelHostname, Operator: v1.NodeSelectorOpIn, Values: []string{"node-1"}},
							{Key: "zone", Operator: v1.NodeSelectorOpIn, Values: []string{"z1"}},
						},
						MatchFields: []v1.NodeSelectorRequirement{
							{Key: "metadata.name", Operator: v1.NodeSelectorOpIn, Values: []string{"node-1"}},
						},
					},
				},
			},
		},
	}

	tmpl := SanitizePodTemplate(pod)
	require.Equal(t, pod.Spec.NodeSelector, tmpl.Spec.NodeSelector)
	require.Equal(t, pod.Spec.Affinity, tmpl.Spec.Affinity)
}

func TestSanitizePodTemplate_DoesNotMutateSource(t *testing.T) {
	pod := sourcePod()
	_ = SanitizePodTemplate(pod)

	if pod.Spec.NodeName != "node-1" {
		t.Errorf("source pod spec.NodeName was mutated: %q", pod.Spec.NodeName)
	}
	if pod.UID != types.UID("abc-123") {
		t.Errorf("source pod UID was mutated: %q", pod.UID)
	}
	if len(pod.ManagedFields) != 1 {
		t.Errorf("source pod ManagedFields was mutated: %+v", pod.ManagedFields)
	}
}

func TestSanitizePodTemplateDropsLegacySeccompAnnotations(t *testing.T) {
	retainedAnnotations := map[string]string{
		"example.com/keep": "value",
		"container.apparmor.security.beta.kubernetes.io/" + "main": "runtime/default",
		"seccomp.security.alpha.kubernetes.io/pod" + "-other":      "keep",
	}
	for _, tc := range []struct {
		name             string
		annotations      map[string]string
		podProfile       *v1.SeccompProfile
		containerProfile *v1.SeccompProfile
		initProfile      *v1.SeccompProfile
		nilAnnotations   bool
	}{
		{
			name:        "matching Pod annotation",
			annotations: map[string]string{"seccomp.security.alpha.kubernetes.io/pod": "runtime/default"},
			podProfile:  &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault},
		},
		{
			name:        "conflicting Pod annotation",
			annotations: map[string]string{"seccomp.security.alpha.kubernetes.io/pod": "unconfined"},
			podProfile:  &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault},
		},
		{
			name:        "annotation-only Pod profile",
			annotations: map[string]string{"seccomp.security.alpha.kubernetes.io/pod": "localhost/unused.json"},
		},
		{
			name:             "matching container annotation",
			annotations:      map[string]string{"container.seccomp.security.alpha.kubernetes.io/" + "main": "runtime/default"},
			containerProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault},
		},
		{
			name:             "conflicting container annotation",
			annotations:      map[string]string{"container.seccomp.security.alpha.kubernetes.io/" + "main": "unconfined"},
			podProfile:       &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault},
			containerProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost, LocalhostProfile: new("profiles/app.json")},
		},
		{
			name:        "annotation-only container profile",
			annotations: map[string]string{"container.seccomp.security.alpha.kubernetes.io/" + "main": "runtime/default"},
		},
		{
			name:        "init container annotation",
			annotations: map[string]string{"container.seccomp.security.alpha.kubernetes.io/" + "setup": "unconfined"},
			initProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault},
		},
		{
			name: "annotations for excluded or missing containers",
			annotations: map[string]string{
				"container.seccomp.security.alpha.kubernetes.io/" + "debug": "runtime/default",
				"container.seccomp.security.alpha.kubernetes.io/" + "gone":  "localhost/unused.json",
			},
		},
		{name: "no legacy annotations"},
		{name: "nil annotations", nilAnnotations: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pod := sourcePod()
			pod.Spec.InitContainers = []v1.Container{{Name: "setup", Image: "setup"}}
			pod.Spec.EphemeralContainers = []v1.EphemeralContainer{{EphemeralContainerCommon: v1.EphemeralContainerCommon{Name: "debug", Image: "debug"}}}
			if tc.podProfile != nil {
				pod.Spec.SecurityContext = &v1.PodSecurityContext{SeccompProfile: tc.podProfile.DeepCopy()}
			}
			if tc.containerProfile != nil {
				pod.Spec.Containers[0].SecurityContext = &v1.SecurityContext{SeccompProfile: tc.containerProfile.DeepCopy()}
			}
			if tc.initProfile != nil {
				pod.Spec.InitContainers[0].SecurityContext = &v1.SecurityContext{SeccompProfile: tc.initProfile.DeepCopy()}
			}
			pod.Annotations = nil
			if !tc.nilAnnotations {
				pod.Annotations = make(map[string]string)
				maps.Copy(pod.Annotations, retainedAnnotations)
				maps.Copy(pod.Annotations, tc.annotations)
			}
			original := pod.DeepCopy()

			template := SanitizePodTemplate(pod)
			if tc.nilAnnotations {
				require.Nil(t, template.Annotations)
			} else {
				require.Equal(t, retainedAnnotations, template.Annotations)
				template.Annotations["example.com/keep"] = "changed"
			}
			require.Equal(t, original.Spec.SecurityContext, template.Spec.SecurityContext)
			require.Equal(t, original.Spec.Containers[0].SecurityContext, template.Spec.Containers[0].SecurityContext)
			require.Equal(t, original.Spec.InitContainers[0].SecurityContext, template.Spec.InitContainers[0].SecurityContext)
			require.Equal(t, original, pod, "sanitization and changes to its output must not mutate the source")
		})
	}
}

func TestSanitizePodTemplatePreservesEmptyAffinity(t *testing.T) {
	pod := sourcePod()
	pod.Spec.Affinity = &v1.Affinity{}

	tmpl := SanitizePodTemplate(pod)
	require.Equal(t, pod.Spec.Affinity, tmpl.Spec.Affinity)
}

func TestSanitizePodTemplate_NilPod(t *testing.T) {
	if got := SanitizePodTemplate(nil); got != nil {
		t.Errorf("expected nil for nil pod, got %+v", got)
	}
}

func TestSanitizePodTemplateExcludesNonRestoredFields(t *testing.T) {
	pod := sourcePod()
	pod.Spec.EphemeralContainers = []v1.EphemeralContainer{{EphemeralContainerCommon: v1.EphemeralContainerCommon{Name: "debug", Image: "debug"}}}
	pod.Spec.SchedulingGates = []v1.PodSchedulingGate{{Name: "example.com/ready"}}
	pod.Spec.Containers[0].Resources.Requests = v1.ResourceList{v1.ResourceCPU: resource.MustParse("2")}
	pod.Spec.Tolerations = []v1.Toleration{{Key: "dedicated", Operator: v1.TolerationOpExists}}
	original := pod.DeepCopy()
	template := SanitizePodTemplate(pod)
	require.Empty(t, template.Spec.EphemeralContainers)
	require.Empty(t, template.Spec.SchedulingGates)
	require.Equal(t, pod.Spec.Containers[0].Resources, template.Spec.Containers[0].Resources)
	require.Equal(t, pod.Spec.Tolerations, template.Spec.Tolerations)
	require.Equal(t, original, pod)
}
