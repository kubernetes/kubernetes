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
	"fmt"
	"maps"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/resource"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"
)

func TestPodSpecChangedFields(t *testing.T) {
	base := &v1.PodSpec{
		Containers: []v1.Container{{
			Name: "app", Image: "image:original",
			Args: []string{"original-secret-argument"},
			Env:  []v1.EnvVar{{Name: "PASSWORD", Value: "original-secret-value"}},
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
			},
			ReadinessProbe: &v1.Probe{ProbeHandler: v1.ProbeHandler{
				HTTPGet: &v1.HTTPGetAction{Port: intstr.FromInt32(80)},
			}},
		}},
		NodeSelector: map[string]string{"disk": "ssd", "zone": "a"},
		Volumes:      []v1.Volume{{Name: "data", VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}}},
	}
	for _, tc := range []struct {
		name   string
		mutate func(*v1.PodSpec)
		paths  []string
	}{
		{name: "equal", mutate: func(*v1.PodSpec) {}},
		{name: "equivalent quantities and empty collections", mutate: func(spec *v1.PodSpec) {
			spec.Containers[0].Resources.Requests[v1.ResourceCPU] = resource.MustParse("1000m")
			spec.Containers[0].Ports = []v1.ContainerPort{}
			spec.Containers[0].Resources.Limits = v1.ResourceList{}
		}},
		{name: "multiple fields", mutate: func(spec *v1.PodSpec) {
			spec.Containers[0].Image = "image:changed"
			spec.Containers[0].Args[0] = "changed-secret-argument"
			spec.Containers[0].Env[0].Value = "changed-secret-value"
		}, paths: []string{"spec.containers[0].args[0]", "spec.containers[0].env[0].value", "spec.containers[0].image"}},
		{name: "changed quantity", mutate: func(spec *v1.PodSpec) {
			spec.Containers[0].Resources.Requests[v1.ResourceCPU] = resource.MustParse("2")
		}, paths: []string{"spec.containers[0].resources.requests[cpu]"}},
		{name: "JSON scalar", mutate: func(spec *v1.PodSpec) {
			spec.Containers[0].ReadinessProbe.HTTPGet.Port = intstr.FromString("http")
		}, paths: []string{"spec.containers[0].readinessProbe.httpGet.port"}},
		{name: "map additions removals and changes", mutate: func(spec *v1.PodSpec) {
			delete(spec.NodeSelector, "disk")
			spec.NodeSelector["region"] = "private-region"
			spec.NodeSelector["zone"] = "private-zone"
		}, paths: []string{"spec.nodeSelector[disk]", "spec.nodeSelector[region]", "spec.nodeSelector[zone]"}},
		{name: "added list element", mutate: func(spec *v1.PodSpec) {
			spec.Containers[0].Env = append(spec.Containers[0].Env, v1.EnvVar{Name: "TOKEN", Value: "changed-secret-value"})
		}, paths: []string{"spec.containers[0].env[1]"}},
		{name: "removed list element", mutate: func(spec *v1.PodSpec) {
			spec.Containers[0].Args = nil
		}, paths: []string{"spec.containers[0].args[0]"}},
		{name: "added pointer", mutate: func(spec *v1.PodSpec) {
			spec.SecurityContext = &v1.PodSecurityContext{RunAsNonRoot: new(true)}
		}, paths: []string{"spec.securityContext"}},
		{name: "removed pointer", mutate: func(spec *v1.PodSpec) {
			spec.Containers[0].ReadinessProbe = nil
		}, paths: []string{"spec.containers[0].readinessProbe"}},
		{name: "embedded fields", mutate: func(spec *v1.PodSpec) {
			spec.Volumes[0].EmptyDir.Medium = v1.StorageMediumMemory
		}, paths: []string{"spec.volumes[0].emptyDir.medium"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want, got := base.DeepCopy(), base.DeepCopy()
			tc.mutate(got)
			wantBefore, gotBefore := want.DeepCopy(), got.DeepCopy()
			paths := PodSpecChangedFields(want, got)
			require.Equal(t, tc.paths, paths)
			require.Equal(t, wantBefore, want)
			require.Equal(t, gotBefore, got)
			for _, value := range []string{"original-secret", "changed-secret", "private-region", "private-zone"} {
				require.NotContains(t, strings.Join(paths, ", "), value)
			}
		})
	}
	t.Run("nil specs", func(t *testing.T) {
		require.Empty(t, PodSpecChangedFields(nil, nil))
		require.Equal(t, []string{"spec"}, PodSpecChangedFields(nil, base))
		require.Equal(t, []string{"spec"}, PodSpecChangedFields(base, nil))
	})
}

func TestPodSpecChangedFieldsHTTPProtocol(t *testing.T) {
	containers := []struct {
		name string
		spec func(v1.Container) *v1.PodSpec
	}{
		{name: "containers", spec: func(container v1.Container) *v1.PodSpec {
			return &v1.PodSpec{Containers: []v1.Container{container}}
		}},
		{name: "initContainers", spec: func(container v1.Container) *v1.PodSpec {
			return &v1.PodSpec{InitContainers: []v1.Container{container}}
		}},
		{name: "ephemeralContainers", spec: func(container v1.Container) *v1.PodSpec {
			return &v1.PodSpec{EphemeralContainers: []v1.EphemeralContainer{{EphemeralContainerCommon: v1.EphemeralContainerCommon{
				Name: container.Name, Image: container.Image,
				ReadinessProbe: container.ReadinessProbe, LivenessProbe: container.LivenessProbe,
				StartupProbe: container.StartupProbe, Lifecycle: container.Lifecycle,
			}}}}
		}},
	}
	actions := []struct {
		name string
		set  func(*v1.Container, *v1.HTTPGetAction)
	}{
		{name: "readinessProbe.httpGet", set: func(container *v1.Container, action *v1.HTTPGetAction) {
			container.ReadinessProbe = &v1.Probe{ProbeHandler: v1.ProbeHandler{HTTPGet: action}}
		}},
		{name: "livenessProbe.httpGet", set: func(container *v1.Container, action *v1.HTTPGetAction) {
			container.LivenessProbe = &v1.Probe{ProbeHandler: v1.ProbeHandler{HTTPGet: action}}
		}},
		{name: "startupProbe.httpGet", set: func(container *v1.Container, action *v1.HTTPGetAction) {
			container.StartupProbe = &v1.Probe{ProbeHandler: v1.ProbeHandler{HTTPGet: action}}
		}},
		{name: "lifecycle.postStart.httpGet", set: func(container *v1.Container, action *v1.HTTPGetAction) {
			container.Lifecycle = &v1.Lifecycle{PostStart: &v1.LifecycleHandler{HTTPGet: action}}
		}},
		{name: "lifecycle.preStop.httpGet", set: func(container *v1.Container, action *v1.HTTPGetAction) {
			container.Lifecycle = &v1.Lifecycle{PreStop: &v1.LifecycleHandler{HTTPGet: action}}
		}},
	}
	for _, container := range containers {
		for _, action := range actions {
			for _, tc := range []struct {
				name      string
				want, got *v1.HTTPProtocol
				equal     bool
			}{
				{name: "both omitted", equal: true},
				{name: "omitted and HTTP1", got: ptr.To(v1.HTTPProtocolHTTP1), equal: true},
				{name: "both HTTP1", want: ptr.To(v1.HTTPProtocolHTTP1), got: ptr.To(v1.HTTPProtocolHTTP1), equal: true},
				{name: "both HTTP2", want: ptr.To(v1.HTTPProtocolHTTP2), got: ptr.To(v1.HTTPProtocolHTTP2), equal: true},
				{name: "omitted and HTTP2", got: ptr.To(v1.HTTPProtocolHTTP2)},
				{name: "HTTP1 and HTTP2", want: ptr.To(v1.HTTPProtocolHTTP1), got: ptr.To(v1.HTTPProtocolHTTP2)},
				{name: "omitted and explicit empty", got: ptr.To(v1.HTTPProtocol(""))},
				{name: "omitted and unknown", got: ptr.To(v1.HTTPProtocol("private-protocol"))},
				{name: "HTTP1 and explicit empty", want: ptr.To(v1.HTTPProtocolHTTP1), got: ptr.To(v1.HTTPProtocol(""))},
				{name: "both explicit empty", want: ptr.To(v1.HTTPProtocol("")), got: ptr.To(v1.HTTPProtocol("")), equal: true},
			} {
				t.Run(container.name+"/"+action.name+"/"+tc.name, func(t *testing.T) {
					wantAction := &v1.HTTPGetAction{Port: intstr.FromInt32(80), Protocol: tc.want}
					gotAction := &v1.HTTPGetAction{Port: intstr.FromInt32(80), Protocol: tc.got}
					wantContainer, gotContainer := v1.Container{Name: "app", Image: "image"}, v1.Container{Name: "app", Image: "image"}
					action.set(&wantContainer, wantAction)
					action.set(&gotContainer, gotAction)
					want, got := container.spec(wantContainer), container.spec(gotContainer)
					wantBefore, gotBefore := want.DeepCopy(), got.DeepCopy()
					var expected []string
					if !tc.equal {
						expected = []string{fmt.Sprintf("spec.%s[0].%s.protocol", container.name, action.name)}
					}
					paths := PodSpecChangedFields(want, got)
					require.Equal(t, expected, paths)
					require.Equal(t, expected, PodSpecChangedFields(got, want))
					require.Empty(t, PodSpecChangedFields(want, want))
					require.Equal(t, wantBefore, want)
					require.Equal(t, gotBefore, got)
					require.Same(t, tc.want, wantAction.Protocol, "comparison replaced the captured protocol pointer")
					require.Same(t, tc.got, gotAction.Protocol, "comparison replaced the destination protocol pointer")
					require.NotContains(t, strings.Join(paths, ", "), "private-protocol")
				})
			}
		}
	}
	// This equivalence belongs only to checkpoint comparison.
	require.False(t, equality.Semantic.DeepEqual((*v1.HTTPProtocol)(nil), ptr.To(v1.HTTPProtocolHTTP1)))
}

func TestPodSpecChangedFieldsHTTPProtocolRetainsOtherChanges(t *testing.T) {
	base := sourcePod().Spec.DeepCopy()
	base.Containers[0].ReadinessProbe = &v1.Probe{ProbeHandler: v1.ProbeHandler{
		HTTPGet: &v1.HTTPGetAction{Port: intstr.FromInt32(80)},
	}}
	base.Containers[0].SecurityContext = &v1.SecurityContext{RunAsUser: ptr.To[int64](1000)}
	base.Containers[0].Resources.Requests = v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")}
	for _, tc := range []struct {
		name   string
		mutate func(*v1.Container)
		paths  []string
	}{
		{name: "equivalent quantity", mutate: func(container *v1.Container) {
			container.Resources.Requests[v1.ResourceCPU] = resource.MustParse("1000m")
		}},
		{name: "different credentials", mutate: func(container *v1.Container) {
			container.SecurityContext.RunAsUser = ptr.To[int64](0)
		}, paths: []string{"spec.containers[0].securityContext.runAsUser"}},
		{name: "omitted credentials", mutate: func(container *v1.Container) {
			container.SecurityContext.RunAsUser = nil
		}, paths: []string{"spec.containers[0].securityContext.runAsUser"}},
		{name: "different resources", mutate: func(container *v1.Container) {
			container.Resources.Requests[v1.ResourceCPU] = resource.MustParse("2")
		}, paths: []string{"spec.containers[0].resources.requests[cpu]"}},
		{name: "different port", mutate: func(container *v1.Container) {
			container.ReadinessProbe.HTTPGet.Port = intstr.FromInt32(81)
		}, paths: []string{"spec.containers[0].readinessProbe.httpGet.port"}},
		{name: "different port and protocol", mutate: func(container *v1.Container) {
			container.ReadinessProbe.HTTPGet.Port = intstr.FromInt32(81)
			container.ReadinessProbe.HTTPGet.Protocol = ptr.To(v1.HTTPProtocol("private-protocol"))
		}, paths: []string{"spec.containers[0].readinessProbe.httpGet.port", "spec.containers[0].readinessProbe.httpGet.protocol"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want, got := base.DeepCopy(), base.DeepCopy()
			got.Containers[0].ReadinessProbe.HTTPGet.Protocol = ptr.To(v1.HTTPProtocolHTTP1)
			tc.mutate(&got.Containers[0])
			wantBefore, gotBefore := want.DeepCopy(), got.DeepCopy()
			require.Equal(t, tc.paths, PodSpecChangedFields(want, got))
			require.Equal(t, tc.paths, PodSpecChangedFields(got, want))
			require.Equal(t, wantBefore, want)
			require.Equal(t, gotBefore, got)
		})
	}
}

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
			RestoreFrom:  &v1.CheckpointReference{Name: "ckpt"},
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

	// Node-local scheduling state and restoreFrom must be cleared.
	if tmpl.Spec.NodeName != "" {
		t.Errorf("spec.NodeName not cleared: %q", tmpl.Spec.NodeName)
	}
	if tmpl.Spec.RestoreFrom != nil {
		t.Errorf("spec.RestoreFrom not cleared: %v", *tmpl.Spec.RestoreFrom)
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
	if pod.Spec.RestoreFrom == nil {
		t.Errorf("source pod spec.RestoreFrom was mutated (cleared)")
	} else if pod.Spec.RestoreFrom.Name != "ckpt" {
		t.Errorf("source pod spec.RestoreFrom.Name was mutated: %q", pod.Spec.RestoreFrom.Name)
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

func TestSanitizePodTemplatePreservesSourceRestorePin(t *testing.T) {
	pod := sourcePod()
	pod.Spec.Affinity = &v1.Affinity{
		NodeAffinity: &v1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
				NodeSelectorTerms: []v1.NodeSelectorTerm{{
					MatchFields: []v1.NodeSelectorRequirement{{
						Key:      "metadata.name",
						Operator: v1.NodeSelectorOpIn,
						Values:   []string{"node-1"},
					}},
				}},
			},
		},
	}

	tmpl := SanitizePodTemplate(pod)
	require.Equal(t, pod.Spec.Affinity, tmpl.Spec.Affinity)
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

func TestRestoreNodeAffinityPinIsIdempotent(t *testing.T) {
	pin := v1.NodeSelectorRequirement{Key: "metadata.name", Operator: v1.NodeSelectorOpIn, Values: []string{"node-1"}}
	for _, selector := range []*v1.NodeSelector{
		nil, {}, {NodeSelectorTerms: []v1.NodeSelectorTerm{{}}},
		{NodeSelectorTerms: []v1.NodeSelectorTerm{{MatchFields: []v1.NodeSelectorRequirement{pin}}}},
		{NodeSelectorTerms: []v1.NodeSelectorTerm{{}, {MatchExpressions: []v1.NodeSelectorRequirement{{Key: "zone", Operator: v1.NodeSelectorOpIn, Values: []string{"prod"}}}}}},
	} {
		pod := sourcePod()
		pod.Spec.Affinity = &v1.Affinity{NodeAffinity: &v1.NodeAffinity{RequiredDuringSchedulingIgnoredDuringExecution: selector}}
		original := pod.DeepCopy()
		template := SanitizePodTemplate(pod)
		require.Equal(t, original.Spec.Affinity, template.Spec.Affinity, "capture must preserve the complete source expression")
		spec := template.Spec.DeepCopy()
		AddRestoreNodeAffinity(spec, "node-1")
		first := spec.DeepCopy()
		AddRestoreNodeAffinity(spec, "node-1")
		require.Equal(t, first, spec, "reinvocation must not duplicate the pin")
		require.Equal(t, original, pod)
		require.Equal(t, original.Spec.Affinity, template.Spec.Affinity)
	}
}
