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
	"context"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apiserverfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	"k8s.io/kubernetes/pkg/apis/core"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	corev1 "k8s.io/kubernetes/pkg/apis/core/v1"
	nodeapi "k8s.io/kubernetes/pkg/apis/node"
	_ "k8s.io/kubernetes/pkg/apis/node/install"
	nodeutil "k8s.io/kubernetes/pkg/apis/node/util"
	nodevalidation "k8s.io/kubernetes/pkg/apis/node/validation"
	"k8s.io/kubernetes/pkg/features"
	podregistry "k8s.io/kubernetes/pkg/registry/core/pod"
	"sigs.k8s.io/structured-merge-diff/v7/fieldpath"
)

func newPodCheckpoint() *nodeapi.PodCheckpoint {
	return &nodeapi.PodCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "checkpoint-1",
			Namespace: "default",
		},
		Spec: nodeapi.PodCheckpointSpec{
			SourcePod: nodeapi.PodReference{Name: "my-pod"},
		},
		Status: &nodeapi.PodCheckpointStatus{
			NodeName: new("node-1"),
		},
	}
}

func TestSelectableFields(t *testing.T) {
	pc := newPodCheckpoint()
	set := SelectableFields(pc)

	if got := set["spec.sourcePod.name"]; got != "my-pod" {
		t.Errorf("spec.sourcePod.name: expected %q, got %q", "my-pod", got)
	}
	if got := set["status.nodeName"]; got != "node-1" {
		t.Errorf("status.nodeName: expected %q, got %q", "node-1", got)
	}
	if got := set["metadata.name"]; got != "checkpoint-1" {
		t.Errorf("metadata.name: expected %q, got %q", "checkpoint-1", got)
	}
	if got := set["metadata.namespace"]; got != "default" {
		t.Errorf("metadata.namespace: expected %q, got %q", "default", got)
	}
}

func TestGetAttrs(t *testing.T) {
	pc := newPodCheckpoint()
	pc.ObjectMeta.Labels = map[string]string{"team": "blue"}

	lbls, flds, err := GetAttrs(pc)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if v := lbls["team"]; v != "blue" {
		t.Errorf("label team: expected %q, got %q", "blue", v)
	}
	if v := flds["spec.sourcePod.name"]; v != "my-pod" {
		t.Errorf("spec.sourcePod.name: expected %q, got %q", "my-pod", v)
	}
	if v := flds["status.nodeName"]; v != "node-1" {
		t.Errorf("status.nodeName: expected %q, got %q", "node-1", v)
	}
}

func TestMatchPodCheckpoint(t *testing.T) {
	testCases := []struct {
		name          string
		fieldSelector fields.Selector
		matches       bool
	}{
		{
			name:          "match on spec.sourcePod.name",
			fieldSelector: fields.SelectorFromSet(fields.Set{"spec.sourcePod.name": "my-pod"}),
			matches:       true,
		},
		{
			name:          "match on status.nodeName",
			fieldSelector: fields.SelectorFromSet(fields.Set{"status.nodeName": "node-1"}),
			matches:       true,
		},
		{
			name:          "match on both fields",
			fieldSelector: fields.SelectorFromSet(fields.Set{"spec.sourcePod.name": "my-pod", "status.nodeName": "node-1"}),
			matches:       true,
		},
		{
			name:          "no match on spec.sourcePod.name",
			fieldSelector: fields.SelectorFromSet(fields.Set{"spec.sourcePod.name": "other-pod"}),
			matches:       false,
		},
		{
			name:          "no match on status.nodeName",
			fieldSelector: fields.SelectorFromSet(fields.Set{"status.nodeName": "node-2"}),
			matches:       false,
		},
	}

	pc := newPodCheckpoint()
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			predicate := MatchPodCheckpoint(labels.Everything(), tc.fieldSelector)
			got, err := predicate.Matches(pc)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.matches {
				t.Errorf("expected matches=%v, got %v", tc.matches, got)
			}
		})
	}
}

// The status endpoint's reset fields must cover everything PrepareForUpdate
// discards, or server-side apply records ownership of values that are dropped.
func TestStatusStrategyResetFields(t *testing.T) {
	reset := StatusStrategy.GetResetFields()["node.k8s.io/v1alpha1"]
	if reset == nil {
		t.Fatal("expected reset fields for node.k8s.io/v1alpha1")
	}
	for _, path := range []string{"metadata", "spec"} {
		if !reset.Has(fieldpath.MakePathOrDie(path)) {
			t.Errorf("expected status reset fields to include %q", path)
		}
	}
}

func TestPodCheckpointCapturedTemplateValidation(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.EvictionRequestAPI, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SecurityProfileOCI, true)
	responders := make([]core.EvictionResponder, 11)
	for i := range responders {
		responders[i] = core.EvictionResponder{Name: fmt.Sprintf("example.com/r%d", i), Priority: new(int32(1000))}
	}
	ociProfile := func() *core.SeccompProfile {
		return &core.SeccompProfile{Type: core.SeccompProfileTypeOCI, OCI: &core.SecurityProfileOCI{
			Ref:         "registry.example.com/seccomp/profile@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
			BaseProfile: &core.SecurityProfileOCIBase{Type: core.SecurityProfileOCIBaseTypeLocalhost, LocalhostProfile: new("profile.json")},
		}}
	}
	for _, beta := range []bool{true, false} {
		t.Run(fmt.Sprintf("beta=%t", beta), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, apiserverfeatures.DeclarativeValidationBeta, beta)
			for _, tc := range []struct {
				name      string
				mutate    func(*core.PodTemplateSpec)
				existing  bool
				clear     bool
				replace   bool
				wantField string
			}{
				{name: "valid"},
				{name: "maximum responders", mutate: func(p *core.PodTemplateSpec) { p.Spec.EvictionResponders = responders[:10] }},
				{name: "too many responders", mutate: func(p *core.PodTemplateSpec) { p.Spec.EvictionResponders = responders }, wantField: "status.checkpointedPodTemplate.spec.evictionResponders"},
				{name: "reserved responder priority", mutate: func(p *core.PodTemplateSpec) {
					p.Spec.EvictionResponders = []core.EvictionResponder{{Name: "example.com/worker", Priority: new(int32(1))}}
				}, wantField: "status.checkpointedPodTemplate.spec.evictionResponders[0].priority"},
				{name: "handwritten image validation", mutate: func(p *core.PodTemplateSpec) { p.Spec.Containers[0].Image = "" }, wantField: "status.checkpointedPodTemplate.spec.containers[0].image"},
				{name: "size limit", mutate: func(p *core.PodTemplateSpec) {
					p.Spec.Containers[0].Env = []core.EnvVar{{Name: "DATA", Value: strings.Repeat("x", nodevalidation.MaxCheckpointedPodTemplateBytes)}}
				}, wantField: "status.checkpointedPodTemplate"},
				{name: "valid OCI profile", mutate: func(p *core.PodTemplateSpec) {
					p.Spec.SecurityContext = &core.PodSecurityContext{SeccompProfile: ociProfile()}
				}},
				{name: "OCI base path", mutate: func(p *core.PodTemplateSpec) {
					p.Spec.SecurityContext = &core.PodSecurityContext{SeccompProfile: ociProfile()}
					p.Spec.SecurityContext.SeccompProfile.OCI.BaseProfile.LocalhostProfile = new("../profile.json")
				}, wantField: "status.checkpointedPodTemplate.spec.securityContext.seccompProfile.oci.baseProfile.localhostProfile"},
				{name: "unchanged valid template", existing: true},
				{name: "unchanged invalid template", existing: true, mutate: func(p *core.PodTemplateSpec) { p.Spec.Containers[0].Image = "" }},
				{name: "clearing template", existing: true, clear: true, wantField: "status.checkpointedPodTemplate"},
				{name: "replacing template", existing: true, replace: true, wantField: "status.checkpointedPodTemplate", mutate: func(p *core.PodTemplateSpec) { p.Spec.Containers[0].Image = "different" }},
				{name: "handwritten and generated overlap", mutate: func(p *core.PodTemplateSpec) {
					p.Spec.EvictionResponders = responders[:1]
					p.Spec.SchedulingGroup = &core.PodSchedulingGroup{PodGroupName: new("group")}
				}, wantField: "status.checkpointedPodTemplate.spec.schedulingGroup"},
			} {
				t.Run(tc.name, func(t *testing.T) {
					old := newPodCheckpoint()
					old.ResourceVersion = "1"
					template := &core.PodTemplateSpec{Spec: podtest.MakePodSpec()}
					if tc.mutate != nil {
						tc.mutate(template)
					}
					if tc.existing {
						old.Status.CheckpointedPodTemplate = template.DeepCopy()
						if tc.replace {
							old.Status.CheckpointedPodTemplate.Spec.Containers[0].Image = "original"
						}
					}
					updated := old.DeepCopy()
					updated.Status.CheckpointedPodTemplate = template
					updated.Status.CompletionTime = new(metav1.Now())
					if tc.clear {
						updated.Status.CheckpointedPodTemplate = nil
					}
					ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{APIGroup: "node.k8s.io", APIVersion: "v1alpha1", Resource: "podcheckpoints", Subresource: "status", IsResourceRequest: true, Verb: "update"})
					StatusStrategy.PrepareForUpdate(ctx, updated, old)
					errs := rest.ValidateUpdate(ctx, updated, old, StatusStrategy)
					if tc.wantField == "" {
						require.Empty(t, errs)
						return
					}
					require.Len(t, errs, 1)
					require.Equal(t, tc.wantField, errs[0].Field)
				})
			}
		})
	}
}

func TestPodCheckpointCapturedTemplateAfterSeccompAnnotationUpdate(t *testing.T) {
	for _, containerAnnotation := range []bool{false, true} {
		t.Run(fmt.Sprintf("container=%t", containerAnnotation), func(t *testing.T) {
			oldPod := podtest.MakePod("source", podtest.SetResourceVersion("1"), podtest.SetSecurityContext(&core.PodSecurityContext{SeccompProfile: &core.SeccompProfile{Type: core.SeccompProfileTypeRuntimeDefault}}))
			oldPod.Spec.Containers[0].SecurityContext = &core.SecurityContext{SeccompProfile: &core.SeccompProfile{Type: core.SeccompProfileTypeRuntimeDefault}}
			updatedPod := oldPod.DeepCopy()
			key := "seccomp.security.alpha.kubernetes.io/pod"
			if containerAnnotation {
				key = "container.seccomp.security.alpha.kubernetes.io/" + oldPod.Spec.Containers[0].Name
			}
			updatedPod.Annotations = map[string]string{key: "unconfined"}
			podCtx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{APIVersion: "v1", Resource: "pods", IsResourceRequest: true, Verb: "update"})
			podregistry.Strategy.PrepareForUpdate(podCtx, updatedPod, oldPod)
			require.Empty(t, rest.ValidateUpdate(podCtx, updatedPod, oldPod, podregistry.Strategy))
			var source v1.Pod
			require.NoError(t, corev1.Convert_core_Pod_To_v1_Pod(updatedPod, &source, nil))
			for _, invalidField := range []bool{false, true} {
				t.Run(fmt.Sprintf("invalidField=%t", invalidField), func(t *testing.T) {
					template := nodeutil.SanitizePodTemplate(&source)
					if invalidField {
						template.Spec.SecurityContext.SeccompProfile.Type = "Invalid"
					}
					old := newPodCheckpoint()
					old.ResourceVersion = "1"
					updated := old.DeepCopy()
					updated.Status.CheckpointedPodTemplate = &core.PodTemplateSpec{}
					require.NoError(t, corev1.Convert_v1_PodTemplateSpec_To_core_PodTemplateSpec(template, updated.Status.CheckpointedPodTemplate, nil))
					ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{APIGroup: "node.k8s.io", APIVersion: "v1alpha1", Resource: "podcheckpoints", Subresource: "status", IsResourceRequest: true, Verb: "update"})
					StatusStrategy.PrepareForUpdate(ctx, updated, old)
					errs := rest.ValidateUpdate(ctx, updated, old, StatusStrategy)
					if !invalidField {
						require.Empty(t, errs)
						return
					}
					require.Len(t, errs, 1)
					require.Equal(t, "status.checkpointedPodTemplate.spec.securityContext.seccompProfile.type", errs[0].Field)
				})
			}
		})
	}
}

func TestCheckpointRestoreProtectionOnCreate(t *testing.T) {
	pc := newPodCheckpoint()
	Strategy.PrepareForCreate(context.Background(), pc)
	if !slices.Contains(pc.Finalizers, nodeapi.PodCheckpointRestoreProtectionFinalizer) {
		t.Fatal("checkpoint was created without restore protection")
	}
	Strategy.PrepareForCreate(context.Background(), pc)
	if len(pc.Finalizers) != 1 {
		t.Fatal("restore protection was duplicated")
	}
}
