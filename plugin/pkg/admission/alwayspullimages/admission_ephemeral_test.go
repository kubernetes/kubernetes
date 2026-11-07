/*
Copyright 2026 The Kubernetes Authors.

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

package alwayspullimages

import (
	"context"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	admissiontesting "k8s.io/apiserver/pkg/admission/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	podregistry "k8s.io/kubernetes/pkg/registry/core/pod"
)

func ephemeralContainersAttributes(pod, oldPod *api.Pod, dryRun bool) admission.Attributes {
	options := &metav1.UpdateOptions{}
	if dryRun {
		options.DryRun = []string{metav1.DryRunAll}
	}
	return admission.NewAttributesRecord(pod, oldPod, api.Kind("Pod").WithVersion("v1"),
		pod.Namespace, pod.Name, api.Resource("pods").WithVersion("v1"), ephemeralcontainers,
		admission.Update, options, dryRun, nil)
}

func ephemeralContainer(name, image string, policy api.PullPolicy) api.EphemeralContainer {
	return api.EphemeralContainer{EphemeralContainerCommon: api.EphemeralContainerCommon{
		Name: name, Image: image, ImagePullPolicy: policy,
		TerminationMessagePath: "/dev/termination-log", TerminationMessagePolicy: api.TerminationMessageReadFile,
	}}
}

type ephemeralContainerUpdate struct {
	name         string
	old, updated []api.EphemeralContainer
	wantPolicies []api.PullPolicy
	wantErrors   []string
	dryRun       bool
}

func ephemeralContainerUpdates() []ephemeralContainerUpdate {
	return []ephemeralContainerUpdate{
		{
			name:         "new image with Never",
			updated:      []api.EphemeralContainer{ephemeralContainer("debugger", "debug", api.PullNever)},
			wantPolicies: []api.PullPolicy{api.PullAlways},
			wantErrors:   []string{"spec.ephemeralContainers[0].imagePullPolicy"},
		},
		{
			name:         "new image with IfNotPresent",
			updated:      []api.EphemeralContainer{ephemeralContainer("debugger", "debug", api.PullIfNotPresent)},
			wantPolicies: []api.PullPolicy{api.PullAlways},
			wantErrors:   []string{"spec.ephemeralContainers[0].imagePullPolicy"},
		},
		{
			name:         "image reused from regular container",
			updated:      []api.EphemeralContainer{ephemeralContainer("debugger", "image", api.PullNever)},
			wantPolicies: []api.PullPolicy{api.PullAlways},
			wantErrors:   []string{"spec.ephemeralContainers[0].imagePullPolicy"},
		},
		{
			name:         "image reused from init container",
			updated:      []api.EphemeralContainer{ephemeralContainer("debugger", "init-image", api.PullNever)},
			wantPolicies: []api.PullPolicy{api.PullAlways},
			wantErrors:   []string{"spec.ephemeralContainers[0].imagePullPolicy"},
		},
		{
			name:         "image reused from existing ephemeral container",
			old:          []api.EphemeralContainer{ephemeralContainer("existing", "debug", api.PullNever)},
			updated:      []api.EphemeralContainer{ephemeralContainer("existing", "debug", api.PullNever), ephemeralContainer("debugger", "debug", api.PullNever)},
			wantPolicies: []api.PullPolicy{api.PullNever, api.PullAlways},
			wantErrors:   []string{"spec.ephemeralContainers[1].imagePullPolicy"},
		},
		{
			name:         "preserve existing policy when adding a new image",
			old:          []api.EphemeralContainer{ephemeralContainer("existing", "old-debug", api.PullNever)},
			updated:      []api.EphemeralContainer{ephemeralContainer("existing", "old-debug", api.PullNever), ephemeralContainer("debugger", "debug", api.PullAlways)},
			wantPolicies: []api.PullPolicy{api.PullNever, api.PullAlways},
		},
		{
			name:         "new container before existing container",
			old:          []api.EphemeralContainer{ephemeralContainer("existing", "old-debug", api.PullIfNotPresent)},
			updated:      []api.EphemeralContainer{ephemeralContainer("debugger", "debug", api.PullNever), ephemeralContainer("existing", "old-debug", api.PullIfNotPresent)},
			wantPolicies: []api.PullPolicy{api.PullAlways, api.PullIfNotPresent},
			wantErrors:   []string{"spec.ephemeralContainers[0].imagePullPolicy"},
		},
		{
			name:         "all new containers are checked",
			updated:      []api.EphemeralContainer{ephemeralContainer("first", "image", api.PullNever), ephemeralContainer("second", "debug", api.PullIfNotPresent)},
			wantPolicies: []api.PullPolicy{api.PullAlways, api.PullAlways},
			wantErrors:   []string{"spec.ephemeralContainers[0].imagePullPolicy", "spec.ephemeralContainers[1].imagePullPolicy"},
		},
		{
			name:         "new container already uses Always",
			updated:      []api.EphemeralContainer{ephemeralContainer("debugger", "image", api.PullAlways)},
			wantPolicies: []api.PullPolicy{api.PullAlways},
		},
		{
			name:         "unchanged existing container",
			old:          []api.EphemeralContainer{ephemeralContainer("existing", "debug", api.PullNever)},
			updated:      []api.EphemeralContainer{ephemeralContainer("existing", "debug", api.PullNever)},
			wantPolicies: []api.PullPolicy{api.PullNever},
		},
		{
			name:         "only existing containers reordered",
			old:          []api.EphemeralContainer{ephemeralContainer("first", "debug", api.PullNever), ephemeralContainer("second", "other-debug", api.PullIfNotPresent)},
			updated:      []api.EphemeralContainer{ephemeralContainer("second", "other-debug", api.PullIfNotPresent), ephemeralContainer("first", "debug", api.PullNever)},
			wantPolicies: []api.PullPolicy{api.PullIfNotPresent, api.PullNever},
		},
		{
			name:         "dry run reuses existing image",
			updated:      []api.EphemeralContainer{ephemeralContainer("debugger", "image", api.PullNever)},
			wantPolicies: []api.PullPolicy{api.PullAlways},
			wantErrors:   []string{"spec.ephemeralContainers[0].imagePullPolicy"},
			dryRun:       true,
		},
	}
}

func ephemeralContainerPods(tc ephemeralContainerUpdate) (*api.Pod, *api.Pod) {
	gracePeriod := int64(30)
	oldPod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod", Namespace: "test", ResourceVersion: "1", Generation: 1},
		Spec: api.PodSpec{
			NodeName: "node", RestartPolicy: api.RestartPolicyAlways, DNSPolicy: api.DNSClusterFirst,
			TerminationGracePeriodSeconds: &gracePeriod,
			Containers: []api.Container{{Name: "app", Image: "image", ImagePullPolicy: api.PullAlways,
				TerminationMessagePath: "/dev/termination-log", TerminationMessagePolicy: api.TerminationMessageReadFile}},
			InitContainers: []api.Container{{Name: "init", Image: "init-image", ImagePullPolicy: api.PullAlways,
				TerminationMessagePath: "/dev/termination-log", TerminationMessagePolicy: api.TerminationMessageReadFile}},
			EphemeralContainers: tc.old,
		},
	}
	pod := oldPod.DeepCopy()
	pod.Spec.EphemeralContainers = tc.updated
	return pod, oldPod
}

func TestAdmitEphemeralContainers(t *testing.T) {
	for _, tc := range ephemeralContainerUpdates() {
		t.Run(tc.name, func(t *testing.T) {
			pod, oldPod := ephemeralContainerPods(tc)
			originalOldPod := oldPod.DeepCopy()
			expected := pod.DeepCopy()
			for i, policy := range tc.wantPolicies {
				expected.Spec.EphemeralContainers[i].ImagePullPolicy = policy
			}
			handler := admissiontesting.WithReinvocationTesting(t, NewAlwaysPullImages())
			if err := handler.Admit(context.Background(), ephemeralContainersAttributes(pod, oldPod, tc.dryRun), nil); err != nil {
				t.Fatal(err)
			}
			if !equality.Semantic.DeepEqual(pod, expected) {
				t.Errorf("unexpected pod mutation: got %+v, want %+v", pod.Spec.EphemeralContainers, expected.Spec.EphemeralContainers)
			}
			if !equality.Semantic.DeepEqual(oldPod, originalOldPod) {
				t.Error("old pod was mutated")
			}
			podregistry.EphemeralContainersStrategy.PrepareForUpdate(context.Background(), pod, oldPod)
			if errs := podregistry.EphemeralContainersStrategy.ValidateUpdate(context.Background(), pod, oldPod); len(errs) != 0 {
				t.Fatalf("mutation produced an invalid update: %v", errs)
			}
			if err := NewAlwaysPullImages().Validate(context.Background(), ephemeralContainersAttributes(pod, oldPod, tc.dryRun), nil); err != nil {
				t.Fatalf("mutated update failed validating admission: %v", err)
			}
		})
	}
}

func TestValidateEphemeralContainers(t *testing.T) {
	for _, tc := range ephemeralContainerUpdates() {
		t.Run(tc.name, func(t *testing.T) {
			pod, oldPod := ephemeralContainerPods(tc)
			podregistry.EphemeralContainersStrategy.PrepareForUpdate(context.Background(), pod, oldPod)
			if errs := podregistry.EphemeralContainersStrategy.ValidateUpdate(context.Background(), pod, oldPod); len(errs) != 0 {
				t.Fatalf("invalid fixture: %v", errs)
			}
			original := pod.DeepCopy()
			err := NewAlwaysPullImages().Validate(context.Background(), ephemeralContainersAttributes(pod, oldPod, tc.dryRun), nil)
			if len(tc.wantErrors) == 0 {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil {
				t.Fatal("expected new container's pull policy to be rejected")
			} else {
				for _, path := range tc.wantErrors {
					if !strings.Contains(err.Error(), path) {
						t.Errorf("error %q does not mention %s", err, path)
					}
				}
				if got := strings.Count(err.Error(), ".imagePullPolicy:"); got != len(tc.wantErrors) {
					t.Errorf("got %d pull policy errors, want %d: %v", got, len(tc.wantErrors), err)
				}
			}
			if !equality.Semantic.DeepEqual(pod, original) {
				t.Error("validation mutated the pod")
			}
		})
	}
}
