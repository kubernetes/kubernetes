/*
Copyright 2016 The Kubernetes Authors.

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

package podsecuritypolicy

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/diff"
	kadmission "k8s.io/apiserver/pkg/admission"
	admissiontesting "k8s.io/apiserver/pkg/admission/testing"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/authorization/authorizerfactory"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	kapi "k8s.io/kubernetes/pkg/apis/core"
	k8s_api_v1 "k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/security/apparmor"
	kpsp "k8s.io/kubernetes/pkg/security/podsecuritypolicy"
	"k8s.io/kubernetes/pkg/security/podsecuritypolicy/seccomp"
	psputil "k8s.io/kubernetes/pkg/security/podsecuritypolicy/util"
	utilpointer "k8s.io/utils/pointer"
)

const defaultContainerName = "test-c"

// NewTestAdmission provides an admission plugin with test implementations of internal structs.
func NewTestAdmission(psps []*policy.PodSecurityPolicy, authz authorizer.Authorizer) *Plugin {
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	store := informerFactory.Policy().V1beta1().PodSecurityPolicies().Informer().GetStore()
	for _, psp := range psps {
		store.Add(psp)
	}
	lister := informerFactory.Policy().V1beta1().PodSecurityPolicies().Lister()
	if authz == nil {
		authz = &TestAuthorizer{}
	}
	return &Plugin{
		Handler:         kadmission.NewHandler(kadmission.Create, kadmission.Update),
		strategyFactory: kpsp.NewSimpleStrategyFactory(),
		authz:           authz,
		lister:          lister,
	}
}

// TestAuthorizer is a testing struct for testing that fulfills the authorizer interface.
type TestAuthorizer struct {
	// usernameToNamespaceToAllowedPSPs contains the map of allowed PSPs.
	// if nil, all PSPs are allowed.
	usernameToNamespaceToAllowedPSPs map[string]map[string]map[string]bool
	// allowedAPIGroupName specifies an API Group name that contains PSP resources.
	// In order to be authorized, AttributesRecord must have this group name.
	// When empty, API Group name isn't taken into account.
	// TODO: remove this when PSP will be completely moved out of the extensions and we'll lookup only in "policy" group.
	allowedAPIGroupName string
}

func (t *TestAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorized authorizer.Decision, reason string, err error) {
	if t.usernameToNamespaceToAllowedPSPs == nil {
		return authorizer.DecisionAllow, "", nil
	}
	allowedInNamespace := t.usernameToNamespaceToAllowedPSPs[a.GetUser().GetName()][a.GetNamespace()][a.GetName()]
	allowedClusterWide := t.usernameToNamespaceToAllowedPSPs[a.GetUser().GetName()][""][a.GetName()]
	allowedAPIGroup := len(t.allowedAPIGroupName) == 0 || a.GetAPIGroup() == t.allowedAPIGroupName
	if allowedAPIGroup && (allowedInNamespace || allowedClusterWide) {
		return authorizer.DecisionAllow, "", nil
	}
	return authorizer.DecisionNoOpinion, "", nil
}

var _ authorizer.Authorizer = &TestAuthorizer{}

func useInitContainers(pod *kapi.Pod) *kapi.Pod {
	pod.Spec.InitContainers = pod.Spec.Containers
	pod.Spec.Containers = []kapi.Container{}
	return pod
}

func TestAdmitSeccomp(t *testing.T) {
	containerName := "container"
	tests := map[string]struct {
		pspAnnotations     map[string]string
		podAnnotations     map[string]string
		shouldPassAdmit    bool
		shouldPassValidate bool
	}{
		"no seccomp, no pod annotations": {
			pspAnnotations:     nil,
			podAnnotations:     nil,
			shouldPassAdmit:    true,
			shouldPassValidate: true,
		},
		"no seccomp, pod annotations": {
			pspAnnotations: nil,
			podAnnotations: map[string]string{
				kapi.SeccompPodAnnotationKey: "foo",
			},
			shouldPassAdmit:    false,
			shouldPassValidate: false,
		},
		"no seccomp, container annotations": {
			pspAnnotations: nil,
			podAnnotations: map[string]string{
				kapi.SeccompContainerAnnotationKeyPrefix + containerName: "foo",
			},
			shouldPassAdmit:    false,
			shouldPassValidate: false,
		},
		"seccomp, allow any no pod annotation": {
			pspAnnotations: map[string]string{
				seccomp.AllowedProfilesAnnotationKey: seccomp.AllowAny,
			},
			podAnnotations:     nil,
			shouldPassAdmit:    true,
			shouldPassValidate: true,
		},
		"seccomp, allow any pod annotation": {
			pspAnnotations: map[string]string{
				seccomp.AllowedProfilesAnnotationKey: seccomp.AllowAny,
			},
			podAnnotations: map[string]string{
				kapi.SeccompPodAnnotationKey: "foo",
			},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
		},
		"seccomp, allow any container annotation": {
			pspAnnotations: map[string]string{
				seccomp.AllowedProfilesAnnotationKey: seccomp.AllowAny,
			},
			podAnnotations: map[string]string{
				kapi.SeccompContainerAnnotationKeyPrefix + containerName: "foo",
			},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
		},
		"seccomp, allow specific pod annotation failure": {
			pspAnnotations: map[string]string{
				seccomp.AllowedProfilesAnnotationKey: "foo",
			},
			podAnnotations: map[string]string{
				kapi.SeccompPodAnnotationKey: "bar",
			},
			shouldPassAdmit:    false,
			shouldPassValidate: false,
		},
		"seccomp, allow specific container annotation failure": {
			pspAnnotations: map[string]string{
				// provide a default so we don't have to give the pod annotation
				seccomp.DefaultProfileAnnotationKey:  "foo",
				seccomp.AllowedProfilesAnnotationKey: "foo",
			},
			podAnnotations: map[string]string{
				kapi.SeccompContainerAnnotationKeyPrefix + containerName: "bar",
			},
			shouldPassAdmit:    false,
			shouldPassValidate: false,
		},
		"seccomp, allow specific pod annotation pass": {
			pspAnnotations: map[string]string{
				seccomp.AllowedProfilesAnnotationKey: "foo",
			},
			podAnnotations: map[string]string{
				kapi.SeccompPodAnnotationKey: "foo",
			},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
		},
		"seccomp, allow specific container annotation pass": {
			pspAnnotations: map[string]string{
				// provide a default so we don't have to give the pod annotation
				seccomp.DefaultProfileAnnotationKey:  "foo",
				seccomp.AllowedProfilesAnnotationKey: "foo,bar",
			},
			podAnnotations: map[string]string{
				kapi.SeccompContainerAnnotationKeyPrefix + containerName: "bar",
			},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
		},
	}
	for k, v := range tests {
		psp := restrictivePSP()
		psp.Annotations = v.pspAnnotations
		pod := &kapi.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: v.podAnnotations,
			},
			Spec: kapi.PodSpec{
				Containers: []kapi.Container{
					{Name: containerName},
				},
			},
		}
		testPSPAdmit(k, []*policy.PodSecurityPolicy{psp}, pod, v.shouldPassAdmit, v.shouldPassValidate, psp.Name, t)
	}
}

func TestAdmitPrivileged(t *testing.T) {
	createPodWithPriv := func(priv bool) *kapi.Pod {
		pod := goodPod()
		pod.Spec.Containers[0].SecurityContext.Privileged = &priv
		return pod
	}

	nonPrivilegedPSP := restrictivePSP()
	nonPrivilegedPSP.Name = "non-priv"
	nonPrivilegedPSP.Spec.Privileged = false

	privilegedPSP := restrictivePSP()
	privilegedPSP.Name = "priv"
	privilegedPSP.Spec.Privileged = true

	trueValue := true
	falseValue := false

	tests := map[string]struct {
		pod                *kapi.Pod
		psps               []*policy.PodSecurityPolicy
		shouldPassAdmit    bool
		shouldPassValidate bool
		expectedPriv       *bool
		expectedPSP        string
	}{
		"pod with priv=nil allowed under non priv PSP": {
			pod:                goodPod(),
			psps:               []*policy.PodSecurityPolicy{nonPrivilegedPSP},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedPriv:       nil,
			expectedPSP:        nonPrivilegedPSP.Name,
		},
		"pod with priv=nil allowed under priv PSP": {
			pod:                goodPod(),
			psps:               []*policy.PodSecurityPolicy{privilegedPSP},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedPriv:       nil,
			expectedPSP:        privilegedPSP.Name,
		},
		"pod with priv=false allowed under non priv PSP": {
			pod:                createPodWithPriv(false),
			psps:               []*policy.PodSecurityPolicy{nonPrivilegedPSP},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedPriv:       &falseValue,
			expectedPSP:        nonPrivilegedPSP.Name,
		},
		"pod with priv=false allowed under priv PSP": {
			pod:                createPodWithPriv(false),
			psps:               []*policy.PodSecurityPolicy{privilegedPSP},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedPriv:       &falseValue,
			expectedPSP:        privilegedPSP.Name,
		},
		"pod with priv=true denied by non priv PSP": {
			pod:                createPodWithPriv(true),
			psps:               []*policy.PodSecurityPolicy{nonPrivilegedPSP},
			shouldPassAdmit:    false,
			shouldPassValidate: false,
		},
		"pod with priv=true allowed by priv PSP": {
			pod:                createPodWithPriv(true),
			psps:               []*policy.PodSecurityPolicy{nonPrivilegedPSP, privilegedPSP},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedPriv:       &trueValue,
			expectedPSP:        privilegedPSP.Name,
		},
	}

	for k, v := range tests {
		testPSPAdmit(k, v.psps, v.pod, v.shouldPassAdmit, v.shouldPassValidate, v.expectedPSP, t)

		if v.shouldPassAdmit {
			priv := v.pod.Spec.Containers[0].SecurityContext.Privileged
			if (priv == nil) != (v.expectedPriv == nil) {
				t.Errorf("%s expected privileged to be %v, got %v", k, v.expectedPriv, priv)
			} else if priv != nil && *priv != *v.expectedPriv {
				t.Errorf("%s expected privileged to be %v, got %v", k, *v.expectedPriv, *priv)
			}
		}
	}
}

func defaultPod(t *testing.T, pod *kapi.Pod) *kapi.Pod {
	v1Pod := &v1.Pod{}
	if err := legacyscheme.Scheme.Convert(pod, v1Pod, nil); err != nil {
		t.Fatal(err)
	}
	legacyscheme.Scheme.Default(v1Pod)
	apiPod := &kapi.Pod{}
	if err := legacyscheme.Scheme.Convert(v1Pod, apiPod, nil); err != nil {
		t.Fatal(err)
	}
	return apiPod
}

func TestAdmitPreferNonmutating(t *testing.T) {
	mutating1 := restrictivePSP()
	mutating1.Name = "mutating1"
	mutating1.Spec.RunAsUser.Ranges = []policy.IDRange{{Min: int64(1), Max: int64(1)}}

	mutating2 := restrictivePSP()
	mutating2.Name = "mutating2"
	mutating2.Spec.RunAsUser.Ranges = []policy.IDRange{{Min: int64(2), Max: int64(2)}}

	privilegedPSP := permissivePSP()
	privilegedPSP.Name = "privileged"

	unprivilegedRunAsAnyPod := defaultPod(t, &kapi.Pod{
		ObjectMeta: metav1.ObjectMeta{},
		Spec: kapi.PodSpec{
			ServiceAccountName: "default",
			Containers:         []kapi.Container{{Name: "mycontainer", Image: "myimage"}},
		},
	})
	changedPod := unprivilegedRunAsAnyPod.DeepCopy()
	changedPod.Spec.Containers[0].Image = "myimage2"

	podWithSC := unprivilegedRunAsAnyPod.DeepCopy()
	podWithSC.Annotations = map[string]string{psputil.ValidatedPSPAnnotation: privilegedPSP.Name}
	changedPodWithSC := changedPod.DeepCopy()
	changedPodWithSC.Annotations = map[string]string{psputil.ValidatedPSPAnnotation: privilegedPSP.Name}

	gcChangedPod := unprivilegedRunAsAnyPod.DeepCopy()
	gcChangedPod.OwnerReferences = []metav1.OwnerReference{{Kind: "Foo", Name: "bar"}}
	gcChangedPod.Finalizers = []string{"foo"}

	podWithAnnotation := unprivilegedRunAsAnyPod.DeepCopy()
	podWithAnnotation.ObjectMeta.Annotations = map[string]string{
		// "mutating2" is lexicographically behind "mutating1", so "mutating1" should be
		// chosen because it's the canonical PSP order.
		psputil.ValidatedPSPAnnotation: mutating2.Name,
	}

	tests := map[string]struct {
		operation             kadmission.Operation
		pod                   *kapi.Pod
		podBeforeUpdate       *kapi.Pod
		psps                  []*policy.PodSecurityPolicy
		shouldPassValidate    bool
		expectMutation        bool
		expectedContainerUser *int64
		expectedPSP           string
	}{
		"pod should not be mutated by allow-all strategies": {
			operation:             kadmission.Create,
			pod:                   unprivilegedRunAsAnyPod.DeepCopy(),
			psps:                  []*policy.PodSecurityPolicy{privilegedPSP},
			shouldPassValidate:    true,
			expectMutation:        false,
			expectedContainerUser: nil,
			expectedPSP:           privilegedPSP.Name,
		},
		"pod should prefer non-mutating PSP on create": {
			operation:             kadmission.Create,
			pod:                   unprivilegedRunAsAnyPod.DeepCopy(),
			psps:                  []*policy.PodSecurityPolicy{mutating2, mutating1, privilegedPSP},
			shouldPassValidate:    true,
			expectMutation:        false,
			expectedContainerUser: nil,
			expectedPSP:           privilegedPSP.Name,
		},
		"pod should use deterministic mutating PSP on create": {
			operation:             kadmission.Create,
			pod:                   unprivilegedRunAsAnyPod.DeepCopy(),
			psps:                  []*policy.PodSecurityPolicy{mutating2, mutating1},
			shouldPassValidate:    true,
			expectMutation:        true,
			expectedContainerUser: &mutating1.Spec.RunAsUser.Ranges[0].Min,
			expectedPSP:           mutating1.Name,
		},
		"pod should use deterministic mutating PSP on create even if ValidatedPSPAnnotation is set": {
			operation:             kadmission.Create,
			pod:                   podWithAnnotation,
			psps:                  []*policy.PodSecurityPolicy{mutating2, mutating1},
			shouldPassValidate:    true,
			expectMutation:        true,
			expectedContainerUser: &mutating1.Spec.RunAsUser.Ranges[0].Min,
			expectedPSP:           mutating1.Name,
		},
		"pod should prefer non-mutating PSP on update": {
			operation:             kadmission.Update,
			pod:                   changedPodWithSC.DeepCopy(),
			podBeforeUpdate:       podWithSC.DeepCopy(),
			psps:                  []*policy.PodSecurityPolicy{mutating2, mutating1, privilegedPSP},
			shouldPassValidate:    true,
			expectMutation:        false,
			expectedContainerUser: nil,
			expectedPSP:           privilegedPSP.Name,
		},
		"pod should not mutate on update, but fail validation": {
			operation:             kadmission.Update,
			pod:                   changedPod.DeepCopy(),
			podBeforeUpdate:       unprivilegedRunAsAnyPod.DeepCopy(),
			psps:                  []*policy.PodSecurityPolicy{mutating2, mutating1},
			shouldPassValidate:    false,
			expectMutation:        false,
			expectedContainerUser: nil,
			expectedPSP:           "",
		},
		"pod should be allowed if completely unchanged on update": {
			operation:             kadmission.Update,
			pod:                   unprivilegedRunAsAnyPod.DeepCopy(),
			podBeforeUpdate:       unprivilegedRunAsAnyPod.DeepCopy(),
			psps:                  []*policy.PodSecurityPolicy{mutating2, mutating1},
			shouldPassValidate:    true,
			expectMutation:        false,
			expectedContainerUser: nil,
			expectedPSP:           "",
		},
		"pod should be allowed if unchanged on update except finalizers,ownerrefs": {
			operation:             kadmission.Update,
			pod:                   gcChangedPod.DeepCopy(),
			podBeforeUpdate:       unprivilegedRunAsAnyPod.DeepCopy(),
			psps:                  []*policy.PodSecurityPolicy{mutating2, mutating1},
			shouldPassValidate:    true,
			expectMutation:        false,
			expectedContainerUser: nil,
			expectedPSP:           "",
		},
	}

	for k, v := range tests {
		testPSPAdmitAdvanced(k, v.operation, v.psps, nil, &user.DefaultInfo{}, v.pod, v.podBeforeUpdate, true, v.shouldPassValidate, v.expectMutation, v.expectedPSP, t)

		actualPodUser := (*int64)(nil)
		if v.pod.Spec.SecurityContext != nil {
			actualPodUser = v.pod.Spec.SecurityContext.RunAsUser
		}
		if actualPodUser != nil {
			t.Errorf("%s expected pod user nil, got %v", k, *actualPodUser)
		}

		actualContainerUser := (*int64)(nil)
		if v.pod.Spec.Containers[0].SecurityContext != nil {
			actualContainerUser = v.pod.Spec.Containers[0].SecurityContext.RunAsUser
		}
		if (actualContainerUser == nil) != (v.expectedContainerUser == nil) {
			t.Errorf("%s expected container user %v, got %v", k, v.expectedContainerUser, actualContainerUser)
		} else if actualContainerUser != nil && *actualContainerUser != *v.expectedContainerUser {
			t.Errorf("%s expected container user %v, got %v", k, *v.expectedContainerUser, *actualContainerUser)
		}
	}
}

func TestFailClosedOnInvalidPod(t *testing.T) {
	plugin := NewTestAdmission(nil, nil)
	pod := &v1.Pod{}
	attrs := kadmission.NewAttributesRecord(pod, nil, kapi.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, kapi.Resource("pods").WithVersion("version"), "", kadmission.Create, &metav1.CreateOptions{}, false, &user.DefaultInfo{})

	err := plugin.Admit(context.TODO(), attrs, nil)
	if err == nil {
		t.Fatalf("expected versioned pod object to fail mutating admission")
	}
	if !strings.Contains(err.Error(), "unexpected type") {
		t.Errorf("expected type error on Admit but got: %v", err)
	}

	err = plugin.Validate(context.TODO(), attrs, nil)
	if err == nil {
		t.Fatalf("expected versioned pod object to fail validating admission")
	}
	if !strings.Contains(err.Error(), "unexpected type") {
		t.Errorf("expected type error on Validate but got: %v", err)
	}
}

func TestAdmitCaps(t *testing.T) {
	createPodWithCaps := func(caps *kapi.Capabilities) *kapi.Pod {
		pod := goodPod()
		pod.Spec.Containers[0].SecurityContext.Capabilities = caps
		return pod
	}

	restricted := restrictivePSP()

	allowsFooInAllowed := restrictivePSP()
	allowsFooInAllowed.Name = "allowCapInAllowed"
	allowsFooInAllowed.Spec.AllowedCapabilities = []v1.Capability{"foo"}

	allowsFooInRequired := restrictivePSP()
	allowsFooInRequired.Name = "allowCapInRequired"
	allowsFooInRequired.Spec.DefaultAddCapabilities = []v1.Capability{"foo"}

	requiresFooToBeDropped := restrictivePSP()
	requiresFooToBeDropped.Name = "requireDrop"
	requiresFooToBeDropped.Spec.RequiredDropCapabilities = []v1.Capability{"foo"}

	allowAllInAllowed := restrictivePSP()
	allowAllInAllowed.Name = "allowAllCapsInAllowed"
	allowAllInAllowed.Spec.AllowedCapabilities = []v1.Capability{policy.AllowAllCapabilities}

	tc := map[string]struct {
		pod                  *kapi.Pod
		psps                 []*policy.PodSecurityPolicy
		shouldPassAdmit      bool
		shouldPassValidate   bool
		expectedCapabilities *kapi.Capabilities
		expectedPSP          string
	}{
		// UC 1: if a PSP does not define allowed or required caps then a pod requesting a cap
		// should be rejected.
		"should reject cap add when not allowed or required": {
			pod:                createPodWithCaps(&kapi.Capabilities{Add: []kapi.Capability{"foo"}}),
			psps:               []*policy.PodSecurityPolicy{restricted},
			shouldPassAdmit:    false,
			shouldPassValidate: false,
		},
		// UC 2: if a PSP allows a cap in the allowed field it should accept the pod request
		// to add the cap.
		"should accept cap add when in allowed": {
			pod:                createPodWithCaps(&kapi.Capabilities{Add: []kapi.Capability{"foo"}}),
			psps:               []*policy.PodSecurityPolicy{restricted, allowsFooInAllowed},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedPSP:        allowsFooInAllowed.Name,
		},
		// UC 3: if a PSP requires a cap then it should accept the pod request
		// to add the cap.
		"should accept cap add when in required": {
			pod:                createPodWithCaps(&kapi.Capabilities{Add: []kapi.Capability{"foo"}}),
			psps:               []*policy.PodSecurityPolicy{restricted, allowsFooInRequired},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedPSP:        allowsFooInRequired.Name,
		},
		// UC 4: if a PSP requires a cap to be dropped then it should fail both
		// in the verification of adds and verification of drops
		"should reject cap add when requested cap is required to be dropped": {
			pod:                createPodWithCaps(&kapi.Capabilities{Add: []kapi.Capability{"foo"}}),
			psps:               []*policy.PodSecurityPolicy{restricted, requiresFooToBeDropped},
			shouldPassAdmit:    false,
			shouldPassValidate: false,
		},
		// UC 5: if a PSP requires a cap to be dropped it should accept
		// a manual request to drop the cap.
		"should accept cap drop when cap is required to be dropped": {
			pod:                createPodWithCaps(&kapi.Capabilities{Drop: []kapi.Capability{"foo"}}),
			psps:               []*policy.PodSecurityPolicy{requiresFooToBeDropped},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedPSP:        requiresFooToBeDropped.Name,
		},
		// UC 6: required add is defaulted
		"required add is defaulted": {
			pod:                goodPod(),
			psps:               []*policy.PodSecurityPolicy{allowsFooInRequired},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedCapabilities: &kapi.Capabilities{
				Add: []kapi.Capability{"foo"},
			},
			expectedPSP: allowsFooInRequired.Name,
		},
		// UC 7: required drop is defaulted
		"required drop is defaulted": {
			pod:                goodPod(),
			psps:               []*policy.PodSecurityPolicy{requiresFooToBeDropped},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedCapabilities: &kapi.Capabilities{
				Drop: []kapi.Capability{"foo"},
			},
			expectedPSP: requiresFooToBeDropped.Name,
		},
		// UC 8: using '*' in allowed caps
		"should accept cap add when all caps are allowed": {
			pod:                createPodWithCaps(&kapi.Capabilities{Add: []kapi.Capability{"foo"}}),
			psps:               []*policy.PodSecurityPolicy{restricted, allowAllInAllowed},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedPSP:        allowAllInAllowed.Name,
		},
	}

	for k, v := range tc {
		testPSPAdmit(k, v.psps, v.pod, v.shouldPassAdmit, v.shouldPassValidate, v.expectedPSP, t)

		if v.expectedCapabilities != nil {
			if !reflect.DeepEqual(v.expectedCapabilities, v.pod.Spec.Containers[0].SecurityContext.Capabilities) {
				t.Errorf("%s resulted in caps that were not expected - expected: %v, received: %v", k, v.expectedCapabilities, v.pod.Spec.Containers[0].SecurityContext.Capabilities)
			}
		}
	}

	for k, v := range tc {
		useInitContainers(v.pod)
		testPSPAdmit(k, v.psps, v.pod, v.shouldPassAdmit, v.shouldPassValidate, v.expectedPSP, t)

		if v.expectedCapabilities != nil {
			if !reflect.DeepEqual(v.expectedCapabilities, v.pod.Spec.InitContainers[0].SecurityContext.Capabilities) {
				t.Errorf("%s resulted in caps that were not expected - expected: %v, received: %v", k, v.expectedCapabilities, v.pod.Spec.InitContainers[0].SecurityContext.Capabilities)
			}
		}
	}

}

func TestAdmitVolumes(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIInlineVolume, true)()

	val := reflect.ValueOf(kapi.VolumeSource{})

	for i := 0; i < val.NumField(); i++ {
		// reflectively create the volume source
		fieldVal := val.Type().Field(i)

		volumeSource := kapi.VolumeSource{}
		volumeSourceVolume := reflect.New(fieldVal.Type.Elem())

		reflect.ValueOf(&volumeSource).Elem().FieldByName(fieldVal.Name).Set(volumeSourceVolume)
		volume := kapi.Volume{VolumeSource: volumeSource}

		// sanity check before moving on
		fsType, err := psputil.GetVolumeFSType(volume)
		if err != nil {
			t.Errorf("error getting FSType for %s: %s", fieldVal.Name, err.Error())
			continue
		}

		// add the volume to the pod
		pod := goodPod()
		pod.Spec.Volumes = []kapi.Volume{volume}

		// create a PSP that allows no volumes
		psp := restrictivePSP()

		// expect a denial for this PSP
		testPSPAdmit(fmt.Sprintf("%s denial", string(fsType)), []*policy.PodSecurityPolicy{psp}, pod, false, false, "", t)

		// also expect a denial for this PSP if it's an init container
		useInitContainers(pod)
		testPSPAdmit(fmt.Sprintf("%s denial", string(fsType)), []*policy.PodSecurityPolicy{psp}, pod, false, false, "", t)

		// now add the fstype directly to the psp and it should validate
		psp.Spec.Volumes = []policy.FSType{fsType}
		testPSPAdmit(fmt.Sprintf("%s direct accept", string(fsType)), []*policy.PodSecurityPolicy{psp}, pod, true, true, psp.Name, t)

		// now change the psp to allow any volumes and the pod should still validate
		psp.Spec.Volumes = []policy.FSType{policy.All}
		testPSPAdmit(fmt.Sprintf("%s wildcard accept", string(fsType)), []*policy.PodSecurityPolicy{psp}, pod, true, true, psp.Name, t)
	}
}

func TestAdmitHostNetwork(t *testing.T) {
	createPodWithHostNetwork := func(hostNetwork bool) *kapi.Pod {
		pod := goodPod()
		pod.Spec.SecurityContext.HostNetwork = hostNetwork
		return pod
	}

	noHostNetwork := restrictivePSP()
	noHostNetwork.Name = "no-hostnetwork"
	noHostNetwork.Spec.HostNetwork = false

	hostNetwork := restrictivePSP()
	hostNetwork.Name = "hostnetwork"
	hostNetwork.Spec.HostNetwork = true

	tests := map[string]struct {
		pod                 *kapi.Pod
		psps                []*policy.PodSecurityPolicy
		shouldPassAdmit     bool
		shouldPassValidate  bool
		expectedHostNetwork bool
		expectedPSP         string
	}{
		"pod without hostnetwork request allowed under noHostNetwork PSP": {
			pod:                 goodPod(),
			psps:                []*policy.PodSecurityPolicy{noHostNetwork},
			shouldPassAdmit:     true,
			shouldPassValidate:  true,
			expectedHostNetwork: false,
			expectedPSP:         noHostNetwork.Name,
		},
		"pod without hostnetwork request allowed under hostNetwork PSP": {
			pod:                 goodPod(),
			psps:                []*policy.PodSecurityPolicy{hostNetwork},
			shouldPassAdmit:     true,
			shouldPassValidate:  true,
			expectedHostNetwork: false,
			expectedPSP:         hostNetwork.Name,
		},
		"pod with hostnetwork request denied by noHostNetwork PSP": {
			pod:                createPodWithHostNetwork(true),
			psps:               []*policy.PodSecurityPolicy{noHostNetwork},
			shouldPassAdmit:    false,
			shouldPassValidate: false,
		},
		"pod with hostnetwork request allowed by hostNetwork PSP": {
			pod:                 createPodWithHostNetwork(true),
			psps:                []*policy.PodSecurityPolicy{noHostNetwork, hostNetwork},
			shouldPassAdmit:     true,
			shouldPassValidate:  true,
			expectedHostNetwork: true,
			expectedPSP:         hostNetwork.Name,
		},
	}

	for k, v := range tests {
		testPSPAdmit(k, v.psps, v.pod, v.shouldPassAdmit, v.shouldPassValidate, v.expectedPSP, t)

		if v.shouldPassAdmit {
			if v.pod.Spec.SecurityContext.HostNetwork != v.expectedHostNetwork {
				t.Errorf("%s expected hostNetwork to be %t", k, v.expectedHostNetwork)
			}
		}
	}

	// test again with init containers
	for k, v := range tests {
		useInitContainers(v.pod)
		testPSPAdmit(k, v.psps, v.pod, v.shouldPassAdmit, v.shouldPassValidate, v.expectedPSP, t)

		if v.shouldPassAdmit {
			if v.pod.Spec.SecurityContext.HostNetwork != v.expectedHostNetwork {
				t.Errorf("%s expected hostNetwork to be %t", k, v.expectedHostNetwork)
			}
		}
	}
}

func TestAdmitHostPorts(t *testing.T) {
	createPodWithHostPorts := func(port int32) *kapi.Pod {
		pod := goodPod()
		pod.Spec.Containers[0].Ports = []kapi.ContainerPort{
			{HostPort: port},
		}
		return pod
	}

	noHostPorts := restrictivePSP()
	noHostPorts.Name = "noHostPorts"

	hostPorts := restrictivePSP()
	hostPorts.Name = "hostPorts"
	hostPorts.Spec.HostPorts = []policy.HostPortRange{
		{Min: 1, Max: 10},
	}

	tests := map[string]struct {
		pod                *kapi.Pod
		psps               []*policy.PodSecurityPolicy
		shouldPassAdmit    bool
		shouldPassValidate bool
		expectedPSP        string
	}{
		"host port out of range": {
			pod:                createPodWithHostPorts(11),
			psps:               []*policy.PodSecurityPolicy{hostPorts},
			shouldPassAdmit:    false,
			shouldPassValidate: false,
		},
		"host port in range": {
			pod:                createPodWithHostPorts(5),
			psps:               []*policy.PodSecurityPolicy{hostPorts},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedPSP:        hostPorts.Name,
		},
		"no host ports with range": {
			pod:                goodPod(),
			psps:               []*policy.PodSecurityPolicy{hostPorts},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedPSP:        hostPorts.Name,
		},
		"no host ports without range": {
			pod:                goodPod(),
			psps:               []*policy.PodSecurityPolicy{noHostPorts},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedPSP:        noHostPorts.Name,
		},
		"host ports without range": {
			pod:                createPodWithHostPorts(5),
			psps:               []*policy.PodSecurityPolicy{noHostPorts},
			shouldPassAdmit:    false,
			shouldPassValidate: false,
		},
	}

	for i := 0; i < 2; i++ {
		for k, v := range tests {
			v.pod.Spec.Containers, v.pod.Spec.InitContainers = v.pod.Spec.InitContainers, v.pod.Spec.Containers
			testPSPAdmit(k, v.psps, v.pod, v.shouldPassAdmit, v.shouldPassValidate, v.expectedPSP, t)
		}
	}
}

func TestAdmitHostPID(t *testing.T) {
	createPodWithHostPID := func(hostPID bool) *kapi.Pod {
		pod := goodPod()
		pod.Spec.SecurityContext.HostPID = hostPID
		return pod
	}

	noHostPID := restrictivePSP()
	noHostPID.Name = "no-hostpid"
	noHostPID.Spec.HostPID = false

	hostPID := restrictivePSP()
	hostPID.Name = "hostpid"
	hostPID.Spec.HostPID = true

	tests := map[string]struct {
		pod                *kapi.Pod
		psps               []*policy.PodSecurityPolicy
		shouldPassAdmit    bool
		shouldPassValidate bool
		expectedHostPID    bool
		expectedPSP        string
	}{
		"pod without hostpid request allowed under noHostPID PSP": {
			pod:                goodPod(),
			psps:               []*policy.PodSecurityPolicy{noHostPID},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedHostPID:    false,
			expectedPSP:        noHostPID.Name,
		},
		"pod without hostpid request allowed under hostPID PSP": {
			pod:                goodPod(),
			psps:               []*policy.PodSecurityPolicy{hostPID},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedHostPID:    false,
			expectedPSP:        hostPID.Name,
		},
		"pod with hostpid request denied by noHostPID PSP": {
			pod:             createPodWithHostPID(true),
			psps:            []*policy.PodSecurityPolicy{noHostPID},
			shouldPassAdmit: false,
		},
		"pod with hostpid request allowed by hostPID PSP": {
			pod:                createPodWithHostPID(true),
			psps:               []*policy.PodSecurityPolicy{noHostPID, hostPID},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedHostPID:    true,
			expectedPSP:        hostPID.Name,
		},
	}

	for k, v := range tests {
		testPSPAdmit(k, v.psps, v.pod, v.shouldPassAdmit, v.shouldPassValidate, v.expectedPSP, t)

		if v.shouldPassAdmit {
			if v.pod.Spec.SecurityContext.HostPID != v.expectedHostPID {
				t.Errorf("%s expected hostPID to be %t", k, v.expectedHostPID)
			}
		}
	}
}

func TestAdmitHostIPC(t *testing.T) {
	createPodWithHostIPC := func(hostIPC bool) *kapi.Pod {
		pod := goodPod()
		pod.Spec.SecurityContext.HostIPC = hostIPC
		return pod
	}

	noHostIPC := restrictivePSP()
	noHostIPC.Name = "no-hostIPC"
	noHostIPC.Spec.HostIPC = false

	hostIPC := restrictivePSP()
	hostIPC.Name = "hostIPC"
	hostIPC.Spec.HostIPC = true

	tests := map[string]struct {
		pod                *kapi.Pod
		psps               []*policy.PodSecurityPolicy
		shouldPassAdmit    bool
		shouldPassValidate bool
		expectedHostIPC    bool
		expectedPSP        string
	}{
		"pod without hostIPC request allowed under noHostIPC PSP": {
			pod:                goodPod(),
			psps:               []*policy.PodSecurityPolicy{noHostIPC},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedHostIPC:    false,
			expectedPSP:        noHostIPC.Name,
		},
		"pod without hostIPC request allowed under hostIPC PSP": {
			pod:                goodPod(),
			psps:               []*policy.PodSecurityPolicy{hostIPC},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedHostIPC:    false,
			expectedPSP:        hostIPC.Name,
		},
		"pod with hostIPC request denied by noHostIPC PSP": {
			pod:                createPodWithHostIPC(true),
			psps:               []*policy.PodSecurityPolicy{noHostIPC},
			shouldPassAdmit:    false,
			shouldPassValidate: false,
		},
		"pod with hostIPC request allowed by hostIPC PSP": {
			pod:                createPodWithHostIPC(true),
			psps:               []*policy.PodSecurityPolicy{noHostIPC, hostIPC},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedHostIPC:    true,
			expectedPSP:        hostIPC.Name,
		},
	}

	for k, v := range tests {
		testPSPAdmit(k, v.psps, v.pod, v.shouldPassAdmit, v.shouldPassValidate, v.expectedPSP, t)

		if v.shouldPassAdmit {
			if v.pod.Spec.SecurityContext.HostIPC != v.expectedHostIPC {
				t.Errorf("%s expected hostIPC to be %t", k, v.expectedHostIPC)
			}
		}
	}
}

func createPodWithSecurityContexts(podSC *kapi.PodSecurityContext, containerSC *kapi.SecurityContext) *kapi.Pod {
	pod := goodPod()
	pod.Spec.SecurityContext = podSC
	pod.Spec.Containers[0].SecurityContext = containerSC
	return pod
}

func TestAdmitSELinux(t *testing.T) {
	runAsAny := permissivePSP()
	runAsAny.Name = "runAsAny"
	runAsAny.Spec.SELinux.Rule = policy.SELinuxStrategyRunAsAny
	runAsAny.Spec.SELinux.SELinuxOptions = nil

	mustRunAs := permissivePSP()
	mustRunAs.Name = "mustRunAs"
	mustRunAs.Spec.SELinux.Rule = policy.SELinuxStrategyMustRunAs
	mustRunAs.Spec.SELinux.SELinuxOptions = &v1.SELinuxOptions{}
	mustRunAs.Spec.SELinux.SELinuxOptions.Level = "level"
	mustRunAs.Spec.SELinux.SELinuxOptions.Role = "role"
	mustRunAs.Spec.SELinux.SELinuxOptions.Type = "type"
	mustRunAs.Spec.SELinux.SELinuxOptions.User = "user"

	getInternalSEOptions := func(policy *policy.PodSecurityPolicy) *kapi.SELinuxOptions {
		opt := kapi.SELinuxOptions{}
		k8s_api_v1.Convert_v1_SELinuxOptions_To_core_SELinuxOptions(policy.Spec.SELinux.SELinuxOptions, &opt, nil)
		return &opt
	}

	tests := map[string]struct {
		pod                 *kapi.Pod
		psps                []*policy.PodSecurityPolicy
		shouldPassAdmit     bool
		shouldPassValidate  bool
		expectedPodSC       *kapi.PodSecurityContext
		expectedContainerSC *kapi.SecurityContext
		expectedPSP         string
	}{
		"runAsAny with no request": {
			pod:                 createPodWithSecurityContexts(nil, nil),
			psps:                []*policy.PodSecurityPolicy{runAsAny},
			shouldPassAdmit:     true,
			shouldPassValidate:  true,
			expectedPodSC:       nil,
			expectedContainerSC: nil,
			expectedPSP:         runAsAny.Name,
		},
		"runAsAny with empty pod request": {
			pod:                 createPodWithSecurityContexts(&kapi.PodSecurityContext{}, nil),
			psps:                []*policy.PodSecurityPolicy{runAsAny},
			shouldPassAdmit:     true,
			shouldPassValidate:  true,
			expectedPodSC:       &kapi.PodSecurityContext{},
			expectedContainerSC: nil,
			expectedPSP:         runAsAny.Name,
		},
		"runAsAny with empty container request": {
			pod:                 createPodWithSecurityContexts(nil, &kapi.SecurityContext{}),
			psps:                []*policy.PodSecurityPolicy{runAsAny},
			shouldPassAdmit:     true,
			shouldPassValidate:  true,
			expectedPodSC:       nil,
			expectedContainerSC: &kapi.SecurityContext{},
			expectedPSP:         runAsAny.Name,
		},

		"runAsAny with pod request": {
			pod:                 createPodWithSecurityContexts(&kapi.PodSecurityContext{SELinuxOptions: &kapi.SELinuxOptions{User: "foo"}}, nil),
			psps:                []*policy.PodSecurityPolicy{runAsAny},
			shouldPassAdmit:     true,
			shouldPassValidate:  true,
			expectedPodSC:       &kapi.PodSecurityContext{SELinuxOptions: &kapi.SELinuxOptions{User: "foo"}},
			expectedContainerSC: nil,
			expectedPSP:         runAsAny.Name,
		},
		"runAsAny with container request": {
			pod:                 createPodWithSecurityContexts(nil, &kapi.SecurityContext{SELinuxOptions: &kapi.SELinuxOptions{User: "foo"}}),
			psps:                []*policy.PodSecurityPolicy{runAsAny},
			shouldPassAdmit:     true,
			shouldPassValidate:  true,
			expectedPodSC:       nil,
			expectedContainerSC: &kapi.SecurityContext{SELinuxOptions: &kapi.SELinuxOptions{User: "foo"}},
			expectedPSP:         runAsAny.Name,
		},
		"runAsAny with pod and container request": {
			pod:                 createPodWithSecurityContexts(&kapi.PodSecurityContext{SELinuxOptions: &kapi.SELinuxOptions{User: "bar"}}, &kapi.SecurityContext{SELinuxOptions: &kapi.SELinuxOptions{User: "foo"}}),
			psps:                []*policy.PodSecurityPolicy{runAsAny},
			shouldPassAdmit:     true,
			shouldPassValidate:  true,
			expectedPodSC:       &kapi.PodSecurityContext{SELinuxOptions: &kapi.SELinuxOptions{User: "bar"}},
			expectedContainerSC: &kapi.SecurityContext{SELinuxOptions: &kapi.SELinuxOptions{User: "foo"}},
			expectedPSP:         runAsAny.Name,
		},

		"mustRunAs with bad pod request": {
			pod:                createPodWithSecurityContexts(&kapi.PodSecurityContext{SELinuxOptions: &kapi.SELinuxOptions{User: "foo"}}, nil),
			psps:               []*policy.PodSecurityPolicy{mustRunAs},
			shouldPassAdmit:    false,
			shouldPassValidate: false,
		},
		"mustRunAs with bad container request": {
			pod:                createPodWithSecurityContexts(nil, &kapi.SecurityContext{SELinuxOptions: &kapi.SELinuxOptions{User: "foo"}}),
			psps:               []*policy.PodSecurityPolicy{mustRunAs},
			shouldPassAdmit:    false,
			shouldPassValidate: false,
		},
		"mustRunAs with no request": {
			pod:                 createPodWithSecurityContexts(nil, nil),
			psps:                []*policy.PodSecurityPolicy{mustRunAs},
			shouldPassAdmit:     true,
			shouldPassValidate:  true,
			expectedPodSC:       &kapi.PodSecurityContext{SELinuxOptions: getInternalSEOptions(mustRunAs)},
			expectedContainerSC: nil,
			expectedPSP:         mustRunAs.Name,
		},
		"mustRunAs with good pod request": {
			pod: createPodWithSecurityContexts(
				&kapi.PodSecurityContext{SELinuxOptions: &kapi.SELinuxOptions{Level: "level", Role: "role", Type: "type", User: "user"}},
				nil,
			),
			psps:                []*policy.PodSecurityPolicy{mustRunAs},
			shouldPassAdmit:     true,
			shouldPassValidate:  true,
			expectedPodSC:       &kapi.PodSecurityContext{SELinuxOptions: getInternalSEOptions(mustRunAs)},
			expectedContainerSC: nil,
			expectedPSP:         mustRunAs.Name,
		},
		"mustRunAs with good container request": {
			pod: createPodWithSecurityContexts(
				&kapi.PodSecurityContext{SELinuxOptions: &kapi.SELinuxOptions{Level: "level", Role: "role", Type: "type", User: "user"}},
				nil,
			),
			psps:                []*policy.PodSecurityPolicy{mustRunAs},
			shouldPassAdmit:     true,
			shouldPassValidate:  true,
			expectedPodSC:       &kapi.PodSecurityContext{SELinuxOptions: getInternalSEOptions(mustRunAs)},
			expectedContainerSC: nil,
			expectedPSP:         mustRunAs.Name,
		},
	}

	for k, v := range tests {
		testPSPAdmit(k, v.psps, v.pod, v.shouldPassAdmit, v.shouldPassValidate, v.expectedPSP, t)

		if v.shouldPassAdmit {
			if !reflect.DeepEqual(v.expectedPodSC, v.pod.Spec.SecurityContext) {
				t.Errorf("%s unexpected diff:\n%s", k, diff.ObjectGoPrintSideBySide(v.expectedPodSC, v.pod.Spec.SecurityContext))
			}
			if !reflect.DeepEqual(v.expectedContainerSC, v.pod.Spec.Containers[0].SecurityContext) {
				t.Errorf("%s unexpected diff:\n%s", k, diff.ObjectGoPrintSideBySide(v.expectedContainerSC, v.pod.Spec.Containers[0].SecurityContext))
			}
		}
	}
}

func TestAdmitAppArmor(t *testing.T) {
	createPodWithAppArmor := func(profile string) *kapi.Pod {
		pod := goodPod()
		apparmor.SetProfileNameFromPodAnnotations(pod.Annotations, defaultContainerName, profile)
		return pod
	}

	unconstrainedPSP := restrictivePSP()
	defaultedPSP := restrictivePSP()
	defaultedPSP.Annotations = map[string]string{
		v1.AppArmorBetaDefaultProfileAnnotationKey: v1.AppArmorBetaProfileRuntimeDefault,
	}
	appArmorPSP := restrictivePSP()
	appArmorPSP.Annotations = map[string]string{
		v1.AppArmorBetaAllowedProfilesAnnotationKey: v1.AppArmorBetaProfileRuntimeDefault,
	}
	appArmorDefaultPSP := restrictivePSP()
	appArmorDefaultPSP.Annotations = map[string]string{
		v1.AppArmorBetaDefaultProfileAnnotationKey:  v1.AppArmorBetaProfileRuntimeDefault,
		v1.AppArmorBetaAllowedProfilesAnnotationKey: v1.AppArmorBetaProfileRuntimeDefault + "," + v1.AppArmorBetaProfileNamePrefix + "foo",
	}

	tests := map[string]struct {
		pod                *kapi.Pod
		psp                *policy.PodSecurityPolicy
		shouldPassAdmit    bool
		shouldPassValidate bool
		expectedProfile    string
	}{
		"unconstrained with no profile": {
			pod:                goodPod(),
			psp:                unconstrainedPSP,
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedProfile:    "",
		},
		"unconstrained with profile": {
			pod:                createPodWithAppArmor(v1.AppArmorBetaProfileRuntimeDefault),
			psp:                unconstrainedPSP,
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedProfile:    v1.AppArmorBetaProfileRuntimeDefault,
		},
		"unconstrained with default profile": {
			pod:                goodPod(),
			psp:                defaultedPSP,
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedProfile:    v1.AppArmorBetaProfileRuntimeDefault,
		},
		"AppArmor enforced with no profile": {
			pod:                goodPod(),
			psp:                appArmorPSP,
			shouldPassAdmit:    false,
			shouldPassValidate: false,
		},
		"AppArmor enforced with default profile": {
			pod:                goodPod(),
			psp:                appArmorDefaultPSP,
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedProfile:    v1.AppArmorBetaProfileRuntimeDefault,
		},
		"AppArmor enforced with good profile": {
			pod:                createPodWithAppArmor(v1.AppArmorBetaProfileNamePrefix + "foo"),
			psp:                appArmorDefaultPSP,
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedProfile:    v1.AppArmorBetaProfileNamePrefix + "foo",
		},
		"AppArmor enforced with local profile": {
			pod:                createPodWithAppArmor(v1.AppArmorBetaProfileNamePrefix + "bar"),
			psp:                appArmorPSP,
			shouldPassAdmit:    false,
			shouldPassValidate: false,
		},
	}

	for k, v := range tests {
		testPSPAdmit(k, []*policy.PodSecurityPolicy{v.psp}, v.pod, v.shouldPassAdmit, v.shouldPassValidate, v.psp.Name, t)

		if v.shouldPassAdmit {
			assert.Equal(t, v.expectedProfile, apparmor.GetProfileNameFromPodAnnotations(v.pod.Annotations, defaultContainerName), k)
		}
	}
}

func TestAdmitRunAsUser(t *testing.T) {
	podSC := func(user *int64) *kapi.PodSecurityContext {
		return &kapi.PodSecurityContext{RunAsUser: user}
	}
	containerSC := func(user *int64) *kapi.SecurityContext {
		return &kapi.SecurityContext{RunAsUser: user}
	}

	runAsAny := permissivePSP()
	runAsAny.Name = "runAsAny"
	runAsAny.Spec.RunAsUser.Rule = policy.RunAsUserStrategyRunAsAny

	mustRunAs := permissivePSP()
	mustRunAs.Name = "mustRunAs"
	mustRunAs.Spec.RunAsUser.Rule = policy.RunAsUserStrategyMustRunAs
	mustRunAs.Spec.RunAsUser.Ranges = []policy.IDRange{
		{Min: int64(999), Max: int64(1000)},
	}

	runAsNonRoot := permissivePSP()
	runAsNonRoot.Name = "runAsNonRoot"
	runAsNonRoot.Spec.RunAsUser.Rule = policy.RunAsUserStrategyMustRunAsNonRoot

	trueValue := true

	tests := map[string]struct {
		pod                 *kapi.Pod
		psps                []*policy.PodSecurityPolicy
		shouldPassAdmit     bool
		shouldPassValidate  bool
		expectedPodSC       *kapi.PodSecurityContext
		expectedContainerSC *kapi.SecurityContext
		expectedPSP         string
	}{
		"runAsAny no pod request": {
			pod:                 createPodWithSecurityContexts(nil, nil),
			psps:                []*policy.PodSecurityPolicy{runAsAny},
			shouldPassAdmit:     true,
			shouldPassValidate:  true,
			expectedPodSC:       nil,
			expectedContainerSC: nil,
			expectedPSP:         runAsAny.Name,
		},
		"runAsAny pod request": {
			pod:                 createPodWithSecurityContexts(podSC(utilpointer.Int64Ptr(1)), nil),
			psps:                []*policy.PodSecurityPolicy{runAsAny},
			shouldPassAdmit:     true,
			shouldPassValidate:  true,
			expectedPodSC:       podSC(utilpointer.Int64Ptr(1)),
			expectedContainerSC: nil,
			expectedPSP:         runAsAny.Name,
		},
		"runAsAny container request": {
			pod:                 createPodWithSecurityContexts(nil, containerSC(utilpointer.Int64Ptr(1))),
			psps:                []*policy.PodSecurityPolicy{runAsAny},
			shouldPassAdmit:     true,
			shouldPassValidate:  true,
			expectedPodSC:       nil,
			expectedContainerSC: containerSC(utilpointer.Int64Ptr(1)),
			expectedPSP:         runAsAny.Name,
		},

		"mustRunAs pod request out of range": {
			pod:                createPodWithSecurityContexts(podSC(utilpointer.Int64Ptr(1)), nil),
			psps:               []*policy.PodSecurityPolicy{mustRunAs},
			shouldPassAdmit:    false,
			shouldPassValidate: false,
		},
		"mustRunAs container request out of range": {
			pod:                createPodWithSecurityContexts(podSC(utilpointer.Int64Ptr(999)), containerSC(utilpointer.Int64Ptr(1))),
			psps:               []*policy.PodSecurityPolicy{mustRunAs},
			shouldPassAdmit:    false,
			shouldPassValidate: false,
		},

		"mustRunAs pod request in range": {
			pod:                 createPodWithSecurityContexts(podSC(utilpointer.Int64Ptr(999)), nil),
			psps:                []*policy.PodSecurityPolicy{mustRunAs},
			shouldPassAdmit:     true,
			shouldPassValidate:  true,
			expectedPodSC:       podSC(&mustRunAs.Spec.RunAsUser.Ranges[0].Min),
			expectedContainerSC: nil,
			expectedPSP:         mustRunAs.Name,
		},
		"mustRunAs container request in range": {
			pod:                 createPodWithSecurityContexts(nil, containerSC(utilpointer.Int64Ptr(999))),
			psps:                []*policy.PodSecurityPolicy{mustRunAs},
			shouldPassAdmit:     true,
			shouldPassValidate:  true,
			expectedPodSC:       nil,
			expectedContainerSC: containerSC(&mustRunAs.Spec.RunAsUser.Ranges[0].Min),
			expectedPSP:         mustRunAs.Name,
		},
		"mustRunAs pod and container request in range": {
			pod:                 createPodWithSecurityContexts(podSC(utilpointer.Int64Ptr(999)), containerSC(utilpointer.Int64Ptr(1000))),
			psps:                []*policy.PodSecurityPolicy{mustRunAs},
			shouldPassAdmit:     true,
			shouldPassValidate:  true,
			expectedPodSC:       podSC(utilpointer.Int64Ptr(999)),
			expectedContainerSC: containerSC(utilpointer.Int64Ptr(1000)),
			expectedPSP:         mustRunAs.Name,
		},
		"mustRunAs no request": {
			pod:                 createPodWithSecurityContexts(nil, nil),
			psps:                []*policy.PodSecurityPolicy{mustRunAs},
			shouldPassAdmit:     true,
			shouldPassValidate:  true,
			expectedPodSC:       nil,
			expectedContainerSC: containerSC(&mustRunAs.Spec.RunAsUser.Ranges[0].Min),
			expectedPSP:         mustRunAs.Name,
		},

		"runAsNonRoot no request": {
			pod:                 createPodWithSecurityContexts(nil, nil),
			psps:                []*policy.PodSecurityPolicy{runAsNonRoot},
			shouldPassAdmit:     true,
			shouldPassValidate:  true,
			expectedPodSC:       nil,
			expectedContainerSC: &kapi.SecurityContext{RunAsNonRoot: &trueValue},
			expectedPSP:         runAsNonRoot.Name,
		},
		"runAsNonRoot pod request root": {
			pod:                createPodWithSecurityContexts(podSC(utilpointer.Int64Ptr(0)), nil),
			psps:               []*policy.PodSecurityPolicy{runAsNonRoot},
			shouldPassAdmit:    false,
			shouldPassValidate: false,
		},
		"runAsNonRoot pod request non-root": {
			pod:                createPodWithSecurityContexts(podSC(utilpointer.Int64Ptr(1)), nil),
			psps:               []*policy.PodSecurityPolicy{runAsNonRoot},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedPodSC:      podSC(utilpointer.Int64Ptr(1)),
			expectedPSP:        runAsNonRoot.Name,
		},
		"runAsNonRoot container request root": {
			pod:                createPodWithSecurityContexts(podSC(utilpointer.Int64Ptr(1)), containerSC(utilpointer.Int64Ptr(0))),
			psps:               []*policy.PodSecurityPolicy{runAsNonRoot},
			shouldPassAdmit:    false,
			shouldPassValidate: false,
		},
		"runAsNonRoot container request non-root": {
			pod:                 createPodWithSecurityContexts(podSC(utilpointer.Int64Ptr(1)), containerSC(utilpointer.Int64Ptr(2))),
			psps:                []*policy.PodSecurityPolicy{runAsNonRoot},
			shouldPassAdmit:     true,
			shouldPassValidate:  true,
			expectedPodSC:       podSC(utilpointer.Int64Ptr(1)),
			expectedContainerSC: containerSC(utilpointer.Int64Ptr(2)),
			expectedPSP:         runAsNonRoot.Name,
		},
	}

	for k, v := range tests {
		testPSPAdmit(k, v.psps, v.pod, v.shouldPassAdmit, v.shouldPassValidate, v.expectedPSP, t)

		if v.shouldPassAdmit {
			if !reflect.DeepEqual(v.expectedPodSC, v.pod.Spec.SecurityContext) {
				t.Errorf("%s unexpected pod sc diff:\n%s", k, diff.ObjectGoPrintSideBySide(v.expectedPodSC, v.pod.Spec.SecurityContext))
			}
			if !reflect.DeepEqual(v.expectedContainerSC, v.pod.Spec.Containers[0].SecurityContext) {
				t.Errorf("%s unexpected container sc diff:\n%s", k, diff.ObjectGoPrintSideBySide(v.expectedContainerSC, v.pod.Spec.Containers[0].SecurityContext))
			}
		}
	}
}

func TestAdmitSupplementalGroups(t *testing.T) {
	podSC := func(group int64) *kapi.PodSecurityContext {
		return &kapi.PodSecurityContext{SupplementalGroups: []int64{group}}
	}

	runAsAny := permissivePSP()
	runAsAny.Name = "runAsAny"
	runAsAny.Spec.SupplementalGroups.Rule = policy.SupplementalGroupsStrategyRunAsAny

	mustRunAs := permissivePSP()
	mustRunAs.Name = "mustRunAs"
	mustRunAs.Spec.SupplementalGroups.Rule = policy.SupplementalGroupsStrategyMustRunAs
	mustRunAs.Spec.SupplementalGroups.Ranges = []policy.IDRange{{Min: int64(999), Max: int64(1000)}}

	tests := map[string]struct {
		pod                *kapi.Pod
		psps               []*policy.PodSecurityPolicy
		shouldPassAdmit    bool
		shouldPassValidate bool
		expectedPodSC      *kapi.PodSecurityContext
		expectedPSP        string
	}{
		"runAsAny no pod request": {
			pod:                createPodWithSecurityContexts(nil, nil),
			psps:               []*policy.PodSecurityPolicy{runAsAny},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedPodSC:      nil,
			expectedPSP:        runAsAny.Name,
		},
		"runAsAny empty pod request": {
			pod:                createPodWithSecurityContexts(&kapi.PodSecurityContext{}, nil),
			psps:               []*policy.PodSecurityPolicy{runAsAny},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedPodSC:      &kapi.PodSecurityContext{},
			expectedPSP:        runAsAny.Name,
		},
		"runAsAny empty pod request empty supplemental groups": {
			pod:                createPodWithSecurityContexts(&kapi.PodSecurityContext{SupplementalGroups: []int64{}}, nil),
			psps:               []*policy.PodSecurityPolicy{runAsAny},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedPodSC:      &kapi.PodSecurityContext{SupplementalGroups: []int64{}},
			expectedPSP:        runAsAny.Name,
		},
		"runAsAny pod request": {
			pod:                createPodWithSecurityContexts(podSC(1), nil),
			psps:               []*policy.PodSecurityPolicy{runAsAny},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedPodSC:      &kapi.PodSecurityContext{SupplementalGroups: []int64{1}},
			expectedPSP:        runAsAny.Name,
		},
		"mustRunAs no pod request": {
			pod:                createPodWithSecurityContexts(nil, nil),
			psps:               []*policy.PodSecurityPolicy{mustRunAs},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedPodSC:      podSC(mustRunAs.Spec.SupplementalGroups.Ranges[0].Min),
			expectedPSP:        mustRunAs.Name,
		},
		"mustRunAs bad pod request": {
			pod:                createPodWithSecurityContexts(podSC(1), nil),
			psps:               []*policy.PodSecurityPolicy{mustRunAs},
			shouldPassAdmit:    false,
			shouldPassValidate: false,
		},
		"mustRunAs good pod request": {
			pod:                createPodWithSecurityContexts(podSC(999), nil),
			psps:               []*policy.PodSecurityPolicy{mustRunAs},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedPodSC:      podSC(999),
			expectedPSP:        mustRunAs.Name,
		},
	}

	for k, v := range tests {
		testPSPAdmit(k, v.psps, v.pod, v.shouldPassAdmit, v.shouldPassValidate, v.expectedPSP, t)

		if v.shouldPassAdmit {
			if !reflect.DeepEqual(v.expectedPodSC, v.pod.Spec.SecurityContext) {
				t.Errorf("%s unexpected pod sc diff:\n%s", k, diff.ObjectGoPrintSideBySide(v.expectedPodSC, v.pod.Spec.SecurityContext))
			}
		}
	}
}

func TestAdmitFSGroup(t *testing.T) {
	createPodWithFSGroup := func(group int64) *kapi.Pod {
		pod := goodPod()
		// doesn't matter if we set it here or on the container, the
		// admission controller uses DetermineEffectiveSC to get the defaulting
		// behavior so it can validate what will be applied at runtime
		pod.Spec.SecurityContext.FSGroup = utilpointer.Int64Ptr(group)
		return pod
	}

	runAsAny := restrictivePSP()
	runAsAny.Name = "runAsAny"
	runAsAny.Spec.FSGroup.Rule = policy.FSGroupStrategyRunAsAny

	mustRunAs := restrictivePSP()
	mustRunAs.Name = "mustRunAs"

	tests := map[string]struct {
		pod                *kapi.Pod
		psps               []*policy.PodSecurityPolicy
		shouldPassAdmit    bool
		shouldPassValidate bool
		expectedFSGroup    *int64
		expectedPSP        string
	}{
		"runAsAny no pod request": {
			pod:                goodPod(),
			psps:               []*policy.PodSecurityPolicy{runAsAny},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedFSGroup:    nil,
			expectedPSP:        runAsAny.Name,
		},
		"runAsAny pod request": {
			pod:                createPodWithFSGroup(1),
			psps:               []*policy.PodSecurityPolicy{runAsAny},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedFSGroup:    utilpointer.Int64Ptr(1),
			expectedPSP:        runAsAny.Name,
		},
		"mustRunAs no pod request": {
			pod:                goodPod(),
			psps:               []*policy.PodSecurityPolicy{mustRunAs},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedFSGroup:    &mustRunAs.Spec.SupplementalGroups.Ranges[0].Min,
			expectedPSP:        mustRunAs.Name,
		},
		"mustRunAs bad pod request": {
			pod:                createPodWithFSGroup(1),
			psps:               []*policy.PodSecurityPolicy{mustRunAs},
			shouldPassAdmit:    false,
			shouldPassValidate: false,
		},
		"mustRunAs good pod request": {
			pod:                createPodWithFSGroup(999),
			psps:               []*policy.PodSecurityPolicy{mustRunAs},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedFSGroup:    utilpointer.Int64Ptr(999),
			expectedPSP:        mustRunAs.Name,
		},
	}

	for k, v := range tests {
		testPSPAdmit(k, v.psps, v.pod, v.shouldPassAdmit, v.shouldPassValidate, v.expectedPSP, t)

		if v.shouldPassAdmit {
			if v.pod.Spec.SecurityContext.FSGroup == nil && v.expectedFSGroup == nil {
				// ok, don't need to worry about identifying specific diffs
				continue
			}
			if v.pod.Spec.SecurityContext.FSGroup == nil && v.expectedFSGroup != nil {
				t.Errorf("%s expected FSGroup to be: %v but found nil", k, *v.expectedFSGroup)
				continue
			}
			if v.pod.Spec.SecurityContext.FSGroup != nil && v.expectedFSGroup == nil {
				t.Errorf("%s expected FSGroup to be nil but found: %v", k, *v.pod.Spec.SecurityContext.FSGroup)
				continue
			}
			if *v.expectedFSGroup != *v.pod.Spec.SecurityContext.FSGroup {
				t.Errorf("%s expected FSGroup to be: %v but found %v", k, *v.expectedFSGroup, *v.pod.Spec.SecurityContext.FSGroup)
			}
		}
	}
}

func TestAdmitReadOnlyRootFilesystem(t *testing.T) {
	createPodWithRORFS := func(rorfs bool) *kapi.Pod {
		pod := goodPod()
		pod.Spec.Containers[0].SecurityContext.ReadOnlyRootFilesystem = &rorfs
		return pod
	}

	noRORFS := restrictivePSP()
	noRORFS.Name = "no-rorfs"
	noRORFS.Spec.ReadOnlyRootFilesystem = false

	rorfs := restrictivePSP()
	rorfs.Name = "rorfs"
	rorfs.Spec.ReadOnlyRootFilesystem = true

	tests := map[string]struct {
		pod                *kapi.Pod
		psps               []*policy.PodSecurityPolicy
		shouldPassAdmit    bool
		shouldPassValidate bool
		expectedRORFS      bool
		expectedPSP        string
	}{
		"no-rorfs allows pod request with rorfs": {
			pod:                createPodWithRORFS(true),
			psps:               []*policy.PodSecurityPolicy{noRORFS},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedRORFS:      true,
			expectedPSP:        noRORFS.Name,
		},
		"no-rorfs allows pod request without rorfs": {
			pod:                createPodWithRORFS(false),
			psps:               []*policy.PodSecurityPolicy{noRORFS},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedRORFS:      false,
			expectedPSP:        noRORFS.Name,
		},
		"rorfs rejects pod request without rorfs": {
			pod:                createPodWithRORFS(false),
			psps:               []*policy.PodSecurityPolicy{rorfs},
			shouldPassAdmit:    false,
			shouldPassValidate: false,
		},
		"rorfs defaults nil pod request": {
			pod:                goodPod(),
			psps:               []*policy.PodSecurityPolicy{rorfs},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedRORFS:      true,
			expectedPSP:        rorfs.Name,
		},
		"rorfs accepts pod request with rorfs": {
			pod:                createPodWithRORFS(true),
			psps:               []*policy.PodSecurityPolicy{rorfs},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedRORFS:      true,
			expectedPSP:        rorfs.Name,
		},
	}

	for k, v := range tests {
		testPSPAdmit(k, v.psps, v.pod, v.shouldPassAdmit, v.shouldPassValidate, v.expectedPSP, t)

		if v.shouldPassAdmit {
			if v.pod.Spec.Containers[0].SecurityContext.ReadOnlyRootFilesystem == nil ||
				*v.pod.Spec.Containers[0].SecurityContext.ReadOnlyRootFilesystem != v.expectedRORFS {
				t.Errorf("%s expected ReadOnlyRootFilesystem to be %t but found %#v", k, v.expectedRORFS, v.pod.Spec.Containers[0].SecurityContext.ReadOnlyRootFilesystem)
			}
		}
	}
}

func TestAdmitSysctls(t *testing.T) {
	podWithSysctls := func(safeSysctls []string, unsafeSysctls []string) *kapi.Pod {
		pod := goodPod()
		dummySysctls := func(names []string) []kapi.Sysctl {
			sysctls := make([]kapi.Sysctl, len(names))
			for i, n := range names {
				sysctls[i].Name = n
				sysctls[i].Value = "dummy"
			}
			return sysctls
		}
		pod.Spec.SecurityContext = &kapi.PodSecurityContext{
			Sysctls: dummySysctls(append(safeSysctls, unsafeSysctls...)),
		}

		return pod
	}

	safeSysctls := restrictivePSP()
	safeSysctls.Name = "no sysctls"

	noSysctls := restrictivePSP()
	noSysctls.Name = "empty sysctls"
	noSysctls.Spec.ForbiddenSysctls = []string{"*"}

	mixedSysctls := restrictivePSP()
	mixedSysctls.Name = "wildcard sysctls"
	mixedSysctls.Spec.ForbiddenSysctls = []string{"net.*"}
	mixedSysctls.Spec.AllowedUnsafeSysctls = []string{"a.*", "b.*"}

	aUnsafeSysctl := restrictivePSP()
	aUnsafeSysctl.Name = "a sysctl"
	aUnsafeSysctl.Spec.AllowedUnsafeSysctls = []string{"a"}

	bUnsafeSysctl := restrictivePSP()
	bUnsafeSysctl.Name = "b sysctl"
	bUnsafeSysctl.Spec.AllowedUnsafeSysctls = []string{"b"}

	cUnsafeSysctl := restrictivePSP()
	cUnsafeSysctl.Name = "c sysctl"
	cUnsafeSysctl.Spec.AllowedUnsafeSysctls = []string{"c"}

	catchallSysctls := restrictivePSP()
	catchallSysctls.Name = "catchall sysctl"
	catchallSysctls.Spec.AllowedUnsafeSysctls = []string{"*"}

	tests := map[string]struct {
		pod                *kapi.Pod
		psps               []*policy.PodSecurityPolicy
		shouldPassAdmit    bool
		shouldPassValidate bool
		expectedPSP        string
	}{
		"pod without any sysctls request allowed under safeSysctls PSP": {
			pod:                goodPod(),
			psps:               []*policy.PodSecurityPolicy{safeSysctls},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedPSP:        safeSysctls.Name,
		},
		"pod without any sysctls request allowed under noSysctls PSP": {
			pod:                goodPod(),
			psps:               []*policy.PodSecurityPolicy{noSysctls},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedPSP:        noSysctls.Name,
		},
		"pod with safe sysctls request allowed under safeSysctls PSP": {
			pod:                podWithSysctls([]string{"kernel.shm_rmid_forced", "net.ipv4.tcp_syncookies"}, []string{}),
			psps:               []*policy.PodSecurityPolicy{safeSysctls},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedPSP:        safeSysctls.Name,
		},
		"pod with unsafe sysctls request disallowed under noSysctls PSP": {
			pod:                podWithSysctls([]string{}, []string{"a", "b"}),
			psps:               []*policy.PodSecurityPolicy{noSysctls},
			shouldPassAdmit:    false,
			shouldPassValidate: false,
			expectedPSP:        noSysctls.Name,
		},
		"pod with unsafe sysctls a, b request disallowed under aUnsafeSysctl SCC": {
			pod:                podWithSysctls([]string{}, []string{"a", "b"}),
			psps:               []*policy.PodSecurityPolicy{aUnsafeSysctl},
			shouldPassAdmit:    false,
			shouldPassValidate: false,
		},
		"pod with unsafe sysctls b request disallowed under aUnsafeSysctl SCC": {
			pod:                podWithSysctls([]string{}, []string{"b"}),
			psps:               []*policy.PodSecurityPolicy{aUnsafeSysctl},
			shouldPassAdmit:    false,
			shouldPassValidate: false,
		},
		"pod with unsafe sysctls a request allowed under aUnsafeSysctl SCC": {
			pod:                podWithSysctls([]string{}, []string{"a"}),
			psps:               []*policy.PodSecurityPolicy{aUnsafeSysctl},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedPSP:        aUnsafeSysctl.Name,
		},
		"pod with safe net sysctl request allowed under aUnsafeSysctl SCC": {
			pod:                podWithSysctls([]string{"net.ipv4.ip_local_port_range"}, []string{}),
			psps:               []*policy.PodSecurityPolicy{aUnsafeSysctl},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedPSP:        aUnsafeSysctl.Name,
		},
		"pod with safe sysctls request disallowed under noSysctls PSP": {
			pod:                podWithSysctls([]string{"net.ipv4.ip_local_port_range"}, []string{}),
			psps:               []*policy.PodSecurityPolicy{noSysctls},
			shouldPassAdmit:    false,
			shouldPassValidate: false,
		},
		"pod with matching sysctls request allowed under mixedSysctls PSP": {
			pod:                podWithSysctls([]string{"kernel.shm_rmid_forced"}, []string{"a.b", "b.a"}),
			psps:               []*policy.PodSecurityPolicy{mixedSysctls},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedPSP:        mixedSysctls.Name,
		},
		"pod with not-matching unsafe sysctls request disallowed under mixedSysctls PSP": {
			pod:                podWithSysctls([]string{}, []string{"e"}),
			psps:               []*policy.PodSecurityPolicy{mixedSysctls},
			shouldPassAdmit:    false,
			shouldPassValidate: false,
		},
		"pod with not-matching safe sysctls request disallowed under mixedSysctls PSP": {
			pod:                podWithSysctls([]string{"net.ipv4.ip_local_port_range"}, []string{}),
			psps:               []*policy.PodSecurityPolicy{mixedSysctls},
			shouldPassAdmit:    false,
			shouldPassValidate: false,
		},
		"pod with sysctls request allowed under catchallSysctls PSP": {
			pod:                podWithSysctls([]string{"net.ipv4.ip_local_port_range"}, []string{"f"}),
			psps:               []*policy.PodSecurityPolicy{catchallSysctls},
			shouldPassAdmit:    true,
			shouldPassValidate: true,
			expectedPSP:        catchallSysctls.Name,
		},
	}

	for k, v := range tests {
		origSysctl := v.pod.Spec.SecurityContext.Sysctls

		testPSPAdmit(k, v.psps, v.pod, v.shouldPassAdmit, v.shouldPassValidate, v.expectedPSP, t)

		if v.shouldPassAdmit {
			if !reflect.DeepEqual(v.pod.Spec.SecurityContext.Sysctls, origSysctl) {
				t.Errorf("%s: wrong sysctls: expected=%v, got=%v", k, origSysctl, v.pod.Spec.SecurityContext.Sysctls)
			}
		}
	}
}

func testPSPAdmit(testCaseName string, psps []*policy.PodSecurityPolicy, pod *kapi.Pod, shouldPassAdmit, shouldPassValidate bool, expectedPSP string, t *testing.T) {
	testPSPAdmitAdvanced(testCaseName, kadmission.Create, psps, nil, &user.DefaultInfo{}, pod, nil, shouldPassAdmit, shouldPassValidate, true, expectedPSP, t)
}

// fakeAttributes decorate kadmission.Attributes. It's used to trace the added annotations.
type fakeAttributes struct {
	kadmission.Attributes
	annotations map[string]string
}

func (f fakeAttributes) AddAnnotation(k, v string) error {
	f.annotations[k] = v
	return f.Attributes.AddAnnotation(k, v)
}

func testPSPAdmitAdvanced(testCaseName string, op kadmission.Operation, psps []*policy.PodSecurityPolicy, authz authorizer.Authorizer, userInfo user.Info, pod, oldPod *kapi.Pod, shouldPassAdmit, shouldPassValidate bool, canMutate bool, expectedPSP string, t *testing.T) {
	originalPod := pod.DeepCopy()
	plugin := NewTestAdmission(psps, authz)

	attrs := kadmission.NewAttributesRecord(pod, oldPod, kapi.Kind("Pod").WithVersion("version"), pod.Namespace, "", kapi.Resource("pods").WithVersion("version"), "", op, nil, false, userInfo)
	annotations := make(map[string]string)
	attrs = &fakeAttributes{attrs, annotations}
	err := admissiontesting.WithReinvocationTesting(t, plugin).Admit(context.TODO(), attrs, nil)

	if shouldPassAdmit && err != nil {
		t.Errorf("%s: expected no errors on Admit but received %v", testCaseName, err)
	}

	if shouldPassAdmit && err == nil {
		if pod.Annotations[psputil.ValidatedPSPAnnotation] != expectedPSP {
			t.Errorf("%s: expected to be admitted under %q PSP but found %q", testCaseName, expectedPSP, pod.Annotations[psputil.ValidatedPSPAnnotation])
		}

		if !canMutate {
			podWithoutPSPAnnotation := pod.DeepCopy()
			delete(podWithoutPSPAnnotation.Annotations, psputil.ValidatedPSPAnnotation)

			originalPodWithoutPSPAnnotation := originalPod.DeepCopy()
			delete(originalPodWithoutPSPAnnotation.Annotations, psputil.ValidatedPSPAnnotation)

			if !apiequality.Semantic.DeepEqual(originalPodWithoutPSPAnnotation.Spec, podWithoutPSPAnnotation.Spec) {
				t.Errorf("%s: expected no mutation on Admit, got %s", testCaseName, diff.ObjectGoPrintSideBySide(originalPodWithoutPSPAnnotation.Spec, podWithoutPSPAnnotation.Spec))
			}
		}
	}

	if !shouldPassAdmit && err == nil {
		t.Errorf("%s: expected errors on Admit but received none", testCaseName)
	}

	err = plugin.Validate(context.TODO(), attrs, nil)
	psp := ""
	if shouldPassAdmit && op == kadmission.Create {
		psp = expectedPSP
	}
	validateAuditAnnotation(t, testCaseName, annotations, "podsecuritypolicy.policy.k8s.io/admit-policy", psp)
	if shouldPassValidate && err != nil {
		t.Errorf("%s: expected no errors on Validate but received %v", testCaseName, err)
	} else if !shouldPassValidate && err == nil {
		t.Errorf("%s: expected errors on Validate but received none", testCaseName)
	}
	if shouldPassValidate {
		validateAuditAnnotation(t, testCaseName, annotations, "podsecuritypolicy.policy.k8s.io/validate-policy", expectedPSP)
	} else {
		validateAuditAnnotation(t, testCaseName, annotations, "podsecuritypolicy.policy.k8s.io/validate-policy", "")
	}
}

func validateAuditAnnotation(t *testing.T, testCaseName string, annotations map[string]string, key, value string) {
	if annotations[key] != value {
		t.Errorf("%s: expected to have annotations[%s] set to %q, got %q", testCaseName, key, value, annotations[key])
	}
}

func TestAssignSecurityContext(t *testing.T) {
	// psp that will deny privileged container requests and has a default value for a field (uid)
	psp := restrictivePSP()
	provider, err := kpsp.NewSimpleProvider(psp, "namespace", kpsp.NewSimpleStrategyFactory())
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	createContainer := func(priv bool) kapi.Container {
		return kapi.Container{
			SecurityContext: &kapi.SecurityContext{
				Privileged: &priv,
			},
		}
	}

	testCases := map[string]struct {
		pod            *kapi.Pod
		shouldValidate bool
		expectedUID    *int64
	}{
		"pod and container SC is not changed when invalid": {
			pod: &kapi.Pod{
				Spec: kapi.PodSpec{
					SecurityContext: &kapi.PodSecurityContext{},
					Containers:      []kapi.Container{createContainer(true)},
				},
			},
			shouldValidate: false,
		},
		"must validate all containers": {
			pod: &kapi.Pod{
				Spec: kapi.PodSpec{
					// good container and bad container
					SecurityContext: &kapi.PodSecurityContext{},
					Containers:      []kapi.Container{createContainer(false), createContainer(true)},
				},
			},
			shouldValidate: false,
		},
		"pod validates": {
			pod: &kapi.Pod{
				Spec: kapi.PodSpec{
					SecurityContext: &kapi.PodSecurityContext{},
					Containers:      []kapi.Container{createContainer(false)},
				},
			},
			shouldValidate: true,
		},
	}

	for k, v := range testCases {
		errs := assignSecurityContext(provider, v.pod)
		if v.shouldValidate && len(errs) > 0 {
			t.Errorf("%s expected to validate but received errors %v", k, errs)
			continue
		}
		if !v.shouldValidate && len(errs) == 0 {
			t.Errorf("%s expected validation errors but received none", k)
			continue
		}
	}
}

func TestCreateProvidersFromConstraints(t *testing.T) {
	testCases := map[string]struct {
		// use a generating function so we can test for non-mutation
		psp         func() *policy.PodSecurityPolicy
		expectedErr string
	}{
		"valid psp": {
			psp: func() *policy.PodSecurityPolicy {
				return &policy.PodSecurityPolicy{
					ObjectMeta: metav1.ObjectMeta{
						Name: "valid psp",
					},
					Spec: policy.PodSecurityPolicySpec{
						SELinux: policy.SELinuxStrategyOptions{
							Rule: policy.SELinuxStrategyRunAsAny,
						},
						RunAsUser: policy.RunAsUserStrategyOptions{
							Rule: policy.RunAsUserStrategyRunAsAny,
						},
						RunAsGroup: &policy.RunAsGroupStrategyOptions{
							Rule: policy.RunAsGroupStrategyRunAsAny,
						},
						FSGroup: policy.FSGroupStrategyOptions{
							Rule: policy.FSGroupStrategyRunAsAny,
						},
						SupplementalGroups: policy.SupplementalGroupsStrategyOptions{
							Rule: policy.SupplementalGroupsStrategyRunAsAny,
						},
					},
				}
			},
		},
		"bad psp strategy options": {
			psp: func() *policy.PodSecurityPolicy {
				return &policy.PodSecurityPolicy{
					ObjectMeta: metav1.ObjectMeta{
						Name: "bad psp user options",
					},
					Spec: policy.PodSecurityPolicySpec{
						SELinux: policy.SELinuxStrategyOptions{
							Rule: policy.SELinuxStrategyRunAsAny,
						},
						RunAsUser: policy.RunAsUserStrategyOptions{
							Rule: policy.RunAsUserStrategyMustRunAs,
						},
						RunAsGroup: &policy.RunAsGroupStrategyOptions{
							Rule: policy.RunAsGroupStrategyRunAsAny,
						},
						FSGroup: policy.FSGroupStrategyOptions{
							Rule: policy.FSGroupStrategyRunAsAny,
						},
						SupplementalGroups: policy.SupplementalGroupsStrategyOptions{
							Rule: policy.SupplementalGroupsStrategyRunAsAny,
						},
					},
				}
			},
			expectedErr: "MustRunAs requires at least one range",
		},
	}

	for k, v := range testCases {
		admit := &Plugin{
			Handler:         kadmission.NewHandler(kadmission.Create, kadmission.Update),
			strategyFactory: kpsp.NewSimpleStrategyFactory(),
		}

		psp := v.psp()
		_, errs := admit.createProvidersFromPolicies([]*policy.PodSecurityPolicy{psp}, "namespace")

		if !reflect.DeepEqual(psp, v.psp()) {
			diff := diff.ObjectDiff(psp, v.psp())
			t.Errorf("%s createProvidersFromPolicies mutated policy. diff:\n%s", k, diff)
		}
		if len(v.expectedErr) > 0 && len(errs) != 1 {
			t.Errorf("%s expected a single error '%s' but received %v", k, v.expectedErr, errs)
			continue
		}
		if len(v.expectedErr) == 0 && len(errs) != 0 {
			t.Errorf("%s did not expect an error but received %v", k, errs)
			continue
		}

		// check that we got the error we expected
		if len(v.expectedErr) > 0 {
			if !strings.Contains(errs[0].Error(), v.expectedErr) {
				t.Errorf("%s expected error '%s' but received %v", k, v.expectedErr, errs[0])
			}
		}
	}
}

func TestPolicyAuthorization(t *testing.T) {
	policyWithName := func(name string) *policy.PodSecurityPolicy {
		p := permissivePSP()
		p.Name = name
		return p
	}

	tests := map[string]struct {
		user           user.Info
		sa             string
		ns             string
		expectedPolicy string
		inPolicies     []*policy.PodSecurityPolicy
		allowed        map[string]map[string]map[string]bool
		allowedGroup   string
	}{
		"policy allowed by user (extensions API Group)": {
			user: &user.DefaultInfo{Name: "user"},
			sa:   "sa",
			ns:   "test",
			allowed: map[string]map[string]map[string]bool{
				"user": {
					"test": {"policy": true},
				},
			},
			inPolicies:     []*policy.PodSecurityPolicy{policyWithName("policy")},
			expectedPolicy: "policy",
		},
		"policy allowed by sa (extensions API Group)": {
			user: &user.DefaultInfo{Name: "user"},
			sa:   "sa",
			ns:   "test",
			allowed: map[string]map[string]map[string]bool{
				serviceaccount.MakeUsername("test", "sa"): {
					"test": {"policy": true},
				},
			},
			inPolicies:     []*policy.PodSecurityPolicy{policyWithName("policy")},
			expectedPolicy: "policy",
		},
		"policy allowed by user (policy API Group)": {
			user: &user.DefaultInfo{Name: "user"},
			sa:   "sa",
			ns:   "test",
			allowed: map[string]map[string]map[string]bool{
				"user": {
					"test": {"policy": true},
				},
			},
			inPolicies:     []*policy.PodSecurityPolicy{policyWithName("policy")},
			expectedPolicy: "policy",
			allowedGroup:   policy.GroupName,
		},
		"policy allowed by sa (policy API Group)": {
			user: &user.DefaultInfo{Name: "user"},
			sa:   "sa",
			ns:   "test",
			allowed: map[string]map[string]map[string]bool{
				serviceaccount.MakeUsername("test", "sa"): {
					"test": {"policy": true},
				},
			},
			inPolicies:     []*policy.PodSecurityPolicy{policyWithName("policy")},
			expectedPolicy: "policy",
			allowedGroup:   policy.GroupName,
		},
		"no policies allowed": {
			user:           &user.DefaultInfo{Name: "user"},
			sa:             "sa",
			ns:             "test",
			allowed:        map[string]map[string]map[string]bool{},
			inPolicies:     []*policy.PodSecurityPolicy{policyWithName("policy")},
			expectedPolicy: "",
		},
		"multiple policies allowed": {
			user: &user.DefaultInfo{Name: "user"},
			sa:   "sa",
			ns:   "test",
			allowed: map[string]map[string]map[string]bool{
				serviceaccount.MakeUsername("test", "sa"): {
					"test":  {"policy1": true},
					"":      {"policy4": true},
					"other": {"policy6": true},
				},
				"user": {
					"test":  {"policy2": true},
					"":      {"policy5": true},
					"other": {"policy7": true},
				},
			},
			inPolicies: []*policy.PodSecurityPolicy{
				// Prefix to force checking these policies first.
				policyWithName("a_policy1"), // not allowed in this namespace
				policyWithName("a_policy2"), // not allowed in this namespace
				policyWithName("policy2"),   // allowed by sa
				policyWithName("policy3"),   // allowed by user
				policyWithName("policy4"),   // not allowed
				policyWithName("policy5"),   // allowed by sa at cluster level
				policyWithName("policy6"),   // allowed by user at cluster level
			},
			expectedPolicy: "policy2",
		},
		"policies are not allowed for nil user info": {
			user: nil,
			sa:   "sa",
			ns:   "test",
			allowed: map[string]map[string]map[string]bool{
				serviceaccount.MakeUsername("test", "sa"): {
					"test": {"policy1": true},
				},
				"user": {
					"test": {"policy2": true},
				},
			},
			inPolicies: []*policy.PodSecurityPolicy{
				policyWithName("policy1"),
				policyWithName("policy2"),
				policyWithName("policy3"),
			},
			// only the policies for the sa are allowed when user info is nil
			expectedPolicy: "policy1",
		},
		"policies are not allowed for nil sa info": {
			user: &user.DefaultInfo{Name: "user"},
			sa:   "",
			ns:   "test",
			allowed: map[string]map[string]map[string]bool{
				serviceaccount.MakeUsername("test", "sa"): {
					"test": {"policy1": true},
				},
				"user": {
					"test": {"policy2": true},
				},
			},
			inPolicies: []*policy.PodSecurityPolicy{
				policyWithName("policy1"),
				policyWithName("policy2"),
				policyWithName("policy3"),
			},
			// only the policies for the user are allowed when sa info is nil
			expectedPolicy: "policy2",
		},
		"policies are not allowed for nil sa and user info": {
			user: nil,
			sa:   "",
			ns:   "test",
			allowed: map[string]map[string]map[string]bool{
				serviceaccount.MakeUsername("test", "sa"): {
					"test": {"policy1": true},
				},
				"user": {
					"test": {"policy2": true},
				},
			},
			inPolicies: []*policy.PodSecurityPolicy{
				policyWithName("policy1"),
				policyWithName("policy2"),
				policyWithName("policy3"),
			},
			// no policies are allowed if sa and user are both nil
			expectedPolicy: "",
		},
	}
	for k, v := range tests {
		var (
			oldPod     *kapi.Pod
			shouldPass = v.expectedPolicy != ""
			authz      = &TestAuthorizer{usernameToNamespaceToAllowedPSPs: v.allowed, allowedAPIGroupName: v.allowedGroup}
			canMutate  = true
		)
		pod := goodPod()
		pod.Namespace = v.ns
		pod.Spec.ServiceAccountName = v.sa
		testPSPAdmitAdvanced(k, kadmission.Create, v.inPolicies, authz, v.user,
			pod, oldPod, shouldPass, shouldPass, canMutate, v.expectedPolicy, t)
	}
}

func TestPolicyAuthorizationErrors(t *testing.T) {
	policyWithName := func(name string) *policy.PodSecurityPolicy {
		p := restrictivePSP()
		p.Name = name
		return p
	}

	const (
		sa       = "sa"
		ns       = "test"
		userName = "user"
	)

	tests := map[string]struct {
		inPolicies           []*policy.PodSecurityPolicy
		allowed              map[string]map[string]map[string]bool
		expectValidationErrs int
	}{
		"policies not allowed": {
			allowed: map[string]map[string]map[string]bool{},
			inPolicies: []*policy.PodSecurityPolicy{
				policyWithName("policy1"),
				policyWithName("policy2"),
			},
			expectValidationErrs: 0,
		},
		"policy allowed by user": {
			allowed: map[string]map[string]map[string]bool{
				"user": {
					"test": {"policy1": true},
				},
			},
			inPolicies: []*policy.PodSecurityPolicy{
				policyWithName("policy1"),
				policyWithName("policy2"),
			},
			expectValidationErrs: 1,
		},
		"policy allowed by service account": {
			allowed: map[string]map[string]map[string]bool{
				serviceaccount.MakeUsername("test", "sa"): {
					"test": {"policy2": true},
				},
			},
			inPolicies: []*policy.PodSecurityPolicy{
				policyWithName("policy1"),
				policyWithName("policy2"),
			},
			expectValidationErrs: 1,
		},
		"multiple policies allowed": {
			allowed: map[string]map[string]map[string]bool{
				"user": {
					"test": {"policy1": true},
				},
				serviceaccount.MakeUsername("test", "sa"): {
					"test": {"policy2": true},
				},
			},
			inPolicies: []*policy.PodSecurityPolicy{
				policyWithName("policy1"),
				policyWithName("policy2"),
			},
			expectValidationErrs: 2,
		},
	}
	for desc, tc := range tests {
		t.Run(desc, func(t *testing.T) {
			authz := &TestAuthorizer{usernameToNamespaceToAllowedPSPs: tc.allowed}
			pod := goodPod()
			pod.Namespace = ns
			pod.Spec.ServiceAccountName = sa
			pod.Spec.SecurityContext.HostPID = true

			plugin := NewTestAdmission(tc.inPolicies, authz)
			attrs := kadmission.NewAttributesRecord(pod, nil, kapi.Kind("Pod").WithVersion("version"), ns, "", kapi.Resource("pods").WithVersion("version"), "", kadmission.Create, &metav1.CreateOptions{}, false, &user.DefaultInfo{Name: userName})

			allowedPod, _, validationErrs, err := plugin.computeSecurityContext(context.Background(), attrs, pod, true, "")
			assert.Nil(t, allowedPod)
			assert.NoError(t, err)
			assert.Len(t, validationErrs, tc.expectValidationErrs)
		})
	}
}

func TestPreferValidatedPSP(t *testing.T) {
	restrictivePSPWithName := func(name string) *policy.PodSecurityPolicy {
		p := restrictivePSP()
		p.Name = name
		return p
	}

	permissivePSPWithName := func(name string) *policy.PodSecurityPolicy {
		p := permissivePSP()
		p.Name = name
		return p
	}

	tests := map[string]struct {
		inPolicies           []*policy.PodSecurityPolicy
		expectValidationErrs int
		validatedPSPHint     string
		expectedPSP          string
	}{
		"no policy saved in annotations, PSPs are ordered lexicographically": {
			inPolicies: []*policy.PodSecurityPolicy{
				restrictivePSPWithName("001restrictive"),
				restrictivePSPWithName("002restrictive"),
				permissivePSPWithName("002permissive"),
				permissivePSPWithName("001permissive"),
				permissivePSPWithName("003permissive"),
			},
			expectValidationErrs: 0,
			validatedPSPHint:     "",
			expectedPSP:          "001permissive",
		},
		"policy saved in annotations is preferred": {
			inPolicies: []*policy.PodSecurityPolicy{
				restrictivePSPWithName("001restrictive"),
				restrictivePSPWithName("002restrictive"),
				permissivePSPWithName("001permissive"),
				permissivePSPWithName("002permissive"),
				permissivePSPWithName("003permissive"),
			},
			expectValidationErrs: 0,
			validatedPSPHint:     "002permissive",
			expectedPSP:          "002permissive",
		},
		"policy saved in annotations is invalid": {
			inPolicies: []*policy.PodSecurityPolicy{
				restrictivePSPWithName("001restrictive"),
				restrictivePSPWithName("002restrictive"),
			},
			expectValidationErrs: 2,
			validatedPSPHint:     "foo",
			expectedPSP:          "",
		},
		"policy saved in annotations is disallowed anymore": {
			inPolicies: []*policy.PodSecurityPolicy{
				restrictivePSPWithName("001restrictive"),
				restrictivePSPWithName("002restrictive"),
			},
			expectValidationErrs: 2,
			validatedPSPHint:     "001restrictive",
			expectedPSP:          "",
		},
		"policy saved in annotations is disallowed anymore, but find another one": {
			inPolicies: []*policy.PodSecurityPolicy{
				restrictivePSPWithName("001restrictive"),
				restrictivePSPWithName("002restrictive"),
				permissivePSPWithName("002permissive"),
				permissivePSPWithName("001permissive"),
			},
			expectValidationErrs: 0,
			validatedPSPHint:     "001restrictive",
			expectedPSP:          "001permissive",
		},
	}
	for desc, tc := range tests {
		t.Run(desc, func(t *testing.T) {
			authz := authorizerfactory.NewAlwaysAllowAuthorizer()
			allowPrivilegeEscalation := true
			pod := goodPod()
			pod.Namespace = "ns"
			pod.Spec.ServiceAccountName = "sa"
			pod.Spec.Containers[0].SecurityContext.AllowPrivilegeEscalation = &allowPrivilegeEscalation

			plugin := NewTestAdmission(tc.inPolicies, authz)
			attrs := kadmission.NewAttributesRecord(pod, nil, kapi.Kind("Pod").WithVersion("version"), "ns", "", kapi.Resource("pods").WithVersion("version"), "", kadmission.Update, &metav1.UpdateOptions{}, false, &user.DefaultInfo{Name: "test"})

			_, pspName, validationErrs, err := plugin.computeSecurityContext(context.Background(), attrs, pod, false, tc.validatedPSPHint)
			assert.NoError(t, err)
			assert.Len(t, validationErrs, tc.expectValidationErrs)
			assert.Equal(t, tc.expectedPSP, pspName)
		})
	}
}

func restrictivePSP() *policy.PodSecurityPolicy {
	allowPrivilegeEscalation := false
	return &policy.PodSecurityPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "restrictive",
			Annotations: map[string]string{},
		},
		Spec: policy.PodSecurityPolicySpec{
			AllowPrivilegeEscalation: &allowPrivilegeEscalation,
			RunAsUser: policy.RunAsUserStrategyOptions{
				Rule: policy.RunAsUserStrategyMustRunAs,
				Ranges: []policy.IDRange{
					{Min: int64(999), Max: int64(999)},
				},
			},
			RunAsGroup: &policy.RunAsGroupStrategyOptions{
				Rule: policy.RunAsGroupStrategyMustRunAs,
				Ranges: []policy.IDRange{
					{Min: int64(999), Max: int64(999)},
				},
			},
			SELinux: policy.SELinuxStrategyOptions{
				Rule: policy.SELinuxStrategyMustRunAs,
				SELinuxOptions: &v1.SELinuxOptions{
					Level: "s9:z0,z1",
				},
			},
			FSGroup: policy.FSGroupStrategyOptions{
				Rule: policy.FSGroupStrategyMustRunAs,
				Ranges: []policy.IDRange{
					{Min: int64(999), Max: int64(999)},
				},
			},
			SupplementalGroups: policy.SupplementalGroupsStrategyOptions{
				Rule: policy.SupplementalGroupsStrategyMustRunAs,
				Ranges: []policy.IDRange{
					{Min: int64(999), Max: int64(999)},
				},
			},
		},
	}
}

func permissivePSP() *policy.PodSecurityPolicy {
	allowPrivilegeEscalation := true
	return &policy.PodSecurityPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "privileged",
			Annotations: map[string]string{},
		},
		Spec: policy.PodSecurityPolicySpec{
			AllowPrivilegeEscalation: &allowPrivilegeEscalation,
			HostIPC:                  true,
			HostNetwork:              true,
			HostPID:                  true,
			HostPorts:                []policy.HostPortRange{{Min: 0, Max: 65536}},
			Volumes:                  []policy.FSType{policy.All},
			AllowedCapabilities:      []v1.Capability{policy.AllowAllCapabilities},
			RunAsUser: policy.RunAsUserStrategyOptions{
				Rule: policy.RunAsUserStrategyRunAsAny,
			},
			RunAsGroup: &policy.RunAsGroupStrategyOptions{
				Rule: policy.RunAsGroupStrategyRunAsAny,
			},
			SELinux: policy.SELinuxStrategyOptions{
				Rule: policy.SELinuxStrategyRunAsAny,
			},
			FSGroup: policy.FSGroupStrategyOptions{
				Rule: policy.FSGroupStrategyRunAsAny,
			},
			SupplementalGroups: policy.SupplementalGroupsStrategyOptions{
				Rule: policy.SupplementalGroupsStrategyRunAsAny,
			},
		},
	}
}

// goodPod is empty and should not be used directly for testing since we're providing
// two different PSPs.  Since no values are specified it would be allowed to match any
// psp when defaults are filled in.
func goodPod() *kapi.Pod {
	return &kapi.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "pod",
			Namespace:   "namespace",
			Annotations: map[string]string{},
		},
		Spec: kapi.PodSpec{
			ServiceAccountName: "default",
			SecurityContext:    &kapi.PodSecurityContext{},
			Containers: []kapi.Container{
				{
					Name:            defaultContainerName,
					SecurityContext: &kapi.SecurityContext{},
				},
			},
		},
	}
}
