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

package admission

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/sets"
	kadmission "k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	kapi "k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	extensionslisters "k8s.io/kubernetes/pkg/client/listers/extensions/internalversion"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/security/apparmor"
	kpsp "k8s.io/kubernetes/pkg/security/podsecuritypolicy"
	"k8s.io/kubernetes/pkg/security/podsecuritypolicy/seccomp"
	psputil "k8s.io/kubernetes/pkg/security/podsecuritypolicy/util"
)

const defaultContainerName = "test-c"

// NewTestAdmission provides an admission plugin with test implementations of internal structs.  It uses
// an authorizer that always returns true.
func NewTestAdmission(lister extensionslisters.PodSecurityPolicyLister) kadmission.Interface {
	return &podSecurityPolicyPlugin{
		Handler:         kadmission.NewHandler(kadmission.Create),
		strategyFactory: kpsp.NewSimpleStrategyFactory(),
		pspMatcher:      getMatchingPolicies,
		authz:           &TestAuthorizer{},
		lister:          lister,
	}
}

// TestAlwaysAllowedAuthorizer is a testing struct for testing that fulfills the authorizer interface.
type TestAuthorizer struct {
	// disallowed contains names of disallowed policies.  Map is keyed by user.Info.GetName()
	disallowed map[string][]string
}

func (t *TestAuthorizer) Authorize(a authorizer.Attributes) (authorized bool, reason string, err error) {
	disallowedForUser, _ := t.disallowed[a.GetUser().GetName()]
	for _, name := range disallowedForUser {
		if a.GetName() == name {
			return false, "", nil
		}
	}
	return true, "", nil
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
		pspAnnotations map[string]string
		podAnnotations map[string]string
		shouldAdmit    bool
	}{
		"no seccomp, no pod annotations": {
			pspAnnotations: nil,
			podAnnotations: nil,
			shouldAdmit:    true,
		},
		"no seccomp, pod annotations": {
			pspAnnotations: nil,
			podAnnotations: map[string]string{
				kapi.SeccompPodAnnotationKey: "foo",
			},
			shouldAdmit: false,
		},
		"no seccomp, container annotations": {
			pspAnnotations: nil,
			podAnnotations: map[string]string{
				kapi.SeccompContainerAnnotationKeyPrefix + containerName: "foo",
			},
			shouldAdmit: false,
		},
		"seccomp, allow any no pod annotation": {
			pspAnnotations: map[string]string{
				seccomp.AllowedProfilesAnnotationKey: seccomp.AllowAny,
			},
			podAnnotations: nil,
			shouldAdmit:    true,
		},
		"seccomp, allow any pod annotation": {
			pspAnnotations: map[string]string{
				seccomp.AllowedProfilesAnnotationKey: seccomp.AllowAny,
			},
			podAnnotations: map[string]string{
				kapi.SeccompPodAnnotationKey: "foo",
			},
			shouldAdmit: true,
		},
		"seccomp, allow any container annotation": {
			pspAnnotations: map[string]string{
				seccomp.AllowedProfilesAnnotationKey: seccomp.AllowAny,
			},
			podAnnotations: map[string]string{
				kapi.SeccompContainerAnnotationKeyPrefix + containerName: "foo",
			},
			shouldAdmit: true,
		},
		"seccomp, allow specific pod annotation failure": {
			pspAnnotations: map[string]string{
				seccomp.AllowedProfilesAnnotationKey: "foo",
			},
			podAnnotations: map[string]string{
				kapi.SeccompPodAnnotationKey: "bar",
			},
			shouldAdmit: false,
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
			shouldAdmit: false,
		},
		"seccomp, allow specific pod annotation pass": {
			pspAnnotations: map[string]string{
				seccomp.AllowedProfilesAnnotationKey: "foo",
			},
			podAnnotations: map[string]string{
				kapi.SeccompPodAnnotationKey: "foo",
			},
			shouldAdmit: true,
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
			shouldAdmit: true,
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
		testPSPAdmit(k, []*extensions.PodSecurityPolicy{psp}, pod, v.shouldAdmit, psp.Name, t)
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

	tests := map[string]struct {
		pod          *kapi.Pod
		psps         []*extensions.PodSecurityPolicy
		shouldPass   bool
		expectedPriv bool
		expectedPSP  string
	}{
		"pod without priv request allowed under non priv PSP": {
			pod:          goodPod(),
			psps:         []*extensions.PodSecurityPolicy{nonPrivilegedPSP},
			shouldPass:   true,
			expectedPriv: false,
			expectedPSP:  nonPrivilegedPSP.Name,
		},
		"pod without priv request allowed under priv PSP": {
			pod:          goodPod(),
			psps:         []*extensions.PodSecurityPolicy{privilegedPSP},
			shouldPass:   true,
			expectedPriv: false,
			expectedPSP:  privilegedPSP.Name,
		},
		"pod with priv request denied by non priv PSP": {
			pod:        createPodWithPriv(true),
			psps:       []*extensions.PodSecurityPolicy{nonPrivilegedPSP},
			shouldPass: false,
		},
		"pod with priv request allowed by priv PSP": {
			pod:          createPodWithPriv(true),
			psps:         []*extensions.PodSecurityPolicy{nonPrivilegedPSP, privilegedPSP},
			shouldPass:   true,
			expectedPriv: true,
			expectedPSP:  privilegedPSP.Name,
		},
	}

	for k, v := range tests {
		testPSPAdmit(k, v.psps, v.pod, v.shouldPass, v.expectedPSP, t)

		if v.shouldPass {
			if v.pod.Spec.Containers[0].SecurityContext.Privileged == nil ||
				*v.pod.Spec.Containers[0].SecurityContext.Privileged != v.expectedPriv {
				t.Errorf("%s expected privileged to be %t", k, v.expectedPriv)
			}
		}
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
	allowsFooInAllowed.Spec.AllowedCapabilities = []kapi.Capability{"foo"}

	allowsFooInRequired := restrictivePSP()
	allowsFooInRequired.Name = "allowCapInRequired"
	allowsFooInRequired.Spec.DefaultAddCapabilities = []kapi.Capability{"foo"}

	requiresFooToBeDropped := restrictivePSP()
	requiresFooToBeDropped.Name = "requireDrop"
	requiresFooToBeDropped.Spec.RequiredDropCapabilities = []kapi.Capability{"foo"}

	tc := map[string]struct {
		pod                  *kapi.Pod
		psps                 []*extensions.PodSecurityPolicy
		shouldPass           bool
		expectedCapabilities *kapi.Capabilities
		expectedPSP          string
	}{
		// UC 1: if a PSP does not define allowed or required caps then a pod requesting a cap
		// should be rejected.
		"should reject cap add when not allowed or required": {
			pod:        createPodWithCaps(&kapi.Capabilities{Add: []kapi.Capability{"foo"}}),
			psps:       []*extensions.PodSecurityPolicy{restricted},
			shouldPass: false,
		},
		// UC 2: if a PSP allows a cap in the allowed field it should accept the pod request
		// to add the cap.
		"should accept cap add when in allowed": {
			pod:         createPodWithCaps(&kapi.Capabilities{Add: []kapi.Capability{"foo"}}),
			psps:        []*extensions.PodSecurityPolicy{restricted, allowsFooInAllowed},
			shouldPass:  true,
			expectedPSP: allowsFooInAllowed.Name,
		},
		// UC 3: if a PSP requires a cap then it should accept the pod request
		// to add the cap.
		"should accept cap add when in required": {
			pod:         createPodWithCaps(&kapi.Capabilities{Add: []kapi.Capability{"foo"}}),
			psps:        []*extensions.PodSecurityPolicy{restricted, allowsFooInRequired},
			shouldPass:  true,
			expectedPSP: allowsFooInRequired.Name,
		},
		// UC 4: if a PSP requires a cap to be dropped then it should fail both
		// in the verification of adds and verification of drops
		"should reject cap add when requested cap is required to be dropped": {
			pod:        createPodWithCaps(&kapi.Capabilities{Add: []kapi.Capability{"foo"}}),
			psps:       []*extensions.PodSecurityPolicy{restricted, requiresFooToBeDropped},
			shouldPass: false,
		},
		// UC 5: if a PSP requires a cap to be dropped it should accept
		// a manual request to drop the cap.
		"should accept cap drop when cap is required to be dropped": {
			pod:         createPodWithCaps(&kapi.Capabilities{Drop: []kapi.Capability{"foo"}}),
			psps:        []*extensions.PodSecurityPolicy{requiresFooToBeDropped},
			shouldPass:  true,
			expectedPSP: requiresFooToBeDropped.Name,
		},
		// UC 6: required add is defaulted
		"required add is defaulted": {
			pod:        goodPod(),
			psps:       []*extensions.PodSecurityPolicy{allowsFooInRequired},
			shouldPass: true,
			expectedCapabilities: &kapi.Capabilities{
				Add: []kapi.Capability{"foo"},
			},
			expectedPSP: allowsFooInRequired.Name,
		},
		// UC 7: required drop is defaulted
		"required drop is defaulted": {
			pod:        goodPod(),
			psps:       []*extensions.PodSecurityPolicy{requiresFooToBeDropped},
			shouldPass: true,
			expectedCapabilities: &kapi.Capabilities{
				Drop: []kapi.Capability{"foo"},
			},
			expectedPSP: requiresFooToBeDropped.Name,
		},
	}

	for k, v := range tc {
		testPSPAdmit(k, v.psps, v.pod, v.shouldPass, v.expectedPSP, t)

		if v.expectedCapabilities != nil {
			if !reflect.DeepEqual(v.expectedCapabilities, v.pod.Spec.Containers[0].SecurityContext.Capabilities) {
				t.Errorf("%s resulted in caps that were not expected - expected: %v, received: %v", k, v.expectedCapabilities, v.pod.Spec.Containers[0].SecurityContext.Capabilities)
			}
		}
	}

	for k, v := range tc {
		useInitContainers(v.pod)
		testPSPAdmit(k, v.psps, v.pod, v.shouldPass, v.expectedPSP, t)

		if v.expectedCapabilities != nil {
			if !reflect.DeepEqual(v.expectedCapabilities, v.pod.Spec.InitContainers[0].SecurityContext.Capabilities) {
				t.Errorf("%s resulted in caps that were not expected - expected: %v, received: %v", k, v.expectedCapabilities, v.pod.Spec.InitContainers[0].SecurityContext.Capabilities)
			}
		}
	}

}

func TestAdmitVolumes(t *testing.T) {
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
		testPSPAdmit(fmt.Sprintf("%s denial", string(fsType)), []*extensions.PodSecurityPolicy{psp}, pod, false, "", t)

		// also expect a denial for this PSP if it's an init container
		useInitContainers(pod)
		testPSPAdmit(fmt.Sprintf("%s denial", string(fsType)), []*extensions.PodSecurityPolicy{psp}, pod, false, "", t)

		// now add the fstype directly to the psp and it should validate
		psp.Spec.Volumes = []extensions.FSType{fsType}
		testPSPAdmit(fmt.Sprintf("%s direct accept", string(fsType)), []*extensions.PodSecurityPolicy{psp}, pod, true, psp.Name, t)

		// now change the psp to allow any volumes and the pod should still validate
		psp.Spec.Volumes = []extensions.FSType{extensions.All}
		testPSPAdmit(fmt.Sprintf("%s wildcard accept", string(fsType)), []*extensions.PodSecurityPolicy{psp}, pod, true, psp.Name, t)
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
		psps                []*extensions.PodSecurityPolicy
		shouldPass          bool
		expectedHostNetwork bool
		expectedPSP         string
	}{
		"pod without hostnetwork request allowed under noHostNetwork PSP": {
			pod:                 goodPod(),
			psps:                []*extensions.PodSecurityPolicy{noHostNetwork},
			shouldPass:          true,
			expectedHostNetwork: false,
			expectedPSP:         noHostNetwork.Name,
		},
		"pod without hostnetwork request allowed under hostNetwork PSP": {
			pod:                 goodPod(),
			psps:                []*extensions.PodSecurityPolicy{hostNetwork},
			shouldPass:          true,
			expectedHostNetwork: false,
			expectedPSP:         hostNetwork.Name,
		},
		"pod with hostnetwork request denied by noHostNetwork PSP": {
			pod:        createPodWithHostNetwork(true),
			psps:       []*extensions.PodSecurityPolicy{noHostNetwork},
			shouldPass: false,
		},
		"pod with hostnetwork request allowed by hostNetwork PSP": {
			pod:                 createPodWithHostNetwork(true),
			psps:                []*extensions.PodSecurityPolicy{noHostNetwork, hostNetwork},
			shouldPass:          true,
			expectedHostNetwork: true,
			expectedPSP:         hostNetwork.Name,
		},
	}

	for k, v := range tests {
		testPSPAdmit(k, v.psps, v.pod, v.shouldPass, v.expectedPSP, t)

		if v.shouldPass {
			if v.pod.Spec.SecurityContext.HostNetwork != v.expectedHostNetwork {
				t.Errorf("%s expected hostNetwork to be %t", k, v.expectedHostNetwork)
			}
		}
	}

	// test again with init containers
	for k, v := range tests {
		useInitContainers(v.pod)
		testPSPAdmit(k, v.psps, v.pod, v.shouldPass, v.expectedPSP, t)

		if v.shouldPass {
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
	hostPorts.Spec.HostPorts = []extensions.HostPortRange{
		{Min: 1, Max: 10},
	}

	tests := map[string]struct {
		pod         *kapi.Pod
		psps        []*extensions.PodSecurityPolicy
		shouldPass  bool
		expectedPSP string
	}{
		"host port out of range": {
			pod:        createPodWithHostPorts(11),
			psps:       []*extensions.PodSecurityPolicy{hostPorts},
			shouldPass: false,
		},
		"host port in range": {
			pod:         createPodWithHostPorts(5),
			psps:        []*extensions.PodSecurityPolicy{hostPorts},
			shouldPass:  true,
			expectedPSP: hostPorts.Name,
		},
		"no host ports with range": {
			pod:         goodPod(),
			psps:        []*extensions.PodSecurityPolicy{hostPorts},
			shouldPass:  true,
			expectedPSP: hostPorts.Name,
		},
		"no host ports without range": {
			pod:         goodPod(),
			psps:        []*extensions.PodSecurityPolicy{noHostPorts},
			shouldPass:  true,
			expectedPSP: noHostPorts.Name,
		},
		"host ports without range": {
			pod:        createPodWithHostPorts(5),
			psps:       []*extensions.PodSecurityPolicy{noHostPorts},
			shouldPass: false,
		},
	}

	for i := 0; i < 2; i++ {
		for k, v := range tests {
			v.pod.Spec.Containers, v.pod.Spec.InitContainers = v.pod.Spec.InitContainers, v.pod.Spec.Containers
			testPSPAdmit(k, v.psps, v.pod, v.shouldPass, v.expectedPSP, t)
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
		pod             *kapi.Pod
		psps            []*extensions.PodSecurityPolicy
		shouldPass      bool
		expectedHostPID bool
		expectedPSP     string
	}{
		"pod without hostpid request allowed under noHostPID PSP": {
			pod:             goodPod(),
			psps:            []*extensions.PodSecurityPolicy{noHostPID},
			shouldPass:      true,
			expectedHostPID: false,
			expectedPSP:     noHostPID.Name,
		},
		"pod without hostpid request allowed under hostPID PSP": {
			pod:             goodPod(),
			psps:            []*extensions.PodSecurityPolicy{hostPID},
			shouldPass:      true,
			expectedHostPID: false,
			expectedPSP:     hostPID.Name,
		},
		"pod with hostpid request denied by noHostPID PSP": {
			pod:        createPodWithHostPID(true),
			psps:       []*extensions.PodSecurityPolicy{noHostPID},
			shouldPass: false,
		},
		"pod with hostpid request allowed by hostPID PSP": {
			pod:             createPodWithHostPID(true),
			psps:            []*extensions.PodSecurityPolicy{noHostPID, hostPID},
			shouldPass:      true,
			expectedHostPID: true,
			expectedPSP:     hostPID.Name,
		},
	}

	for k, v := range tests {
		testPSPAdmit(k, v.psps, v.pod, v.shouldPass, v.expectedPSP, t)

		if v.shouldPass {
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
		pod             *kapi.Pod
		psps            []*extensions.PodSecurityPolicy
		shouldPass      bool
		expectedHostIPC bool
		expectedPSP     string
	}{
		"pod without hostIPC request allowed under noHostIPC PSP": {
			pod:             goodPod(),
			psps:            []*extensions.PodSecurityPolicy{noHostIPC},
			shouldPass:      true,
			expectedHostIPC: false,
			expectedPSP:     noHostIPC.Name,
		},
		"pod without hostIPC request allowed under hostIPC PSP": {
			pod:             goodPod(),
			psps:            []*extensions.PodSecurityPolicy{hostIPC},
			shouldPass:      true,
			expectedHostIPC: false,
			expectedPSP:     hostIPC.Name,
		},
		"pod with hostIPC request denied by noHostIPC PSP": {
			pod:        createPodWithHostIPC(true),
			psps:       []*extensions.PodSecurityPolicy{noHostIPC},
			shouldPass: false,
		},
		"pod with hostIPC request allowed by hostIPC PSP": {
			pod:             createPodWithHostIPC(true),
			psps:            []*extensions.PodSecurityPolicy{noHostIPC, hostIPC},
			shouldPass:      true,
			expectedHostIPC: true,
			expectedPSP:     hostIPC.Name,
		},
	}

	for k, v := range tests {
		testPSPAdmit(k, v.psps, v.pod, v.shouldPass, v.expectedPSP, t)

		if v.shouldPass {
			if v.pod.Spec.SecurityContext.HostIPC != v.expectedHostIPC {
				t.Errorf("%s expected hostIPC to be %t", k, v.expectedHostIPC)
			}
		}
	}
}

func TestAdmitSELinux(t *testing.T) {
	createPodWithSELinux := func(opts *kapi.SELinuxOptions) *kapi.Pod {
		pod := goodPod()
		// doesn't matter if we set it here or on the container, the
		// admission controller uses DetermineEffectiveSC to get the defaulting
		// behavior so it can validate what will be applied at runtime
		pod.Spec.SecurityContext.SELinuxOptions = opts
		return pod
	}

	runAsAny := restrictivePSP()
	runAsAny.Name = "runAsAny"
	runAsAny.Spec.SELinux.Rule = extensions.SELinuxStrategyRunAsAny

	mustRunAs := restrictivePSP()
	mustRunAs.Name = "mustRunAs"
	mustRunAs.Spec.SELinux.SELinuxOptions.Level = "level"
	mustRunAs.Spec.SELinux.SELinuxOptions.Role = "role"
	mustRunAs.Spec.SELinux.SELinuxOptions.Type = "type"
	mustRunAs.Spec.SELinux.SELinuxOptions.User = "user"

	tests := map[string]struct {
		pod             *kapi.Pod
		psps            []*extensions.PodSecurityPolicy
		shouldPass      bool
		expectedSELinux *kapi.SELinuxOptions
		expectedPSP     string
	}{
		"runAsAny with no pod request": {
			pod:             goodPod(),
			psps:            []*extensions.PodSecurityPolicy{runAsAny},
			shouldPass:      true,
			expectedSELinux: nil,
			expectedPSP:     runAsAny.Name,
		},
		"runAsAny with pod request": {
			pod:             createPodWithSELinux(&kapi.SELinuxOptions{User: "foo"}),
			psps:            []*extensions.PodSecurityPolicy{runAsAny},
			shouldPass:      true,
			expectedSELinux: &kapi.SELinuxOptions{User: "foo"},
			expectedPSP:     runAsAny.Name,
		},
		"mustRunAs with bad pod request": {
			pod:        createPodWithSELinux(&kapi.SELinuxOptions{User: "foo"}),
			psps:       []*extensions.PodSecurityPolicy{mustRunAs},
			shouldPass: false,
		},
		"mustRunAs with no pod request": {
			pod:             goodPod(),
			psps:            []*extensions.PodSecurityPolicy{mustRunAs},
			shouldPass:      true,
			expectedSELinux: mustRunAs.Spec.SELinux.SELinuxOptions,
			expectedPSP:     mustRunAs.Name,
		},
		"mustRunAs with good pod request": {
			pod:             createPodWithSELinux(&kapi.SELinuxOptions{Level: "level", Role: "role", Type: "type", User: "user"}),
			psps:            []*extensions.PodSecurityPolicy{mustRunAs},
			shouldPass:      true,
			expectedSELinux: mustRunAs.Spec.SELinux.SELinuxOptions,
			expectedPSP:     mustRunAs.Name,
		},
	}

	for k, v := range tests {
		testPSPAdmit(k, v.psps, v.pod, v.shouldPass, v.expectedPSP, t)

		if v.shouldPass {
			if v.pod.Spec.Containers[0].SecurityContext.SELinuxOptions == nil && v.expectedSELinux == nil {
				// ok, don't need to worry about identifying specific diffs
				continue
			}
			if v.pod.Spec.Containers[0].SecurityContext.SELinuxOptions == nil && v.expectedSELinux != nil {
				t.Errorf("%s expected selinux to be: %v but found nil", k, v.expectedSELinux)
				continue
			}
			if v.pod.Spec.Containers[0].SecurityContext.SELinuxOptions != nil && v.expectedSELinux == nil {
				t.Errorf("%s expected selinux to be nil but found: %v", k, *v.pod.Spec.Containers[0].SecurityContext.SELinuxOptions)
				continue
			}
			if !reflect.DeepEqual(*v.expectedSELinux, *v.pod.Spec.Containers[0].SecurityContext.SELinuxOptions) {
				t.Errorf("%s expected selinux to be: %v but found %v", k, *v.expectedSELinux, *v.pod.Spec.Containers[0].SecurityContext.SELinuxOptions)
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
		apparmor.DefaultProfileAnnotationKey: apparmor.ProfileRuntimeDefault,
	}
	appArmorPSP := restrictivePSP()
	appArmorPSP.Annotations = map[string]string{
		apparmor.AllowedProfilesAnnotationKey: apparmor.ProfileRuntimeDefault,
	}
	appArmorDefaultPSP := restrictivePSP()
	appArmorDefaultPSP.Annotations = map[string]string{
		apparmor.DefaultProfileAnnotationKey:  apparmor.ProfileRuntimeDefault,
		apparmor.AllowedProfilesAnnotationKey: apparmor.ProfileRuntimeDefault + "," + apparmor.ProfileNamePrefix + "foo",
	}

	tests := map[string]struct {
		pod             *kapi.Pod
		psp             *extensions.PodSecurityPolicy
		shouldPass      bool
		expectedProfile string
	}{
		"unconstrained with no profile": {
			pod:             goodPod(),
			psp:             unconstrainedPSP,
			shouldPass:      true,
			expectedProfile: "",
		},
		"unconstrained with profile": {
			pod:             createPodWithAppArmor(apparmor.ProfileRuntimeDefault),
			psp:             unconstrainedPSP,
			shouldPass:      true,
			expectedProfile: apparmor.ProfileRuntimeDefault,
		},
		"unconstrained with default profile": {
			pod:             goodPod(),
			psp:             defaultedPSP,
			shouldPass:      true,
			expectedProfile: apparmor.ProfileRuntimeDefault,
		},
		"AppArmor enforced with no profile": {
			pod:        goodPod(),
			psp:        appArmorPSP,
			shouldPass: false,
		},
		"AppArmor enforced with default profile": {
			pod:             goodPod(),
			psp:             appArmorDefaultPSP,
			shouldPass:      true,
			expectedProfile: apparmor.ProfileRuntimeDefault,
		},
		"AppArmor enforced with good profile": {
			pod:             createPodWithAppArmor(apparmor.ProfileNamePrefix + "foo"),
			psp:             appArmorDefaultPSP,
			shouldPass:      true,
			expectedProfile: apparmor.ProfileNamePrefix + "foo",
		},
		"AppArmor enforced with local profile": {
			pod:        createPodWithAppArmor(apparmor.ProfileNamePrefix + "bar"),
			psp:        appArmorPSP,
			shouldPass: false,
		},
	}

	for k, v := range tests {
		testPSPAdmit(k, []*extensions.PodSecurityPolicy{v.psp}, v.pod, v.shouldPass, v.psp.Name, t)

		if v.shouldPass {
			assert.Equal(t, v.expectedProfile, apparmor.GetProfileNameFromPodAnnotations(v.pod.Annotations, defaultContainerName), k)
		}
	}
}

func TestAdmitRunAsUser(t *testing.T) {
	createPodWithRunAsUser := func(user int64) *kapi.Pod {
		pod := goodPod()
		// doesn't matter if we set it here or on the container, the
		// admission controller uses DetermineEffectiveSC to get the defaulting
		// behavior so it can validate what will be applied at runtime
		pod.Spec.SecurityContext.RunAsUser = &user
		return pod
	}

	runAsAny := restrictivePSP()
	runAsAny.Name = "runAsAny"
	runAsAny.Spec.RunAsUser.Rule = extensions.RunAsUserStrategyRunAsAny

	mustRunAs := restrictivePSP()
	mustRunAs.Name = "mustRunAs"

	runAsNonRoot := restrictivePSP()
	runAsNonRoot.Name = "runAsNonRoot"
	runAsNonRoot.Spec.RunAsUser.Rule = extensions.RunAsUserStrategyMustRunAsNonRoot

	tests := map[string]struct {
		pod               *kapi.Pod
		psps              []*extensions.PodSecurityPolicy
		shouldPass        bool
		expectedRunAsUser *int
		expectedPSP       string
	}{
		"runAsAny no pod request": {
			pod:               goodPod(),
			psps:              []*extensions.PodSecurityPolicy{runAsAny},
			shouldPass:        true,
			expectedRunAsUser: nil,
			expectedPSP:       runAsAny.Name,
		},
		"runAsAny pod request": {
			pod:               createPodWithRunAsUser(1),
			psps:              []*extensions.PodSecurityPolicy{runAsAny},
			shouldPass:        true,
			expectedRunAsUser: intPtr(1),
			expectedPSP:       runAsAny.Name,
		},
		"mustRunAs pod request out of range": {
			pod:        createPodWithRunAsUser(1),
			psps:       []*extensions.PodSecurityPolicy{mustRunAs},
			shouldPass: false,
		},
		"mustRunAs pod request in range": {
			pod:               createPodWithRunAsUser(999),
			psps:              []*extensions.PodSecurityPolicy{mustRunAs},
			shouldPass:        true,
			expectedRunAsUser: intPtr(int(mustRunAs.Spec.RunAsUser.Ranges[0].Min)),
			expectedPSP:       mustRunAs.Name,
		},
		"mustRunAs no pod request": {
			pod:               goodPod(),
			psps:              []*extensions.PodSecurityPolicy{mustRunAs},
			shouldPass:        true,
			expectedRunAsUser: intPtr(int(mustRunAs.Spec.RunAsUser.Ranges[0].Min)),
			expectedPSP:       mustRunAs.Name,
		},
		"runAsNonRoot no pod request": {
			pod:               goodPod(),
			psps:              []*extensions.PodSecurityPolicy{runAsNonRoot},
			shouldPass:        true,
			expectedRunAsUser: nil,
			expectedPSP:       runAsNonRoot.Name,
		},
		"runAsNonRoot pod request root": {
			pod:        createPodWithRunAsUser(0),
			psps:       []*extensions.PodSecurityPolicy{runAsNonRoot},
			shouldPass: false,
		},
		"runAsNonRoot pod request non-root": {
			pod:               createPodWithRunAsUser(1),
			psps:              []*extensions.PodSecurityPolicy{runAsNonRoot},
			shouldPass:        true,
			expectedRunAsUser: intPtr(1),
			expectedPSP:       runAsNonRoot.Name,
		},
	}

	for k, v := range tests {
		testPSPAdmit(k, v.psps, v.pod, v.shouldPass, v.expectedPSP, t)

		if v.shouldPass {
			if v.pod.Spec.Containers[0].SecurityContext.RunAsUser == nil && v.expectedRunAsUser == nil {
				// ok, don't need to worry about identifying specific diffs
				continue
			}
			if v.pod.Spec.Containers[0].SecurityContext.RunAsUser == nil && v.expectedRunAsUser != nil {
				t.Errorf("%s expected RunAsUser to be: %v but found nil", k, v.expectedRunAsUser)
				continue
			}
			if v.pod.Spec.Containers[0].SecurityContext.RunAsUser != nil && v.expectedRunAsUser == nil {
				t.Errorf("%s expected RunAsUser to be nil but found: %v", k, *v.pod.Spec.Containers[0].SecurityContext.RunAsUser)
				continue
			}
			if int64(*v.expectedRunAsUser) != *v.pod.Spec.Containers[0].SecurityContext.RunAsUser {
				t.Errorf("%s expected RunAsUser to be: %v but found %v", k, *v.expectedRunAsUser, *v.pod.Spec.Containers[0].SecurityContext.RunAsUser)
			}
		}
	}
}

func TestAdmitSupplementalGroups(t *testing.T) {
	createPodWithSupGroup := func(group int64) *kapi.Pod {
		pod := goodPod()
		// doesn't matter if we set it here or on the container, the
		// admission controller uses DetermineEffectiveSC to get the defaulting
		// behavior so it can validate what will be applied at runtime
		pod.Spec.SecurityContext.SupplementalGroups = []int64{group}
		return pod
	}

	runAsAny := restrictivePSP()
	runAsAny.Name = "runAsAny"
	runAsAny.Spec.SupplementalGroups.Rule = extensions.SupplementalGroupsStrategyRunAsAny

	mustRunAs := restrictivePSP()
	mustRunAs.Name = "mustRunAs"

	tests := map[string]struct {
		pod               *kapi.Pod
		psps              []*extensions.PodSecurityPolicy
		shouldPass        bool
		expectedSupGroups []int64
		expectedPSP       string
	}{
		"runAsAny no pod request": {
			pod:               goodPod(),
			psps:              []*extensions.PodSecurityPolicy{runAsAny},
			shouldPass:        true,
			expectedSupGroups: []int64{},
			expectedPSP:       runAsAny.Name,
		},
		"runAsAny pod request": {
			pod:               createPodWithSupGroup(1),
			psps:              []*extensions.PodSecurityPolicy{runAsAny},
			shouldPass:        true,
			expectedSupGroups: []int64{1},
			expectedPSP:       runAsAny.Name,
		},
		"mustRunAs no pod request": {
			pod:               goodPod(),
			psps:              []*extensions.PodSecurityPolicy{mustRunAs},
			shouldPass:        true,
			expectedSupGroups: []int64{mustRunAs.Spec.SupplementalGroups.Ranges[0].Min},
			expectedPSP:       mustRunAs.Name,
		},
		"mustRunAs bad pod request": {
			pod:        createPodWithSupGroup(1),
			psps:       []*extensions.PodSecurityPolicy{mustRunAs},
			shouldPass: false,
		},
		"mustRunAs good pod request": {
			pod:               createPodWithSupGroup(999),
			psps:              []*extensions.PodSecurityPolicy{mustRunAs},
			shouldPass:        true,
			expectedSupGroups: []int64{999},
			expectedPSP:       mustRunAs.Name,
		},
	}

	for k, v := range tests {
		testPSPAdmit(k, v.psps, v.pod, v.shouldPass, v.expectedPSP, t)

		if v.shouldPass {
			if v.pod.Spec.SecurityContext.SupplementalGroups == nil && v.expectedSupGroups != nil {
				t.Errorf("%s expected SupplementalGroups to be: %v but found nil", k, v.expectedSupGroups)
				continue
			}
			if v.pod.Spec.SecurityContext.SupplementalGroups != nil && v.expectedSupGroups == nil {
				t.Errorf("%s expected SupplementalGroups to be nil but found: %v", k, v.pod.Spec.SecurityContext.SupplementalGroups)
				continue
			}
			if !reflect.DeepEqual(v.expectedSupGroups, v.pod.Spec.SecurityContext.SupplementalGroups) {
				t.Errorf("%s expected SupplementalGroups to be: %v but found %v", k, v.expectedSupGroups, v.pod.Spec.SecurityContext.SupplementalGroups)
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
		pod.Spec.SecurityContext.FSGroup = &group
		return pod
	}

	runAsAny := restrictivePSP()
	runAsAny.Name = "runAsAny"
	runAsAny.Spec.FSGroup.Rule = extensions.FSGroupStrategyRunAsAny

	mustRunAs := restrictivePSP()
	mustRunAs.Name = "mustRunAs"

	tests := map[string]struct {
		pod             *kapi.Pod
		psps            []*extensions.PodSecurityPolicy
		shouldPass      bool
		expectedFSGroup *int64
		expectedPSP     string
	}{
		"runAsAny no pod request": {
			pod:             goodPod(),
			psps:            []*extensions.PodSecurityPolicy{runAsAny},
			shouldPass:      true,
			expectedFSGroup: nil,
			expectedPSP:     runAsAny.Name,
		},
		"runAsAny pod request": {
			pod:             createPodWithFSGroup(1),
			psps:            []*extensions.PodSecurityPolicy{runAsAny},
			shouldPass:      true,
			expectedFSGroup: int64Ptr(1),
			expectedPSP:     runAsAny.Name,
		},
		"mustRunAs no pod request": {
			pod:             goodPod(),
			psps:            []*extensions.PodSecurityPolicy{mustRunAs},
			shouldPass:      true,
			expectedFSGroup: &mustRunAs.Spec.SupplementalGroups.Ranges[0].Min,
			expectedPSP:     mustRunAs.Name,
		},
		"mustRunAs bad pod request": {
			pod:        createPodWithFSGroup(1),
			psps:       []*extensions.PodSecurityPolicy{mustRunAs},
			shouldPass: false,
		},
		"mustRunAs good pod request": {
			pod:             createPodWithFSGroup(999),
			psps:            []*extensions.PodSecurityPolicy{mustRunAs},
			shouldPass:      true,
			expectedFSGroup: int64Ptr(999),
			expectedPSP:     mustRunAs.Name,
		},
	}

	for k, v := range tests {
		testPSPAdmit(k, v.psps, v.pod, v.shouldPass, v.expectedPSP, t)

		if v.shouldPass {
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
		pod           *kapi.Pod
		psps          []*extensions.PodSecurityPolicy
		shouldPass    bool
		expectedRORFS bool
		expectedPSP   string
	}{
		"no-rorfs allows pod request with rorfs": {
			pod:           createPodWithRORFS(true),
			psps:          []*extensions.PodSecurityPolicy{noRORFS},
			shouldPass:    true,
			expectedRORFS: true,
			expectedPSP:   noRORFS.Name,
		},
		"no-rorfs allows pod request without rorfs": {
			pod:           createPodWithRORFS(false),
			psps:          []*extensions.PodSecurityPolicy{noRORFS},
			shouldPass:    true,
			expectedRORFS: false,
			expectedPSP:   noRORFS.Name,
		},
		"rorfs rejects pod request without rorfs": {
			pod:        createPodWithRORFS(false),
			psps:       []*extensions.PodSecurityPolicy{rorfs},
			shouldPass: false,
		},
		"rorfs defaults nil pod request": {
			pod:           goodPod(),
			psps:          []*extensions.PodSecurityPolicy{rorfs},
			shouldPass:    true,
			expectedRORFS: true,
			expectedPSP:   rorfs.Name,
		},
		"rorfs accepts pod request with rorfs": {
			pod:           createPodWithRORFS(true),
			psps:          []*extensions.PodSecurityPolicy{rorfs},
			shouldPass:    true,
			expectedRORFS: true,
			expectedPSP:   rorfs.Name,
		},
	}

	for k, v := range tests {
		testPSPAdmit(k, v.psps, v.pod, v.shouldPass, v.expectedPSP, t)

		if v.shouldPass {
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
		pod.Annotations[kapi.SysctlsPodAnnotationKey] = kapi.PodAnnotationsFromSysctls(dummySysctls(safeSysctls))
		pod.Annotations[kapi.UnsafeSysctlsPodAnnotationKey] = kapi.PodAnnotationsFromSysctls(dummySysctls(unsafeSysctls))
		return pod
	}

	noSysctls := restrictivePSP()
	noSysctls.Name = "no sysctls"

	emptySysctls := restrictivePSP()
	emptySysctls.Name = "empty sysctls"
	emptySysctls.Annotations[extensions.SysctlsPodSecurityPolicyAnnotationKey] = ""

	mixedSysctls := restrictivePSP()
	mixedSysctls.Name = "wildcard sysctls"
	mixedSysctls.Annotations[extensions.SysctlsPodSecurityPolicyAnnotationKey] = "a.*,b.*,c,d.e.f"

	aSysctl := restrictivePSP()
	aSysctl.Name = "a sysctl"
	aSysctl.Annotations[extensions.SysctlsPodSecurityPolicyAnnotationKey] = "a"

	bSysctl := restrictivePSP()
	bSysctl.Name = "b sysctl"
	bSysctl.Annotations[extensions.SysctlsPodSecurityPolicyAnnotationKey] = "b"

	cSysctl := restrictivePSP()
	cSysctl.Name = "c sysctl"
	cSysctl.Annotations[extensions.SysctlsPodSecurityPolicyAnnotationKey] = "c"

	catchallSysctls := restrictivePSP()
	catchallSysctls.Name = "catchall sysctl"
	catchallSysctls.Annotations[extensions.SysctlsPodSecurityPolicyAnnotationKey] = "*"

	tests := map[string]struct {
		pod         *kapi.Pod
		psps        []*extensions.PodSecurityPolicy
		shouldPass  bool
		expectedPSP string
	}{
		"pod without unsafe sysctls request allowed under noSysctls PSP": {
			pod:         goodPod(),
			psps:        []*extensions.PodSecurityPolicy{noSysctls},
			shouldPass:  true,
			expectedPSP: noSysctls.Name,
		},
		"pod without any sysctls request allowed under emptySysctls PSP": {
			pod:         goodPod(),
			psps:        []*extensions.PodSecurityPolicy{emptySysctls},
			shouldPass:  true,
			expectedPSP: emptySysctls.Name,
		},
		"pod with safe sysctls request allowed under noSysctls PSP": {
			pod:         podWithSysctls([]string{"a", "b"}, []string{}),
			psps:        []*extensions.PodSecurityPolicy{noSysctls},
			shouldPass:  true,
			expectedPSP: noSysctls.Name,
		},
		"pod with unsafe sysctls request allowed under noSysctls PSP": {
			pod:         podWithSysctls([]string{}, []string{"a", "b"}),
			psps:        []*extensions.PodSecurityPolicy{noSysctls},
			shouldPass:  true,
			expectedPSP: noSysctls.Name,
		},
		"pod with safe sysctls request disallowed under emptySysctls PSP": {
			pod:        podWithSysctls([]string{"a", "b"}, []string{}),
			psps:       []*extensions.PodSecurityPolicy{emptySysctls},
			shouldPass: false,
		},
		"pod with unsafe sysctls a, b request disallowed under aSysctls SCC": {
			pod:        podWithSysctls([]string{}, []string{"a", "b"}),
			psps:       []*extensions.PodSecurityPolicy{aSysctl},
			shouldPass: false,
		},
		"pod with unsafe sysctls b request disallowed under aSysctls SCC": {
			pod:        podWithSysctls([]string{}, []string{"b"}),
			psps:       []*extensions.PodSecurityPolicy{aSysctl},
			shouldPass: false,
		},
		"pod with unsafe sysctls a request allowed under aSysctls SCC": {
			pod:         podWithSysctls([]string{}, []string{"a"}),
			psps:        []*extensions.PodSecurityPolicy{aSysctl},
			shouldPass:  true,
			expectedPSP: aSysctl.Name,
		},
		"pod with safe sysctls a, b request disallowed under aSysctls SCC": {
			pod:        podWithSysctls([]string{"a", "b"}, []string{}),
			psps:       []*extensions.PodSecurityPolicy{aSysctl},
			shouldPass: false,
		},
		"pod with safe sysctls b request disallowed under aSysctls SCC": {
			pod:        podWithSysctls([]string{"b"}, []string{}),
			psps:       []*extensions.PodSecurityPolicy{aSysctl},
			shouldPass: false,
		},
		"pod with safe sysctls a request allowed under aSysctls SCC": {
			pod:         podWithSysctls([]string{"a"}, []string{}),
			psps:        []*extensions.PodSecurityPolicy{aSysctl},
			shouldPass:  true,
			expectedPSP: aSysctl.Name,
		},
		"pod with unsafe sysctls request disallowed under emptySysctls PSP": {
			pod:        podWithSysctls([]string{}, []string{"a", "b"}),
			psps:       []*extensions.PodSecurityPolicy{emptySysctls},
			shouldPass: false,
		},
		"pod with matching sysctls request allowed under mixedSysctls PSP": {
			pod:         podWithSysctls([]string{"a.b", "b.c"}, []string{"c", "d.e.f"}),
			psps:        []*extensions.PodSecurityPolicy{mixedSysctls},
			shouldPass:  true,
			expectedPSP: mixedSysctls.Name,
		},
		"pod with not-matching unsafe sysctls request disallowed under mixedSysctls PSP": {
			pod:        podWithSysctls([]string{"a.b", "b.c", "c", "d.e.f"}, []string{"e"}),
			psps:       []*extensions.PodSecurityPolicy{mixedSysctls},
			shouldPass: false,
		},
		"pod with not-matching safe sysctls request disallowed under mixedSysctls PSP": {
			pod:        podWithSysctls([]string{"a.b", "b.c", "c", "d.e.f", "e"}, []string{}),
			psps:       []*extensions.PodSecurityPolicy{mixedSysctls},
			shouldPass: false,
		},
		"pod with sysctls request allowed under catchallSysctls PSP": {
			pod:         podWithSysctls([]string{"e"}, []string{"f"}),
			psps:        []*extensions.PodSecurityPolicy{catchallSysctls},
			shouldPass:  true,
			expectedPSP: catchallSysctls.Name,
		},
		"pod with sysctls request allowed under catchallSysctls PSP, not under mixedSysctls or emptySysctls PSP": {
			pod:         podWithSysctls([]string{"e"}, []string{"f"}),
			psps:        []*extensions.PodSecurityPolicy{mixedSysctls, catchallSysctls, emptySysctls},
			shouldPass:  true,
			expectedPSP: catchallSysctls.Name,
		},
		"pod with safe c sysctl request allowed under cSysctl PSP, not under aSysctl or bSysctl PSP": {
			pod:         podWithSysctls([]string{}, []string{"c"}),
			psps:        []*extensions.PodSecurityPolicy{aSysctl, bSysctl, cSysctl},
			shouldPass:  true,
			expectedPSP: cSysctl.Name,
		},
		"pod with unsafe c sysctl request allowed under cSysctl PSP, not under aSysctl or bSysctl PSP": {
			pod:         podWithSysctls([]string{"c"}, []string{}),
			psps:        []*extensions.PodSecurityPolicy{aSysctl, bSysctl, cSysctl},
			shouldPass:  true,
			expectedPSP: cSysctl.Name,
		},
	}

	for k, v := range tests {
		origSafeSysctls, origUnsafeSysctls, err := kapi.SysctlsFromPodAnnotations(v.pod.Annotations)
		if err != nil {
			t.Fatalf("invalid sysctl annotation: %v", err)
		}

		testPSPAdmit(k, v.psps, v.pod, v.shouldPass, v.expectedPSP, t)

		if v.shouldPass {
			safeSysctls, unsafeSysctls, _ := kapi.SysctlsFromPodAnnotations(v.pod.Annotations)
			if !reflect.DeepEqual(safeSysctls, origSafeSysctls) {
				t.Errorf("%s: wrong safe sysctls: expected=%v, got=%v", k, origSafeSysctls, safeSysctls)
			}
			if !reflect.DeepEqual(unsafeSysctls, origUnsafeSysctls) {
				t.Errorf("%s: wrong unsafe sysctls: expected=%v, got=%v", k, origSafeSysctls, safeSysctls)
			}
		}
	}
}

func testPSPAdmit(testCaseName string, psps []*extensions.PodSecurityPolicy, pod *kapi.Pod, shouldPass bool, expectedPSP string, t *testing.T) {
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	store := informerFactory.Extensions().InternalVersion().PodSecurityPolicies().Informer().GetStore()

	for _, psp := range psps {
		store.Add(psp)
	}

	plugin := NewTestAdmission(informerFactory.Extensions().InternalVersion().PodSecurityPolicies().Lister())

	attrs := kadmission.NewAttributesRecord(pod, nil, kapi.Kind("Pod").WithVersion("version"), "namespace", "", kapi.Resource("pods").WithVersion("version"), "", kadmission.Create, &user.DefaultInfo{})
	err := plugin.Admit(attrs)

	if shouldPass && err != nil {
		t.Errorf("%s: expected no errors but received %v", testCaseName, err)
	}

	if shouldPass && err == nil {
		if pod.Annotations[psputil.ValidatedPSPAnnotation] != expectedPSP {
			t.Errorf("%s: expected to validate under %s but found %s", testCaseName, expectedPSP, pod.Annotations[psputil.ValidatedPSPAnnotation])
		}
	}

	if !shouldPass && err == nil {
		t.Errorf("%s: expected errors but received none", testCaseName)
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

	// these are set up such that the containers always have a nil uid.  If the case should not
	// validate then the uids should not have been updated by the strategy.  If the case should
	// validate then uids should be set.  This is ensuring that we're hanging on to the old SC
	// as we generate/validate and only updating the original container if the entire pod validates
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
		errs := assignSecurityContext(provider, v.pod, nil)
		if v.shouldValidate && len(errs) > 0 {
			t.Errorf("%s expected to validate but received errors %v", k, errs)
			continue
		}
		if !v.shouldValidate && len(errs) == 0 {
			t.Errorf("%s expected validation errors but received none", k)
			continue
		}

		// if we shouldn't have validated ensure that uid is not set on the containers
		if !v.shouldValidate {
			for _, c := range v.pod.Spec.Containers {
				if c.SecurityContext.RunAsUser != nil {
					t.Errorf("%s had non-nil UID %d.  UID should not be set on test cases that don't validate", k, *c.SecurityContext.RunAsUser)
				}
			}
		}

		// if we validated then the pod sc should be updated now with the defaults from the psp
		if v.shouldValidate {
			for _, c := range v.pod.Spec.Containers {
				if *c.SecurityContext.RunAsUser != 999 {
					t.Errorf("%s expected uid to be defaulted to 999 but found %v", k, *c.SecurityContext.RunAsUser)
				}
			}
		}
	}
}

func TestCreateProvidersFromConstraints(t *testing.T) {
	testCases := map[string]struct {
		// use a generating function so we can test for non-mutation
		psp         func() *extensions.PodSecurityPolicy
		expectedErr string
	}{
		"valid psp": {
			psp: func() *extensions.PodSecurityPolicy {
				return &extensions.PodSecurityPolicy{
					ObjectMeta: metav1.ObjectMeta{
						Name: "valid psp",
					},
					Spec: extensions.PodSecurityPolicySpec{
						SELinux: extensions.SELinuxStrategyOptions{
							Rule: extensions.SELinuxStrategyRunAsAny,
						},
						RunAsUser: extensions.RunAsUserStrategyOptions{
							Rule: extensions.RunAsUserStrategyRunAsAny,
						},
						FSGroup: extensions.FSGroupStrategyOptions{
							Rule: extensions.FSGroupStrategyRunAsAny,
						},
						SupplementalGroups: extensions.SupplementalGroupsStrategyOptions{
							Rule: extensions.SupplementalGroupsStrategyRunAsAny,
						},
					},
				}
			},
		},
		"bad psp strategy options": {
			psp: func() *extensions.PodSecurityPolicy {
				return &extensions.PodSecurityPolicy{
					ObjectMeta: metav1.ObjectMeta{
						Name: "bad psp user options",
					},
					Spec: extensions.PodSecurityPolicySpec{
						SELinux: extensions.SELinuxStrategyOptions{
							Rule: extensions.SELinuxStrategyRunAsAny,
						},
						RunAsUser: extensions.RunAsUserStrategyOptions{
							Rule: extensions.RunAsUserStrategyMustRunAs,
						},
						FSGroup: extensions.FSGroupStrategyOptions{
							Rule: extensions.FSGroupStrategyRunAsAny,
						},
						SupplementalGroups: extensions.SupplementalGroupsStrategyOptions{
							Rule: extensions.SupplementalGroupsStrategyRunAsAny,
						},
					},
				}
			},
			expectedErr: "MustRunAsRange requires at least one range",
		},
	}

	for k, v := range testCases {
		admit := &podSecurityPolicyPlugin{
			Handler:         kadmission.NewHandler(kadmission.Create, kadmission.Update),
			strategyFactory: kpsp.NewSimpleStrategyFactory(),
		}

		psp := v.psp()
		_, errs := admit.createProvidersFromPolicies([]*extensions.PodSecurityPolicy{psp}, "namespace")

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

func TestGetMatchingPolicies(t *testing.T) {
	policyWithName := func(name string) *extensions.PodSecurityPolicy {
		p := restrictivePSP()
		p.Name = name
		return p
	}

	tests := map[string]struct {
		user               user.Info
		sa                 user.Info
		expectedPolicies   sets.String
		inPolicies         []*extensions.PodSecurityPolicy
		disallowedPolicies map[string][]string
	}{
		"policy allowed by user": {
			user: &user.DefaultInfo{Name: "user"},
			sa:   &user.DefaultInfo{Name: "sa"},
			disallowedPolicies: map[string][]string{
				"sa": {"policy"},
			},
			inPolicies:       []*extensions.PodSecurityPolicy{policyWithName("policy")},
			expectedPolicies: sets.NewString("policy"),
		},
		"policy allowed by sa": {
			user: &user.DefaultInfo{Name: "user"},
			sa:   &user.DefaultInfo{Name: "sa"},
			disallowedPolicies: map[string][]string{
				"user": {"policy"},
			},
			inPolicies:       []*extensions.PodSecurityPolicy{policyWithName("policy")},
			expectedPolicies: sets.NewString("policy"),
		},
		"no policies allowed": {
			user: &user.DefaultInfo{Name: "user"},
			sa:   &user.DefaultInfo{Name: "sa"},
			disallowedPolicies: map[string][]string{
				"user": {"policy"},
				"sa":   {"policy"},
			},
			inPolicies:       []*extensions.PodSecurityPolicy{policyWithName("policy")},
			expectedPolicies: sets.NewString(),
		},
		"multiple policies allowed": {
			user: &user.DefaultInfo{Name: "user"},
			sa:   &user.DefaultInfo{Name: "sa"},
			disallowedPolicies: map[string][]string{
				"user": {"policy1", "policy3"},
				"sa":   {"policy2", "policy3"},
			},
			inPolicies: []*extensions.PodSecurityPolicy{
				policyWithName("policy1"), // allowed by sa
				policyWithName("policy2"), // allowed by user
				policyWithName("policy3"), // not allowed
			},
			expectedPolicies: sets.NewString("policy1", "policy2"),
		},
		"policies are allowed for nil user info": {
			user: nil,
			sa:   &user.DefaultInfo{Name: "sa"},
			disallowedPolicies: map[string][]string{
				"user": {"policy1", "policy3"},
				"sa":   {"policy2", "policy3"},
			},
			inPolicies: []*extensions.PodSecurityPolicy{
				policyWithName("policy1"),
				policyWithName("policy2"),
				policyWithName("policy3"),
			},
			// all policies are allowed regardless of the permissions when user info is nil
			// (ie. a request hitting the unsecure port)
			expectedPolicies: sets.NewString("policy1", "policy2", "policy3"),
		},
		"policies are allowed for nil sa info": {
			user: &user.DefaultInfo{Name: "user"},
			sa:   nil,
			disallowedPolicies: map[string][]string{
				"user": {"policy1", "policy3"},
				"sa":   {"policy2", "policy3"},
			},
			inPolicies: []*extensions.PodSecurityPolicy{
				policyWithName("policy1"),
				policyWithName("policy2"),
				policyWithName("policy3"),
			},
			// all policies are allowed regardless of the permissions when sa info is nil
			// (ie. a request hitting the unsecure port)
			expectedPolicies: sets.NewString("policy1", "policy2", "policy3"),
		},
	}
	for k, v := range tests {
		informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
		pspInformer := informerFactory.Extensions().InternalVersion().PodSecurityPolicies()
		store := pspInformer.Informer().GetStore()
		for _, psp := range v.inPolicies {
			store.Add(psp)
		}

		authz := &TestAuthorizer{disallowed: v.disallowedPolicies}
		allowedPolicies, err := getMatchingPolicies(pspInformer.Lister(), v.user, v.sa, authz)
		if err != nil {
			t.Errorf("%s got unexpected error %#v", k, err)
			continue
		}
		allowedPolicyNames := sets.NewString()
		for _, p := range allowedPolicies {
			allowedPolicyNames.Insert(p.Name)
		}
		if !v.expectedPolicies.Equal(allowedPolicyNames) {
			t.Errorf("%s received unexpected policies.  Expected %#v but got %#v", k, v.expectedPolicies.List(), allowedPolicyNames.List())
		}
	}
}

func restrictivePSP() *extensions.PodSecurityPolicy {
	return &extensions.PodSecurityPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "restrictive",
			Annotations: map[string]string{},
		},
		Spec: extensions.PodSecurityPolicySpec{
			RunAsUser: extensions.RunAsUserStrategyOptions{
				Rule: extensions.RunAsUserStrategyMustRunAs,
				Ranges: []extensions.IDRange{
					{Min: 999, Max: 999},
				},
			},
			SELinux: extensions.SELinuxStrategyOptions{
				Rule: extensions.SELinuxStrategyMustRunAs,
				SELinuxOptions: &kapi.SELinuxOptions{
					Level: "s9:z0,z1",
				},
			},
			FSGroup: extensions.FSGroupStrategyOptions{
				Rule: extensions.FSGroupStrategyMustRunAs,
				Ranges: []extensions.IDRange{
					{Min: 999, Max: 999},
				},
			},
			SupplementalGroups: extensions.SupplementalGroupsStrategyOptions{
				Rule: extensions.SupplementalGroupsStrategyMustRunAs,
				Ranges: []extensions.IDRange{
					{Min: 999, Max: 999},
				},
			},
		},
	}
}

func createNamespaceForTest() *kapi.Namespace {
	return &kapi.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: "default",
		},
	}
}

func createSAForTest() *kapi.ServiceAccount {
	return &kapi.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "default",
		},
	}
}

// goodPod is empty and should not be used directly for testing since we're providing
// two different PSPs.  Since no values are specified it would be allowed to match any
// psp when defaults are filled in.
func goodPod() *kapi.Pod {
	return &kapi.Pod{
		ObjectMeta: metav1.ObjectMeta{
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

func intPtr(i int) *int {
	return &i
}

func int64Ptr(i int) *int64 {
	i64 := int64(i)
	return &i64
}
