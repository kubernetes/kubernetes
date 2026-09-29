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

package kuberuntime

import (
	"errors"
	"fmt"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	crierrors "k8s.io/cri-api/pkg/errors"
	"k8s.io/kubernetes/pkg/credentialprovider"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/test/utils/ktesting"
)

const (
	testProfileRef      = "registry.example.com/profile@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	testOtherProfileRef = "registry.example.com/other@sha256:fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
)

func ociSeccompProfile(ref string, base *v1.SecurityProfileOCIBase) *v1.SeccompProfile {
	return &v1.SeccompProfile{Type: v1.SeccompProfileTypeOCI, OCI: &v1.SecurityProfileOCI{Ref: ref, BaseProfile: base}}
}

func TestFieldSeccompProfileOCI(t *testing.T) {
	root := filepath.Join("var", "lib", "kubelet", "seccomp")
	tests := []struct {
		name     string
		profile  *v1.SeccompProfile
		disabled bool
		expected *runtimeapi.SecurityProfile
		err      bool
	}{{
		name:     "without base profile",
		profile:  ociSeccompProfile(testProfileRef, nil),
		expected: &runtimeapi.SecurityProfile{ProfileType: runtimeapi.SecurityProfile_OCI, OciRef: testProfileRef},
	}, {
		name:    "with RuntimeDefault base profile",
		profile: ociSeccompProfile(testProfileRef, &v1.SecurityProfileOCIBase{Type: v1.SecurityProfileOCIBaseTypeRuntimeDefault}),
		expected: &runtimeapi.SecurityProfile{
			ProfileType: runtimeapi.SecurityProfile_OCI,
			OciRef:      testProfileRef,
			BaseProfile: &runtimeapi.SecurityProfileBase{Type: runtimeapi.SecurityProfileBase_RuntimeDefault},
		},
	}, {
		name:    "with Localhost base profile",
		profile: ociSeccompProfile(testProfileRef, &v1.SecurityProfileOCIBase{Type: v1.SecurityProfileOCIBaseTypeLocalhost, LocalhostProfile: new("agentic.json")}),
		expected: &runtimeapi.SecurityProfile{
			ProfileType: runtimeapi.SecurityProfile_OCI,
			OciRef:      testProfileRef,
			BaseProfile: &runtimeapi.SecurityProfileBase{Type: runtimeapi.SecurityProfileBase_Localhost, LocalhostRef: filepath.Join(root, "agentic.json")},
		},
	}, {
		name:    "Localhost base profile without path",
		profile: ociSeccompProfile(testProfileRef, &v1.SecurityProfileOCIBase{Type: v1.SecurityProfileOCIBaseTypeLocalhost}),
		err:     true,
	}, {
		name:    "missing oci",
		profile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeOCI},
		err:     true,
	}, {
		name:     "feature disabled",
		profile:  ociSeccompProfile(testProfileRef, nil),
		disabled: true,
		err:      true,
	}}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SecurityProfileOCI, !tc.disabled)
			profile, err := fieldSeccompProfile(tc.profile, root, true)
			if tc.err {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tc.expected, profile)
		})
	}
}

func TestPullSecurityProfileCredentials(t *testing.T) {
	tCtx := ktesting.Init(t)
	image := kubecontainer.ImageSpec{Image: testProfileRef}
	criImage := &runtimeapi.ImageSpec{Image: testProfileRef, UserSpecifiedImage: testProfileRef}
	creds := []credentialprovider.TrackedAuthConfig{
		{AuthConfig: credentialprovider.AuthConfig{Username: "first", Password: "pass"}},
		{AuthConfig: credentialprovider.AuthConfig{Username: "second", Password: "pass"}},
	}

	t.Run("anonymous pull is cached the second time", func(t *testing.T) {
		_, fakeImageService, m, err := createTestRuntimeManager(tCtx)
		require.NoError(t, err)
		for _, expected := range []bool{false, true} {
			cached, err := m.PullSecurityProfile(tCtx, image, nil, nil, runtimeapi.SecurityProfileKind_Seccomp)
			require.NoError(t, err)
			assert.Equal(t, expected, cached)
		}
		fakeImageService.AssertSecurityProfilePulledWithAuth(t, criImage, nil, runtimeapi.SecurityProfileKind_Seccomp, "anonymous pull")
	})

	t.Run("falls back to the next credentials", func(t *testing.T) {
		_, fakeImageService, m, err := createTestRuntimeManager(tCtx)
		require.NoError(t, err)
		fakeImageService.InjectError("PullSecurityProfile", errors.New("unauthorized"))
		_, err = m.PullSecurityProfile(tCtx, image, creds, nil, runtimeapi.SecurityProfileKind_Seccomp)
		require.NoError(t, err)
		fakeImageService.AssertSecurityProfilePulledWithAuth(t, criImage, &runtimeapi.AuthConfig{Username: "second", Password: "pass"}, runtimeapi.SecurityProfileKind_Seccomp, "second credentials")
	})

	t.Run("a rejected profile stops at the first credentials", func(t *testing.T) {
		_, fakeImageService, m, err := createTestRuntimeManager(tCtx)
		require.NoError(t, err)
		fakeImageService.InjectError("PullSecurityProfile", fmt.Errorf("%w: invalid content", crierrors.ErrSecurityProfileInvalid))
		_, err = m.PullSecurityProfile(tCtx, image, creds, nil, runtimeapi.SecurityProfileKind_Seccomp)
		require.EqualError(t, err, "SecurityProfileInvalid: invalid content")
		assert.Equal(t, 1, countCalls(fakeImageService.Called, "PullSecurityProfile"))
	})

	t.Run("Unimplemented stops at the first credentials and keeps its code", func(t *testing.T) {
		_, fakeImageService, m, err := createTestRuntimeManager(tCtx)
		require.NoError(t, err)
		fakeImageService.InjectError("PullSecurityProfile", status.Error(codes.Unimplemented, "unknown method"))
		_, err = m.PullSecurityProfile(tCtx, image, creds, nil, runtimeapi.SecurityProfileKind_Seccomp)
		assert.Equal(t, codes.Unimplemented, status.Code(err))
		assert.Equal(t, 1, countCalls(fakeImageService.Called, "PullSecurityProfile"))
	})

	t.Run("the last error is returned unchanged when all credentials fail", func(t *testing.T) {
		_, fakeImageService, m, err := createTestRuntimeManager(tCtx)
		require.NoError(t, err)
		fakeImageService.InjectError("PullSecurityProfile", status.Error(codes.Unauthenticated, "first"))
		fakeImageService.InjectError("PullSecurityProfile", fmt.Errorf("%w: timeout", crierrors.ErrRegistryUnavailable))
		_, err = m.PullSecurityProfile(tCtx, image, creds, nil, runtimeapi.SecurityProfileKind_Seccomp)
		require.EqualError(t, err, "RegistryUnavailable: timeout")
	})
}

func countCalls(called []string, name string) int {
	n := 0
	for _, c := range called {
		if c == name {
			n++
		}
	}
	return n
}

func TestEnsureSecurityProfiles(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, fakeImageService, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod", Namespace: "ns", UID: "uid"},
		Spec: v1.PodSpec{
			SecurityContext: &v1.PodSecurityContext{SeccompProfile: ociSeccompProfile(testProfileRef, nil)},
			InitContainers:  []v1.Container{{Name: "init", SecurityContext: &v1.SecurityContext{SeccompProfile: ociSeccompProfile(testOtherProfileRef, nil)}}},
			Containers: []v1.Container{
				{Name: "a", SecurityContext: &v1.SecurityContext{SeccompProfile: ociSeccompProfile(testProfileRef, nil)}},
				{Name: "b", SecurityContext: &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault}}},
			},
		},
	}
	require.NoError(t, m.EnsureSecurityProfiles(tCtx, pod, &kubecontainer.PodStatus{}, nil))
	assert.Equal(t, 2, countCalls(fakeImageService.Called, "PullSecurityProfile"), "one pull per unique reference")
	fakeImageService.Lock()
	assert.Equal(t, map[string]runtimeapi.SecurityProfileKind{
		testProfileRef:      runtimeapi.SecurityProfileKind_Seccomp,
		testOtherProfileRef: runtimeapi.SecurityProfileKind_Seccomp,
	}, fakeImageService.SecurityProfiles)
	fakeImageService.Unlock()

	// Pods without OCI profiles do not call the runtime.
	fakeImageService.Called = nil
	require.NoError(t, m.EnsureSecurityProfiles(tCtx, &v1.Pod{Spec: v1.PodSpec{Containers: []v1.Container{{Name: "c"}}}}, &kubecontainer.PodStatus{}, nil))
	assert.Zero(t, countCalls(fakeImageService.Called, "PullSecurityProfile"))
}
