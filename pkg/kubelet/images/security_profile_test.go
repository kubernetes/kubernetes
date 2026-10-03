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

package images

import (
	"errors"
	"fmt"
	"slices"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/util/flowcontrol"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	crierrors "k8s.io/cri-api/pkg/errors"
	"k8s.io/kubernetes/pkg/controller/testutil"
	"k8s.io/kubernetes/pkg/credentialprovider"
	ctest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/test/utils/ktesting"
	testingclock "k8s.io/utils/clock/testing"
)

const testSecurityProfileRef = "registry.example.com/profile@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

func newSecurityProfileTestManager(t *testing.T, serialized bool) (ImageManager, *ctest.FakeRuntime, *testutil.FakeRecorder, *testingclock.FakeClock) {
	return newSecurityProfileTestManagerWithQPS(t, serialized, 0, 0)
}

func newSecurityProfileTestManagerWithQPS(t *testing.T, serialized bool, qps float32, burst int) (ImageManager, *ctest.FakeRuntime, *testutil.FakeRecorder, *testingclock.FakeClock) {
	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	fakeClock := testingclock.NewFakeClock(time.Now())
	backOff.Clock = fakeClock
	fakeRuntime := &ctest.FakeRuntime{T: t}
	fakeRecorder := testutil.NewFakeRecorder()
	manager := NewImageManager(fakeRecorder, &credentialprovider.BasicDockerKeyring{}, fakeRuntime, &mockImagePullManager{config: &mockImagePullManagerConfig{allowAll: true}}, backOff, serialized, nil, qps, burst, &mockPodPullingTimeRecorder{
		startedPullingRecorded:  make(map[types.UID]bool),
		finishedPullingRecorded: make(map[types.UID]bool),
	})
	return manager, fakeRuntime, fakeRecorder, fakeClock
}

func securityProfileEventReasons(recorder *testutil.FakeRecorder) []string {
	recorder.Lock()
	defer recorder.Unlock()
	var reasons []string
	for _, event := range recorder.Events {
		reasons = append(reasons, event.Reason)
	}
	return reasons
}

func TestEnsureSecurityProfile(t *testing.T) {
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod", Namespace: "ns", UID: "uid"}}
	objRef := &v1.ObjectReference{Kind: "Pod", Name: "pod", Namespace: "ns", UID: "uid"}
	sandboxConfig := &runtimeapi.PodSandboxConfig{Metadata: &runtimeapi.PodSandboxMetadata{Name: "pod", Namespace: "ns", Uid: "uid"}}

	for _, serialized := range []bool{true, false} {
		t.Run(fmt.Sprintf("serialized=%v", serialized), func(t *testing.T) {
			tCtx := ktesting.Init(t)

			t.Run("pulls once and reports cached pulls without events", func(t *testing.T) {
				manager, fakeRuntime, recorder, _ := newSecurityProfileTestManager(t, serialized)
				for range 2 {
					require.NoError(t, manager.EnsureSecurityProfile(tCtx, objRef, pod, testSecurityProfileRef, runtimeapi.SecurityProfileKind_Seccomp, nil, sandboxConfig, ""))
				}
				assert.Equal(t, []string{testSecurityProfileRef, testSecurityProfileRef}, fakeRuntime.PulledSecurityProfiles)
				assert.Equal(t, []string{events.PulledSecurityProfile}, securityProfileEventReasons(recorder))
			})

			for _, tc := range []struct {
				name     string
				err      error
				rejected bool
			}{{
				name:     "SecurityProfileInvalid is permanent",
				err:      fmt.Errorf("%w: wrong config media type", crierrors.ErrSecurityProfileInvalid),
				rejected: true,
			}, {
				name:     "Unimplemented is permanent",
				err:      status.Error(codes.Unimplemented, "unknown method PullSecurityProfile"),
				rejected: true,
			}, {
				name: "RegistryUnavailable is retried",
				err:  fmt.Errorf("%w: connection refused", crierrors.ErrRegistryUnavailable),
			}, {
				name: "authentication failures are retried",
				err:  status.Error(codes.Unauthenticated, "unauthorized"),
			}} {
				t.Run(tc.name, func(t *testing.T) {
					manager, fakeRuntime, recorder, fakeClock := newSecurityProfileTestManager(t, serialized)
					fakeRuntime.Err = tc.err

					err := manager.EnsureSecurityProfile(tCtx, objRef, pod, testSecurityProfileRef, runtimeapi.SecurityProfileKind_Seccomp, nil, sandboxConfig, "")
					require.Error(t, err)
					assert.Equal(t, tc.rejected, errors.Is(err, ErrSecurityProfileRejected), err.Error())
					assert.Equal(t, []string{events.FailedToPullSecurityProfile}, securityProfileEventReasons(recorder))

					// Transient failures back off before the next call reaches the runtime.
					err = manager.EnsureSecurityProfile(tCtx, objRef, pod, testSecurityProfileRef, runtimeapi.SecurityProfileKind_Seccomp, nil, sandboxConfig, "")
					require.Error(t, err)
					if tc.rejected {
						assert.NotErrorIs(t, err, ErrSecurityProfilePullBackOff)
						return
					}
					require.ErrorIs(t, err, ErrSecurityProfilePullBackOff)

					fakeClock.Step(2 * time.Second)
					fakeRuntime.Err = nil
					require.NoError(t, manager.EnsureSecurityProfile(tCtx, objRef, pod, testSecurityProfileRef, runtimeapi.SecurityProfileKind_Seccomp, nil, sandboxConfig, ""))
				})
			}
		})
	}
}

func TestSecurityProfilePullErrorReason(t *testing.T) {
	for _, tc := range []struct {
		err      error
		expected string
	}{
		{fmt.Errorf("%w: bad layer", crierrors.ErrSecurityProfileInvalid), securityProfilePullErrorInvalid},
		{fmt.Errorf("%w: timeout", crierrors.ErrRegistryUnavailable), securityProfilePullErrorRegistryUnavailable},
		{fmt.Errorf("%w: no signature", crierrors.ErrSignatureValidationFailed), securityProfilePullErrorSignatureValidation},
		{status.Error(codes.Unauthenticated, "unauthorized"), securityProfilePullErrorUnauthenticated},
		{status.Error(codes.PermissionDenied, "denied"), securityProfilePullErrorUnauthenticated},
		{errors.New("boom"), securityProfilePullErrorOther},
	} {
		assert.Equal(t, tc.expected, securityProfilePullErrorReason(tc.err), tc.err.Error())
	}
}

func TestEnsureSecurityProfileLimits(t *testing.T) {
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod", Namespace: "ns", UID: "uid"}}
	sandboxConfig := &runtimeapi.PodSandboxConfig{Metadata: &runtimeapi.PodSandboxMetadata{Name: "pod", Namespace: "ns", Uid: "uid"}}

	for _, serialized := range []bool{true, false} {
		t.Run(fmt.Sprintf("serialized=%v", serialized), func(t *testing.T) {
			t.Run("not limited by the registry QPS", func(t *testing.T) {
				tCtx := ktesting.Init(t)
				manager, _, _, _ := newSecurityProfileTestManagerWithQPS(t, serialized, 0.001, 1)
				for range 3 {
					require.NoError(t, manager.EnsureSecurityProfile(tCtx, nil, pod, testSecurityProfileRef, runtimeapi.SecurityProfileKind_Seccomp, nil, sandboxConfig, ""))
				}
			})

			t.Run("does not wait for image pulls", func(t *testing.T) {
				tCtx := ktesting.Init(t)
				manager, fakeRuntime, _, _ := newSecurityProfileTestManager(t, serialized)
				fakeRuntime.BlockImagePulls = true
				imagePulled := make(chan error, 1)
				go func() {
					_, _, err := manager.EnsureImageExists(tCtx, nil, pod, "blocked:latest", nil, sandboxConfig, "", v1.PullAlways)
					imagePulled <- err
				}()
				require.Eventually(t, func() bool {
					fakeRuntime.Lock()
					defer fakeRuntime.Unlock()
					return slices.Contains(fakeRuntime.CalledFunctions, "PullImage")
				}, 10*time.Second, 10*time.Millisecond)

				require.NoError(t, manager.EnsureSecurityProfile(tCtx, nil, pod, testSecurityProfileRef, runtimeapi.SecurityProfileKind_Seccomp, nil, sandboxConfig, ""))
				fakeRuntime.UnblockImagePulls(1)
				require.NoError(t, <-imagePulled)
			})

			t.Run("cancellation neither fails nor backs off", func(t *testing.T) {
				tCtx := ktesting.Init(t)
				manager, fakeRuntime, recorder, _ := newSecurityProfileTestManager(t, serialized)
				cancelled := tCtx.WithCancel()
				cancelled.Cancel("test")
				require.Error(t, manager.EnsureSecurityProfile(cancelled, nil, pod, testSecurityProfileRef, runtimeapi.SecurityProfileKind_Seccomp, nil, sandboxConfig, ""))
				assert.Empty(t, securityProfileEventReasons(recorder))
				require.NoError(t, manager.EnsureSecurityProfile(tCtx, nil, pod, testSecurityProfileRef, runtimeapi.SecurityProfileKind_Seccomp, nil, sandboxConfig, ""))
				assert.Equal(t, []string{testSecurityProfileRef}, fakeRuntime.PulledSecurityProfiles)
			})
		})
	}
}
