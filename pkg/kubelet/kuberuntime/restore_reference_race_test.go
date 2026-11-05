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
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/util/flowcontrol"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	nodeapi "k8s.io/kubernetes/pkg/apis/node"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/events"
)

func TestRestoreCheckpointDisappearanceNeverStartsFreshSandbox(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	for _, cachedReady := range []bool{false, true} {
		t.Run(map[bool]string{false: "live checkpoint missing", true: "ready cached checkpoint"}[cachedReady], func(t *testing.T) {
			ctx := ktesting.Init(t)
			runtime, _, m, err := createTestRuntimeManager(ctx)
			require.NoError(t, err)
			helper := &testRuntimeHelper{FakeRuntimeHelper: m.runtimeHelper.(*containertest.FakeRuntimeHelper)}
			if !cachedReady {
				helper.getPodCheckpointPathError = &kubecontainer.RestoreError{Reason: events.CheckpointNotReady, Err: apierrors.NewNotFound(schema.GroupResource{Group: "node.k8s.io", Resource: "podcheckpoints"}, "checkpoint")}
			}
			m.runtimeHelper = helper
			pod := newTestPod()
			pod.Spec.RestoreFrom = &v1.CheckpointReference{Name: "checkpoint"}
			pod.Finalizers = append(pod.Finalizers, nodeapi.PodRestoreProtectionFinalizer)
			result := m.SyncPod(ctx, pod, &kubecontainer.PodStatus{ID: pod.UID}, nil, flowcontrol.NewBackOff(time.Second, time.Minute), false)
			require.NotContains(t, runtime.Called, "RunPodSandbox", "a lost checkpoint reference cannot authorize ordinary startup")
			if cachedReady {
				require.NoError(t, result.Error())
				require.Len(t, runtime.RestoredPods, 1)
			} else {
				require.Error(t, result.Error())
				require.ErrorIs(t, result.Error(), helper.getPodCheckpointPathError)
				require.Empty(t, runtime.RestoredPods)
				require.NotContains(t, runtime.Called, "CreateContainer")
				require.NotContains(t, runtime.Called, "StartContainer")
			}
		})
	}
}
