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

package kubelet

import (
	"context"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/pleg"
	"k8s.io/kubernetes/pkg/kubelet/prober"
	proberesults "k8s.io/kubernetes/pkg/kubelet/prober/results"
	probetest "k8s.io/kubernetes/pkg/kubelet/prober/testing"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/test/utils/ktesting"
)

type recordingProbeManager struct {
	probetest.FakeManager
	added, reconciled, validated int
}

func (m *recordingProbeManager) AddPod(context.Context, *v1.Pod) { m.added++ }

func (m *recordingProbeManager) ReconcilePod(context.Context, *v1.Pod) (bool, error) {
	m.reconciled++
	return false, nil
}

func (m *recordingProbeManager) IsResultCurrent(proberesults.Update, prober.ProbeType) bool {
	m.validated++
	return false
}

func TestSyncPodProbeManagerFeatureGate(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		t.Run(fmt.Sprintf("enabled=%v", enabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutableContainerProbes, enabled)

			ctx := ktesting.Init(t)
			testKubelet := newTestKubelet(t, false)
			defer testKubelet.Cleanup()

			kl := testKubelet.kubelet
			probes := &recordingProbeManager{}
			kl.probeManager = probes

			pods := []*v1.Pod{podWithUIDNameNsSpec("12345678", "foo", "new", v1.PodSpec{Containers: []v1.Container{{Name: "bar"}}})}
			kl.podManager.SetPods(pods)

			kl.HandlePodSyncs(ctx, pods)

			if enabled {
				if probes.reconciled != 1 || probes.added != 0 {
					t.Fatalf("enabled gate: ReconcilePod=%d AddPod=%d, want 1 and 0", probes.reconciled, probes.added)
				}
			} else if probes.added != 1 || probes.reconciled != 0 {
				t.Fatalf("disabled gate: AddPod=%d ReconcilePod=%d, want 1 and 0", probes.added, probes.reconciled)
			}
		})
	}
}

func TestProbeResultValidationFeatureGate(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		for _, kind := range []prober.ProbeType{prober.ProbeTypeLiveness, prober.ProbeTypeReadiness, prober.ProbeTypeStartup} {
			t.Run(fmt.Sprintf("enabled=%v/%v", enabled, kind), func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutableContainerProbes, enabled)

				ctx := ktesting.Init(t)
				testKubelet := newTestKubelet(t, false)
				defer testKubelet.Cleanup()

				kl := testKubelet.kubelet
				probes := &recordingProbeManager{}
				kl.probeManager = probes

				cache := kl.livenessManager
				if kind == prober.ProbeTypeReadiness {
					cache = kl.readinessManager
				}

				if kind == prober.ProbeTypeStartup {
					cache = kl.startupManager
				}

				cache.Set(kubecontainer.ContainerID{Type: "test", ID: "container"}, proberesults.Success, &v1.Pod{})
				kl.syncLoopIteration(ctx, make(chan kubetypes.PodUpdate), kl, make(chan time.Time), make(chan time.Time), make(chan *pleg.PodLifecycleEvent))

				want := 0
				if enabled {
					want = 1
				}

				if probes.validated != want {
					t.Fatalf("result validation called %d times, want %d", probes.validated, want)
				}
			})
		}
	}
}
