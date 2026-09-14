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
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/pleg"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

type validatingProbeResults struct {
	results.Manager
	validated int
}

func (m *validatingProbeResults) IsCurrent(results.Update) bool {
	m.validated++
	return false
}

func TestMutableProbeResultValidationFeatureGate(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		for _, kind := range []kubecontainer.ProbeType{kubecontainer.LivenessProbe, kubecontainer.ReadinessProbe, kubecontainer.StartupProbe} {
			t.Run(fmt.Sprintf("enabled=%v/%v", enabled, kind), func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ContainerScopedProbes, true)
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutableContainerProbes, enabled)
				ctx := ktesting.Init(t)
				testKubelet := newTestKubelet(t, false)
				defer testKubelet.Cleanup()
				kl := testKubelet.kubelet
				cache := &validatingProbeResults{Manager: results.NewManager()}
				switch kind {
				case kubecontainer.LivenessProbe:
					kl.livenessManager = cache
				case kubecontainer.ReadinessProbe:
					kl.readinessManager = cache
				case kubecontainer.StartupProbe:
					kl.startupManager = cache
				}
				cache.Set(kubecontainer.ContainerID{Type: "test", ID: "container"}, results.Success, &v1.Pod{})
				kl.syncLoopIteration(ctx, make(chan kubetypes.PodUpdate), kl, make(chan time.Time), make(chan time.Time), make(chan *pleg.PodLifecycleEvent))
				want := 0
				if enabled {
					want = 1
				}
				if cache.validated != want {
					t.Fatalf("result validation called %d times, want %d", cache.validated, want)
				}
			})
		}
	}
}
