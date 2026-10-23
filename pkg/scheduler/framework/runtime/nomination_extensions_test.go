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

package runtime

import (
	"context"
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// recordingNominationExtensions appends its calls to a log shared by all instances.
type recordingNominationExtensions struct {
	name string
	log  *[]string
}

var _ framework.NominationExtensions = &recordingNominationExtensions{}

func (pl *recordingNominationExtensions) Name() string { return pl.name }

func (pl *recordingNominationExtensions) PreFilter(context.Context, fwk.CycleState, *v1.Pod, []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status) {
	return nil, nil
}

func (pl *recordingNominationExtensions) PreFilterExtensions() fwk.PreFilterExtensions { return nil }

func (pl *recordingNominationExtensions) AddNominatedPod(_ context.Context, _ fwk.CycleState, pod *v1.Pod, nodeName string) {
	*pl.log = append(*pl.log, fmt.Sprintf("%s: add %s %s", pl.name, pod.Name, nodeName))
}

func (pl *recordingNominationExtensions) RemoveNominatedPod(_ context.Context, pod *v1.Pod, nodeName string) {
	*pl.log = append(*pl.log, fmt.Sprintf("%s: remove %s %s", pl.name, pod.Name, nodeName))
}

func TestNominationExtensions(t *testing.T) {
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "preemptor"}}

	for name, tc := range map[string]struct {
		gateEnabled    bool
		wantCalls      []string
		withoutPlugins bool
	}{
		"gate-disabled": {},
		"no-plugins": {
			gateEnabled:    true,
			withoutPlugins: true,
		},
		"with-plugins": {
			gateEnabled: true,
			wantCalls: []string{
				"a: add preemptor node1", "b: add preemptor node1",
				"a: remove preemptor node1", "b: remove preemptor node1",
			},
		},
	} {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAPreemption, tc.gateEnabled)
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			var log []string
			registry := Registry{}
			profile := config.KubeSchedulerProfile{Plugins: &config.Plugins{}}
			if !tc.withoutPlugins {
				// Registered out of order to check that calls are sorted by name.
				for _, pl := range []*recordingNominationExtensions{
					{name: "b", log: &log},
					{name: "a", log: &log},
				} {
					registry[pl.name] = func(context.Context, runtime.Object, fwk.Handle) (fwk.Plugin, error) { return pl, nil }
					profile.Plugins.PreFilter.Enabled = append(profile.Plugins.PreFilter.Enabled, config.Plugin{Name: pl.name})
				}
			}
			f, err := newFrameworkWithQueueSortAndBind(ctx, registry, profile)
			if err != nil {
				t.Fatalf("create framework: %v", err)
			}

			f.RunNominationExtensionsAddNominatedPod(ctx, nil, pod, "node1")
			f.RunNominationExtensionsRemoveNominatedPod(ctx, pod, "node1")
			if diff := cmp.Diff(tc.wantCalls, log); diff != "" {
				t.Errorf("unexpected calls (-want, +got):\n%s", diff)
			}
		})
	}
}
