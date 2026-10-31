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
	name              string
	notEligibleReason string
	log               *[]string
}

var _ framework.NominationExtensions = &recordingNominationExtensions{}

func (pl *recordingNominationExtensions) Name() string { return pl.name }

func (pl *recordingNominationExtensions) PreFilter(context.Context, fwk.CycleState, *v1.Pod, []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status) {
	return nil, nil
}

func (pl *recordingNominationExtensions) PreFilterExtensions() fwk.PreFilterExtensions { return nil }

func (pl *recordingNominationExtensions) AddNominatedPod(_ context.Context, _ fwk.CycleState, pod *v1.Pod, nodeName string, victims []*v1.Pod) {
	*pl.log = append(*pl.log, fmt.Sprintf("%s: add %s %s %d", pl.name, pod.Name, nodeName, len(victims)))
}

func (pl *recordingNominationExtensions) RemoveNominatedPod(_ context.Context, pod *v1.Pod, nodeName string, preemptionFailed bool) {
	*pl.log = append(*pl.log, fmt.Sprintf("%s: remove %s %s %v", pl.name, pod.Name, nodeName, preemptionFailed))
}

func (pl *recordingNominationExtensions) PodEligibleToPreempt(_ context.Context, pod *v1.Pod, nodeName string) (bool, string) {
	*pl.log = append(*pl.log, fmt.Sprintf("%s: eligible %s %s", pl.name, pod.Name, nodeName))
	return pl.notEligibleReason == "", pl.notEligibleReason
}

func TestNominationExtensions(t *testing.T) {
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "preemptor"}}
	victims := []*v1.Pod{{ObjectMeta: metav1.ObjectMeta{Name: "victim"}}}

	for name, tc := range map[string]struct {
		gateEnabled    bool
		notEligible    map[string]string
		wantEligible   bool
		wantReason     string
		wantCalls      []string
		withoutPlugins bool
	}{
		"gate-disabled": {
			wantEligible: true,
		},
		"no-plugins": {
			gateEnabled:    true,
			withoutPlugins: true,
			wantEligible:   true,
		},
		"eligible": {
			gateEnabled:  true,
			wantEligible: true,
			wantCalls: []string{
				"a: add preemptor node1 1", "b: add preemptor node1 1",
				"a: remove preemptor node1 true", "b: remove preemptor node1 true",
				"a: eligible preemptor node1", "b: eligible preemptor node1",
			},
		},
		"not-eligible": {
			gateEnabled:  true,
			notEligible:  map[string]string{"b": "still reclaiming"},
			wantEligible: false,
			wantReason:   "still reclaiming",
			wantCalls: []string{
				"a: add preemptor node1 1", "b: add preemptor node1 1",
				"a: remove preemptor node1 true", "b: remove preemptor node1 true",
				"a: eligible preemptor node1", "b: eligible preemptor node1",
			},
		},
		"stops-at-first-not-eligible": {
			gateEnabled:  true,
			notEligible:  map[string]string{"a": "reason a", "b": "reason b"},
			wantEligible: false,
			wantReason:   "reason a",
			wantCalls: []string{
				"a: add preemptor node1 1", "b: add preemptor node1 1",
				"a: remove preemptor node1 true", "b: remove preemptor node1 true",
				"a: eligible preemptor node1",
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
					{name: "b", notEligibleReason: tc.notEligible["b"], log: &log},
					{name: "a", notEligibleReason: tc.notEligible["a"], log: &log},
				} {
					registry[pl.name] = func(context.Context, runtime.Object, fwk.Handle) (fwk.Plugin, error) { return pl, nil }
					profile.Plugins.PreFilter.Enabled = append(profile.Plugins.PreFilter.Enabled, config.Plugin{Name: pl.name})
				}
			}
			f, err := newFrameworkWithQueueSortAndBind(ctx, registry, profile)
			if err != nil {
				t.Fatalf("create framework: %v", err)
			}

			f.RunNominationExtensionsAddNominatedPod(ctx, nil, pod, "node1", victims)
			f.RunNominationExtensionsRemoveNominatedPod(ctx, pod, "node1", true)
			eligible, reason := f.RunNominationExtensionsPodEligibleToPreempt(ctx, pod, "node1")
			if eligible != tc.wantEligible || reason != tc.wantReason {
				t.Errorf("PodEligibleToPreempt: got (%v, %q), want (%v, %q)", eligible, reason, tc.wantEligible, tc.wantReason)
			}
			if diff := cmp.Diff(tc.wantCalls, log); diff != "" {
				t.Errorf("unexpected calls (-want, +got):\n%s", diff)
			}
		})
	}
}
