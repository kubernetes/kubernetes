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
	name        string
	log         *[]string
	numPrepared int
}

var _ framework.NominationExtensions = &recordingNominationExtensions{}

func (pl *recordingNominationExtensions) Name() string { return pl.name }

func (pl *recordingNominationExtensions) PreFilter(context.Context, fwk.CycleState, *v1.Pod, []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status) {
	return nil, nil
}

func (pl *recordingNominationExtensions) PreFilterExtensions() fwk.PreFilterExtensions { return nil }

// PrepareNomination returns "prepared <n> by <name>", counting its calls.
func (pl *recordingNominationExtensions) PrepareNomination(_ context.Context, _ fwk.CycleState, pod *v1.Pod, nodeName string) any {
	*pl.log = append(*pl.log, fmt.Sprintf("%s: prepare %s %s", pl.name, pod.Name, nodeName))
	pl.numPrepared++
	return fmt.Sprintf("prepared %d by %s", pl.numPrepared, pl.name)
}

func (pl *recordingNominationExtensions) AddNomination(_ context.Context, pod *v1.Pod, nodeName string, prepared any) framework.NominationChangeResult {
	*pl.log = append(*pl.log, fmt.Sprintf("%s: add %s %s %v", pl.name, pod.Name, nodeName, prepared))
	return framework.NominationChangeResult{}
}

// RemoveNomination returns one event per plugin, labeled with its name.
func (pl *recordingNominationExtensions) RemoveNomination(_ context.Context, pod *v1.Pod, nodeName string) framework.NominationChangeResult {
	*pl.log = append(*pl.log, fmt.Sprintf("%s: remove %s %s", pl.name, pod.Name, nodeName))
	return framework.NominationChangeResult{Events: []framework.NominationEvent{{Event: fwk.ClusterEvent{Resource: fwk.ResourceClaim, ActionType: fwk.Update, CustomLabel: pl.name}}}}
}

func TestNominationExtensions(t *testing.T) {
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "preemptor"}}

	for name, tc := range map[string]struct {
		gateEnabled    bool
		wantCalls      []string
		wantEvents     []string
		withoutPlugins bool
	}{
		"gate-disabled": {},
		"no-plugins": {
			gateEnabled:    true,
			withoutPlugins: true,
		},
		"with-plugins": {
			gateEnabled: true,
			wantEvents:  []string{"a", "b"},
			wantCalls: []string{
				"a: prepare preemptor node1", "b: prepare preemptor node1",
				"a: add preemptor node1 prepared 1 by a", "b: add preemptor node1 prepared 1 by b",
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

			done := f.RunNominationExtensionsPrepareNomination(ctx, nil, pod, "node1")
			defer done()
			var events []string
			for _, event := range append(f.RunNominationExtensionsAddNomination(ctx, pod, "node1"), f.RunNominationExtensionsRemoveNomination(ctx, pod, "node1")...) {
				events = append(events, event.Event.CustomLabel)
			}
			if diff := cmp.Diff(tc.wantEvents, events); diff != "" {
				t.Errorf("unexpected events (-want, +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantCalls, log); diff != "" {
				t.Errorf("unexpected calls (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestPreparedNominations(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAPreemption, true)
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var log []string
	pl := &recordingNominationExtensions{name: "a", log: &log}
	registry := Registry{pl.name: func(context.Context, runtime.Object, fwk.Handle) (fwk.Plugin, error) { return pl, nil }}
	profile := config.KubeSchedulerProfile{Plugins: &config.Plugins{PreFilter: config.PluginSet{Enabled: []config.Plugin{{Name: pl.name}}}}}
	f, err := newFrameworkWithQueueSortAndBind(ctx, registry, profile)
	if err != nil {
		t.Fatalf("create framework: %v", err)
	}
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p", UID: "p"}}

	// Only for the node and only once.
	done := f.RunNominationExtensionsPrepareNomination(ctx, nil, pod, "node1")
	f.RunNominationExtensionsAddNomination(ctx, pod, "node2")
	f.RunNominationExtensionsAddNomination(ctx, pod, "node1")
	f.RunNominationExtensionsAddNomination(ctx, pod, "node1")
	done()

	// Dropped by done.
	done = f.RunNominationExtensionsPrepareNomination(ctx, nil, pod, "node1")
	done()
	f.RunNominationExtensionsAddNomination(ctx, pod, "node1")

	// done of an older call keeps the results of a newer one.
	done = f.RunNominationExtensionsPrepareNomination(ctx, nil, pod, "node1")
	doneNewer := f.RunNominationExtensionsPrepareNomination(ctx, nil, pod, "node1")
	done()
	f.RunNominationExtensionsAddNomination(ctx, pod, "node1")
	doneNewer()

	want := []string{
		"a: prepare p node1",
		"a: add p node2 <nil>",
		"a: add p node1 prepared 1 by a",
		"a: add p node1 <nil>",
		"a: prepare p node1",
		"a: add p node1 <nil>",
		"a: prepare p node1",
		"a: prepare p node1",
		"a: add p node1 prepared 4 by a",
	}
	if diff := cmp.Diff(want, log); diff != "" {
		t.Errorf("unexpected calls (-want, +got):\n%s", diff)
	}
}
