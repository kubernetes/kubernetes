/*
Copyright 2017 The Kubernetes Authors.

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

package defaults

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/priorities"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
)

func TestCopyAndReplace(t *testing.T) {
	testCases := []struct {
		set         sets.String
		replaceWhat string
		replaceWith string
		expected    sets.String
	}{
		{
			set:         sets.String{"A": sets.Empty{}, "B": sets.Empty{}},
			replaceWhat: "A",
			replaceWith: "C",
			expected:    sets.String{"B": sets.Empty{}, "C": sets.Empty{}},
		},
		{
			set:         sets.String{"A": sets.Empty{}, "B": sets.Empty{}},
			replaceWhat: "D",
			replaceWith: "C",
			expected:    sets.String{"A": sets.Empty{}, "B": sets.Empty{}},
		},
	}
	for _, testCase := range testCases {
		result := copyAndReplace(testCase.set, testCase.replaceWhat, testCase.replaceWith)
		if !result.Equal(testCase.expected) {
			t.Errorf("expected %v got %v", testCase.expected, result)
		}
	}
}

func TestDefaultPriorities(t *testing.T) {
	result := sets.NewString(
		priorities.SelectorSpreadPriority,
		priorities.InterPodAffinityPriority,
		priorities.LeastRequestedPriority,
		priorities.BalancedResourceAllocation,
		priorities.NodePreferAvoidPodsPriority,
		priorities.NodeAffinityPriority,
		priorities.TaintTolerationPriority,
		priorities.ImageLocalityPriority,
	)
	if expected := defaultPriorities(); !result.Equal(expected) {
		t.Errorf("expected %v got %v", expected, result)
	}
}

func TestDefaultPredicates(t *testing.T) {
	result := sets.NewString(
		predicates.NoVolumeZoneConflictPred,
		predicates.MaxEBSVolumeCountPred,
		predicates.MaxGCEPDVolumeCountPred,
		predicates.MaxAzureDiskVolumeCountPred,
		predicates.MaxCSIVolumeCountPred,
		predicates.MatchInterPodAffinityPred,
		predicates.NoDiskConflictPred,
		predicates.GeneralPred,
		predicates.PodToleratesNodeTaintsPred,
		predicates.CheckVolumeBindingPred,
		predicates.CheckNodeUnschedulablePred,
	)

	if expected := defaultPredicates(); !result.Equal(expected) {
		t.Errorf("expected %v got %v", expected, result)
	}
}

func TestCompatibility(t *testing.T) {
	// Add serialized versions of scheduler config that exercise available options to ensure compatibility between releases
	testcases := []struct {
		name        string
		provider    string
		wantPlugins map[string][]config.Plugin
	}{
		{
			name: "No Provider specified",
			wantPlugins: map[string][]config.Plugin{
				"PreFilterPlugin": {
					{Name: "NodeResourcesFit"},
					{Name: "NodePorts"},
					{Name: "InterPodAffinity"},
				},
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "NodeResourcesFit"},
					{Name: "NodeName"},
					{Name: "NodePorts"},
					{Name: "NodeAffinity"},
					{Name: "VolumeRestrictions"},
					{Name: "TaintToleration"},
					{Name: "EBSLimits"},
					{Name: "GCEPDLimits"},
					{Name: "NodeVolumeLimits"},
					{Name: "AzureDiskLimits"},
					{Name: "VolumeBinding"},
					{Name: "VolumeZone"},
					{Name: "InterPodAffinity"},
				},
				"PostFilterPlugin": {
					{Name: "InterPodAffinity"},
					{Name: "TaintToleration"},
				},
				"ScorePlugin": {
					{Name: "NodeResourcesBalancedAllocation", Weight: 1},
					{Name: "ImageLocality", Weight: 1},
					{Name: "InterPodAffinity", Weight: 1},
					{Name: "NodeResourcesLeastAllocated", Weight: 1},
					{Name: "NodeAffinity", Weight: 1},
					{Name: "NodePreferAvoidPods", Weight: 10000},
					{Name: "DefaultPodTopologySpread", Weight: 1},
					{Name: "TaintToleration", Weight: 1},
				},
			},
		},
		{
			name:     "DefaultProvider",
			provider: config.SchedulerDefaultProviderName,
			wantPlugins: map[string][]config.Plugin{
				"PreFilterPlugin": {
					{Name: "NodeResourcesFit"},
					{Name: "NodePorts"},
					{Name: "InterPodAffinity"},
				},
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "NodeResourcesFit"},
					{Name: "NodeName"},
					{Name: "NodePorts"},
					{Name: "NodeAffinity"},
					{Name: "VolumeRestrictions"},
					{Name: "TaintToleration"},
					{Name: "EBSLimits"},
					{Name: "GCEPDLimits"},
					{Name: "NodeVolumeLimits"},
					{Name: "AzureDiskLimits"},
					{Name: "VolumeBinding"},
					{Name: "VolumeZone"},
					{Name: "InterPodAffinity"},
				},
				"PostFilterPlugin": {
					{Name: "InterPodAffinity"},
					{Name: "TaintToleration"},
				},
				"ScorePlugin": {
					{Name: "NodeResourcesBalancedAllocation", Weight: 1},
					{Name: "ImageLocality", Weight: 1},
					{Name: "InterPodAffinity", Weight: 1},
					{Name: "NodeResourcesLeastAllocated", Weight: 1},
					{Name: "NodeAffinity", Weight: 1},
					{Name: "NodePreferAvoidPods", Weight: 10000},
					{Name: "DefaultPodTopologySpread", Weight: 1},
					{Name: "TaintToleration", Weight: 1},
				},
			},
		},
		{
			name:     "ClusterAutoscalerProvider",
			provider: ClusterAutoscalerProvider,
			wantPlugins: map[string][]config.Plugin{
				"PreFilterPlugin": {
					{Name: "NodeResourcesFit"},
					{Name: "NodePorts"},
					{Name: "InterPodAffinity"},
				},
				"FilterPlugin": {
					{Name: "NodeUnschedulable"},
					{Name: "NodeResourcesFit"},
					{Name: "NodeName"},
					{Name: "NodePorts"},
					{Name: "NodeAffinity"},
					{Name: "VolumeRestrictions"},
					{Name: "TaintToleration"},
					{Name: "EBSLimits"},
					{Name: "GCEPDLimits"},
					{Name: "NodeVolumeLimits"},
					{Name: "AzureDiskLimits"},
					{Name: "VolumeBinding"},
					{Name: "VolumeZone"},
					{Name: "InterPodAffinity"},
				},
				"PostFilterPlugin": {
					{Name: "InterPodAffinity"},
					{Name: "TaintToleration"},
				},
				"ScorePlugin": {
					{Name: "NodeResourcesBalancedAllocation", Weight: 1},
					{Name: "ImageLocality", Weight: 1},
					{Name: "InterPodAffinity", Weight: 1},
					{Name: "NodeResourcesMostAllocated", Weight: 1},
					{Name: "NodeAffinity", Weight: 1},
					{Name: "NodePreferAvoidPods", Weight: 10000},
					{Name: "DefaultPodTopologySpread", Weight: 1},
					{Name: "TaintToleration", Weight: 1},
				},
			},
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			var opts []scheduler.Option
			if len(tc.provider) != 0 {
				opts = append(opts, scheduler.WithAlgorithmSource(config.SchedulerAlgorithmSource{
					Provider: &tc.provider,
				}))
			}

			client := fake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactory(client, 0)

			sched, err := scheduler.New(
				client,
				informerFactory,
				informerFactory.Core().V1().Pods(),
				nil,
				make(chan struct{}),
				opts...,
			)

			if err != nil {
				t.Fatalf("Error constructing: %v", err)
			}

			gotPlugins := sched.Framework.ListPlugins()
			if diff := cmp.Diff(tc.wantPlugins, gotPlugins); diff != "" {
				t.Errorf("unexpected plugins diff (-want, +got): %s", diff)
			}
		})
	}

}
