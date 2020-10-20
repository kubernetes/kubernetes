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

package algorithmprovider

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultpreemption"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/imagelocality"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/interpodaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodename"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeports"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodepreferavoidpods"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeunschedulable"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodevolumelimits"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/podtopologyspread"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/selectorspread"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/tainttoleration"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumebinding"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumerestrictions"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumezone"
)

func TestClusterAutoscalerProvider(t *testing.T) {
	wantConfig := &schedulerapi.Plugins{
		QueueSort: &schedulerapi.PluginSet{
			Enabled: []schedulerapi.Plugin{
				{Name: queuesort.Name},
			},
		},
		PreFilter: &schedulerapi.PluginSet{
			Enabled: []schedulerapi.Plugin{
				{Name: noderesources.FitName},
				{Name: nodeports.Name},
				{Name: podtopologyspread.Name},
				{Name: interpodaffinity.Name},
				{Name: volumebinding.Name},
			},
		},
		Filter: &schedulerapi.PluginSet{
			Enabled: []schedulerapi.Plugin{
				{Name: nodeunschedulable.Name},
				{Name: nodename.Name},
				{Name: tainttoleration.Name},
				{Name: nodeaffinity.Name},
				{Name: nodeports.Name},
				{Name: noderesources.FitName},
				{Name: volumerestrictions.Name},
				{Name: nodevolumelimits.EBSName},
				{Name: nodevolumelimits.GCEPDName},
				{Name: nodevolumelimits.CSIName},
				{Name: nodevolumelimits.AzureDiskName},
				{Name: volumebinding.Name},
				{Name: volumezone.Name},
				{Name: podtopologyspread.Name},
				{Name: interpodaffinity.Name},
			},
		},
		PostFilter: &schedulerapi.PluginSet{
			Enabled: []schedulerapi.Plugin{
				{Name: defaultpreemption.Name},
			},
		},
		PreScore: &schedulerapi.PluginSet{
			Enabled: []schedulerapi.Plugin{
				{Name: interpodaffinity.Name},
				{Name: podtopologyspread.Name},
				{Name: tainttoleration.Name},
			},
		},
		Score: &schedulerapi.PluginSet{
			Enabled: []schedulerapi.Plugin{
				{Name: noderesources.BalancedAllocationName, Weight: 1},
				{Name: imagelocality.Name, Weight: 1},
				{Name: interpodaffinity.Name, Weight: 1},
				{Name: noderesources.MostAllocatedName, Weight: 1},
				{Name: nodeaffinity.Name, Weight: 1},
				{Name: nodepreferavoidpods.Name, Weight: 10000},
				{Name: podtopologyspread.Name, Weight: 2},
				{Name: tainttoleration.Name, Weight: 1},
			},
		},
		Reserve: &schedulerapi.PluginSet{
			Enabled: []schedulerapi.Plugin{
				{Name: volumebinding.Name},
			},
		},
		PreBind: &schedulerapi.PluginSet{
			Enabled: []schedulerapi.Plugin{
				{Name: volumebinding.Name},
			},
		},
		Bind: &schedulerapi.PluginSet{
			Enabled: []schedulerapi.Plugin{
				{Name: defaultbinder.Name},
			},
		},
	}

	r := NewRegistry()
	gotConfig := r[ClusterAutoscalerProvider]
	if diff := cmp.Diff(wantConfig, gotConfig); diff != "" {
		t.Errorf("unexpected config diff (-want, +got): %s", diff)
	}
}

func TestApplyFeatureGates(t *testing.T) {
	tests := []struct {
		name       string
		features   map[featuregate.Feature]bool
		wantConfig *schedulerapi.Plugins
	}{
		{
			name: "Feature gates disabled",
			wantConfig: &schedulerapi.Plugins{
				QueueSort: &schedulerapi.PluginSet{
					Enabled: []schedulerapi.Plugin{
						{Name: queuesort.Name},
					},
				},
				PreFilter: &schedulerapi.PluginSet{
					Enabled: []schedulerapi.Plugin{
						{Name: noderesources.FitName},
						{Name: nodeports.Name},
						{Name: podtopologyspread.Name},
						{Name: interpodaffinity.Name},
						{Name: volumebinding.Name},
					},
				},
				Filter: &schedulerapi.PluginSet{
					Enabled: []schedulerapi.Plugin{
						{Name: nodeunschedulable.Name},
						{Name: nodename.Name},
						{Name: tainttoleration.Name},
						{Name: nodeaffinity.Name},
						{Name: nodeports.Name},
						{Name: noderesources.FitName},
						{Name: volumerestrictions.Name},
						{Name: nodevolumelimits.EBSName},
						{Name: nodevolumelimits.GCEPDName},
						{Name: nodevolumelimits.CSIName},
						{Name: nodevolumelimits.AzureDiskName},
						{Name: volumebinding.Name},
						{Name: volumezone.Name},
						{Name: podtopologyspread.Name},
						{Name: interpodaffinity.Name},
					},
				},
				PostFilter: &schedulerapi.PluginSet{
					Enabled: []schedulerapi.Plugin{
						{Name: defaultpreemption.Name},
					},
				},
				PreScore: &schedulerapi.PluginSet{
					Enabled: []schedulerapi.Plugin{
						{Name: interpodaffinity.Name},
						{Name: podtopologyspread.Name},
						{Name: tainttoleration.Name},
					},
				},
				Score: &schedulerapi.PluginSet{
					Enabled: []schedulerapi.Plugin{
						{Name: noderesources.BalancedAllocationName, Weight: 1},
						{Name: imagelocality.Name, Weight: 1},
						{Name: interpodaffinity.Name, Weight: 1},
						{Name: noderesources.LeastAllocatedName, Weight: 1},
						{Name: nodeaffinity.Name, Weight: 1},
						{Name: nodepreferavoidpods.Name, Weight: 10000},
						{Name: podtopologyspread.Name, Weight: 2},
						{Name: tainttoleration.Name, Weight: 1},
					},
				},
				Reserve: &schedulerapi.PluginSet{
					Enabled: []schedulerapi.Plugin{
						{Name: volumebinding.Name},
					},
				},
				PreBind: &schedulerapi.PluginSet{
					Enabled: []schedulerapi.Plugin{
						{Name: volumebinding.Name},
					},
				},
				Bind: &schedulerapi.PluginSet{
					Enabled: []schedulerapi.Plugin{
						{Name: defaultbinder.Name},
					},
				},
			},
		},
		{
			name: "DefaultPodTopologySpread disabled",
			features: map[featuregate.Feature]bool{
				features.DefaultPodTopologySpread: false,
			},
			wantConfig: &schedulerapi.Plugins{
				QueueSort: &schedulerapi.PluginSet{
					Enabled: []schedulerapi.Plugin{
						{Name: queuesort.Name},
					},
				},
				PreFilter: &schedulerapi.PluginSet{
					Enabled: []schedulerapi.Plugin{
						{Name: noderesources.FitName},
						{Name: nodeports.Name},
						{Name: podtopologyspread.Name},
						{Name: interpodaffinity.Name},
						{Name: volumebinding.Name},
					},
				},
				Filter: &schedulerapi.PluginSet{
					Enabled: []schedulerapi.Plugin{
						{Name: nodeunschedulable.Name},
						{Name: nodename.Name},
						{Name: tainttoleration.Name},
						{Name: nodeaffinity.Name},
						{Name: nodeports.Name},
						{Name: noderesources.FitName},
						{Name: volumerestrictions.Name},
						{Name: nodevolumelimits.EBSName},
						{Name: nodevolumelimits.GCEPDName},
						{Name: nodevolumelimits.CSIName},
						{Name: nodevolumelimits.AzureDiskName},
						{Name: volumebinding.Name},
						{Name: volumezone.Name},
						{Name: podtopologyspread.Name},
						{Name: interpodaffinity.Name},
					},
				},
				PostFilter: &schedulerapi.PluginSet{
					Enabled: []schedulerapi.Plugin{
						{Name: defaultpreemption.Name},
					},
				},
				PreScore: &schedulerapi.PluginSet{
					Enabled: []schedulerapi.Plugin{
						{Name: interpodaffinity.Name},
						{Name: podtopologyspread.Name},
						{Name: tainttoleration.Name},
						{Name: selectorspread.Name},
					},
				},
				Score: &schedulerapi.PluginSet{
					Enabled: []schedulerapi.Plugin{
						{Name: noderesources.BalancedAllocationName, Weight: 1},
						{Name: imagelocality.Name, Weight: 1},
						{Name: interpodaffinity.Name, Weight: 1},
						{Name: noderesources.LeastAllocatedName, Weight: 1},
						{Name: nodeaffinity.Name, Weight: 1},
						{Name: nodepreferavoidpods.Name, Weight: 10000},
						{Name: podtopologyspread.Name, Weight: 2},
						{Name: tainttoleration.Name, Weight: 1},
						{Name: selectorspread.Name, Weight: 1},
					},
				},
				Reserve: &schedulerapi.PluginSet{
					Enabled: []schedulerapi.Plugin{
						{Name: volumebinding.Name},
					},
				},
				PreBind: &schedulerapi.PluginSet{
					Enabled: []schedulerapi.Plugin{
						{Name: volumebinding.Name},
					},
				},
				Bind: &schedulerapi.PluginSet{
					Enabled: []schedulerapi.Plugin{
						{Name: defaultbinder.Name},
					},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for k, v := range test.features {
				defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, k, v)()
			}

			r := NewRegistry()
			gotConfig := r[schedulerapi.SchedulerDefaultProviderName]
			if diff := cmp.Diff(test.wantConfig, gotConfig); diff != "" {
				t.Errorf("unexpected config diff (-want, +got): %s", diff)
			}
		})
	}
}
