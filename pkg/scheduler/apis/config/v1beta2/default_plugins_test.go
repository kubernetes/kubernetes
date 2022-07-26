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

package v1beta2

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kube-scheduler/config/v1beta2"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/utils/pointer"
)

func TestApplyFeatureGates(t *testing.T) {
	tests := []struct {
		name       string
		features   map[featuregate.Feature]bool
		wantConfig *v1beta2.Plugins
	}{
		{
			name: "Feature gates disabled",
			wantConfig: &v1beta2.Plugins{
				QueueSort: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: names.PrioritySort},
					},
				},
				PreFilter: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: names.NodeResourcesFit},
						{Name: names.NodePorts},
						{Name: names.VolumeRestrictions},
						{Name: names.PodTopologySpread},
						{Name: names.InterPodAffinity},
						{Name: names.VolumeBinding},
						{Name: names.NodeAffinity},
					},
				},
				Filter: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: names.NodeUnschedulable},
						{Name: names.NodeName},
						{Name: names.TaintToleration},
						{Name: names.NodeAffinity},
						{Name: names.NodePorts},
						{Name: names.NodeResourcesFit},
						{Name: names.VolumeRestrictions},
						{Name: names.EBSLimits},
						{Name: names.GCEPDLimits},
						{Name: names.NodeVolumeLimits},
						{Name: names.AzureDiskLimits},
						{Name: names.VolumeBinding},
						{Name: names.VolumeZone},
						{Name: names.PodTopologySpread},
						{Name: names.InterPodAffinity},
					},
				},
				PostFilter: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: names.DefaultPreemption},
					},
				},
				PreScore: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: names.InterPodAffinity},
						{Name: names.PodTopologySpread},
						{Name: names.TaintToleration},
						{Name: names.NodeAffinity},
					},
				},
				Score: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: names.NodeResourcesBalancedAllocation, Weight: pointer.Int32(1)},
						{Name: names.ImageLocality, Weight: pointer.Int32(1)},
						{Name: names.InterPodAffinity, Weight: pointer.Int32(1)},
						{Name: names.NodeResourcesFit, Weight: pointer.Int32(1)},
						{Name: names.NodeAffinity, Weight: pointer.Int32(1)},
						{Name: names.PodTopologySpread, Weight: pointer.Int32(2)},
						{Name: names.TaintToleration, Weight: pointer.Int32(1)},
					},
				},
				Reserve: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: names.VolumeBinding},
					},
				},
				PreBind: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: names.VolumeBinding},
					},
				},
				Bind: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: names.DefaultBinder},
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

			gotConfig := getDefaultPlugins()
			if diff := cmp.Diff(test.wantConfig, gotConfig); diff != "" {
				t.Errorf("unexpected config diff (-want, +got): %s", diff)
			}
		})
	}
}

func TestMergePlugins(t *testing.T) {
	tests := []struct {
		name            string
		customPlugins   *v1beta2.Plugins
		defaultPlugins  *v1beta2.Plugins
		expectedPlugins *v1beta2.Plugins
	}{
		{
			name: "AppendCustomPlugin",
			customPlugins: &v1beta2.Plugins{
				Filter: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: "CustomPlugin"},
					},
				},
			},
			defaultPlugins: &v1beta2.Plugins{
				Filter: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
					},
				},
			},
			expectedPlugins: &v1beta2.Plugins{
				Filter: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
						{Name: "CustomPlugin"},
					},
				},
			},
		},
		{
			name: "InsertAfterDefaultPlugins2",
			customPlugins: &v1beta2.Plugins{
				Filter: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: "CustomPlugin"},
						{Name: "DefaultPlugin2"},
					},
					Disabled: []v1beta2.Plugin{
						{Name: "DefaultPlugin2"},
					},
				},
			},
			defaultPlugins: &v1beta2.Plugins{
				Filter: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
					},
				},
			},
			expectedPlugins: &v1beta2.Plugins{
				Filter: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: "DefaultPlugin1"},
						{Name: "CustomPlugin"},
						{Name: "DefaultPlugin2"},
					},
				},
			},
		},
		{
			name: "InsertBeforeAllPlugins",
			customPlugins: &v1beta2.Plugins{
				Filter: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: "CustomPlugin"},
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
					},
					Disabled: []v1beta2.Plugin{
						{Name: "*"},
					},
				},
			},
			defaultPlugins: &v1beta2.Plugins{
				Filter: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
					},
				},
			},
			expectedPlugins: &v1beta2.Plugins{
				Filter: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: "CustomPlugin"},
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
					},
				},
			},
		},
		{
			name: "ReorderDefaultPlugins",
			customPlugins: &v1beta2.Plugins{
				Filter: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: "DefaultPlugin2"},
						{Name: "DefaultPlugin1"},
					},
					Disabled: []v1beta2.Plugin{
						{Name: "*"},
					},
				},
			},
			defaultPlugins: &v1beta2.Plugins{
				Filter: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
					},
				},
			},
			expectedPlugins: &v1beta2.Plugins{
				Filter: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: "DefaultPlugin2"},
						{Name: "DefaultPlugin1"},
					},
				},
			},
		},
		{
			name:          "ApplyNilCustomPlugin",
			customPlugins: nil,
			defaultPlugins: &v1beta2.Plugins{
				Filter: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
					},
				},
			},
			expectedPlugins: &v1beta2.Plugins{
				Filter: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
					},
				},
			},
		},
		{
			name: "CustomPluginOverrideDefaultPlugin",
			customPlugins: &v1beta2.Plugins{
				Filter: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: "Plugin1", Weight: pointer.Int32(2)},
						{Name: "Plugin3", Weight: pointer.Int32(3)},
					},
				},
			},
			defaultPlugins: &v1beta2.Plugins{
				Filter: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: "Plugin1"},
						{Name: "Plugin2"},
						{Name: "Plugin3"},
					},
				},
			},
			expectedPlugins: &v1beta2.Plugins{
				Filter: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: "Plugin1", Weight: pointer.Int32(2)},
						{Name: "Plugin2"},
						{Name: "Plugin3", Weight: pointer.Int32(3)},
					},
				},
			},
		},
		{
			name: "OrderPreserveAfterOverride",
			customPlugins: &v1beta2.Plugins{
				Filter: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: "Plugin2", Weight: pointer.Int32(2)},
						{Name: "Plugin1", Weight: pointer.Int32(1)},
					},
				},
			},
			defaultPlugins: &v1beta2.Plugins{
				Filter: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: "Plugin1"},
						{Name: "Plugin2"},
						{Name: "Plugin3"},
					},
				},
			},
			expectedPlugins: &v1beta2.Plugins{
				Filter: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: "Plugin1", Weight: pointer.Int32(1)},
						{Name: "Plugin2", Weight: pointer.Int32(2)},
						{Name: "Plugin3"},
					},
				},
			},
		},
		{
			name: "RepeatedCustomPlugin",
			customPlugins: &v1beta2.Plugins{
				Filter: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: "Plugin1"},
						{Name: "Plugin2", Weight: pointer.Int32(2)},
						{Name: "Plugin3"},
						{Name: "Plugin2", Weight: pointer.Int32(4)},
					},
				},
			},
			defaultPlugins: &v1beta2.Plugins{
				Filter: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: "Plugin1"},
						{Name: "Plugin2"},
						{Name: "Plugin3"},
					},
				},
			},
			expectedPlugins: &v1beta2.Plugins{
				Filter: v1beta2.PluginSet{
					Enabled: []v1beta2.Plugin{
						{Name: "Plugin1"},
						{Name: "Plugin2", Weight: pointer.Int32(4)},
						{Name: "Plugin3"},
						{Name: "Plugin2", Weight: pointer.Int32(2)},
					},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			test.defaultPlugins = mergePlugins(test.defaultPlugins, test.customPlugins)
			if d := cmp.Diff(test.expectedPlugins, test.defaultPlugins); d != "" {
				t.Fatalf("plugins mismatch (-want +got):\n%s", d)
			}
		})
	}
}
