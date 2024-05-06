/*
Copyright 2022 The Kubernetes Authors.

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

package v1

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2/ktesting"
	v1 "k8s.io/kube-scheduler/config/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/utils/ptr"
)

func TestApplyFeatureGates(t *testing.T) {
	tests := []struct {
		name       string
		features   map[featuregate.Feature]bool
		wantConfig *v1.Plugins
	}{
		{
			name: "Feature gate DynamicResourceAllocation disabled",
			features: map[featuregate.Feature]bool{
				features.DynamicResourceAllocation: false,
			},
			wantConfig: &v1.Plugins{
				MultiPoint: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: names.SchedulingGates},
						{Name: names.PrioritySort},
						{Name: names.NodeUnschedulable},
						{Name: names.NodeName},
						{Name: names.TaintToleration, Weight: ptr.To[int32](3)},
						{Name: names.NodeAffinity, Weight: ptr.To[int32](2)},
						{Name: names.NodePorts},
						{Name: names.NodeResourcesFit, Weight: ptr.To[int32](1)},
						{Name: names.VolumeRestrictions},
						{Name: names.EBSLimits},
						{Name: names.GCEPDLimits},
						{Name: names.NodeVolumeLimits},
						{Name: names.AzureDiskLimits},
						{Name: names.VolumeBinding},
						{Name: names.VolumeZone},
						{Name: names.PodTopologySpread, Weight: ptr.To[int32](2)},
						{Name: names.InterPodAffinity, Weight: ptr.To[int32](2)},
						{Name: names.DefaultPreemption},
						{Name: names.NodeResourcesBalancedAllocation, Weight: ptr.To[int32](1)},
						{Name: names.ImageLocality, Weight: ptr.To[int32](1)},
						{Name: names.DefaultBinder},
					},
				},
			},
		},
		{
			name: "Feature gate DynamicResourceAllocation enabled",
			features: map[featuregate.Feature]bool{
				features.DynamicResourceAllocation: true,
			},
			wantConfig: &v1.Plugins{
				MultiPoint: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: names.SchedulingGates},
						{Name: names.PrioritySort},
						{Name: names.NodeUnschedulable},
						{Name: names.NodeName},
						{Name: names.TaintToleration, Weight: ptr.To[int32](3)},
						{Name: names.NodeAffinity, Weight: ptr.To[int32](2)},
						{Name: names.NodePorts},
						{Name: names.NodeResourcesFit, Weight: ptr.To[int32](1)},
						{Name: names.VolumeRestrictions},
						{Name: names.EBSLimits},
						{Name: names.GCEPDLimits},
						{Name: names.NodeVolumeLimits},
						{Name: names.AzureDiskLimits},
						{Name: names.VolumeBinding},
						{Name: names.VolumeZone},
						{Name: names.PodTopologySpread, Weight: ptr.To[int32](2)},
						{Name: names.InterPodAffinity, Weight: ptr.To[int32](2)},
						{Name: names.DynamicResources},
						{Name: names.DefaultPreemption},
						{Name: names.NodeResourcesBalancedAllocation, Weight: ptr.To[int32](1)},
						{Name: names.ImageLocality, Weight: ptr.To[int32](1)},
						{Name: names.DefaultBinder},
					},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for k, v := range test.features {
				featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, k, v)
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
		customPlugins   *v1.Plugins
		defaultPlugins  *v1.Plugins
		expectedPlugins *v1.Plugins
	}{
		{
			name: "AppendCustomPlugin",
			customPlugins: &v1.Plugins{
				Filter: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "CustomPlugin"},
					},
				},
			},
			defaultPlugins: &v1.Plugins{
				Filter: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
					},
				},
			},
			expectedPlugins: &v1.Plugins{
				Filter: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
						{Name: "CustomPlugin"},
					},
				},
			},
		},
		{
			name: "InsertAfterDefaultPlugins2",
			customPlugins: &v1.Plugins{
				Filter: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "CustomPlugin"},
						{Name: "DefaultPlugin2"},
					},
					Disabled: []v1.Plugin{
						{Name: "DefaultPlugin2"},
					},
				},
			},
			defaultPlugins: &v1.Plugins{
				Filter: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
					},
				},
			},
			expectedPlugins: &v1.Plugins{
				Filter: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "DefaultPlugin1"},
						{Name: "CustomPlugin"},
						{Name: "DefaultPlugin2"},
					},
					Disabled: []v1.Plugin{
						{Name: "DefaultPlugin2"},
					},
				},
			},
		},
		{
			name: "InsertBeforeAllPlugins",
			customPlugins: &v1.Plugins{
				Filter: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "CustomPlugin"},
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
					},
					Disabled: []v1.Plugin{
						{Name: "*"},
					},
				},
			},
			defaultPlugins: &v1.Plugins{
				Filter: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
					},
				},
			},
			expectedPlugins: &v1.Plugins{
				Filter: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "CustomPlugin"},
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
					},
					Disabled: []v1.Plugin{
						{Name: "*"},
					},
				},
			},
		},
		{
			name: "ReorderDefaultPlugins",
			customPlugins: &v1.Plugins{
				Filter: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "DefaultPlugin2"},
						{Name: "DefaultPlugin1"},
					},
					Disabled: []v1.Plugin{
						{Name: "*"},
					},
				},
			},
			defaultPlugins: &v1.Plugins{
				Filter: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
					},
				},
			},
			expectedPlugins: &v1.Plugins{
				Filter: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "DefaultPlugin2"},
						{Name: "DefaultPlugin1"},
					},
					Disabled: []v1.Plugin{
						{Name: "*"},
					},
				},
			},
		},
		{
			name:          "ApplyNilCustomPlugin",
			customPlugins: nil,
			defaultPlugins: &v1.Plugins{
				Filter: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
					},
				},
			},
			expectedPlugins: &v1.Plugins{
				Filter: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
					},
				},
			},
		},
		{
			name: "CustomPluginOverrideDefaultPlugin",
			customPlugins: &v1.Plugins{
				Filter: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "Plugin1", Weight: ptr.To[int32](2)},
						{Name: "Plugin3", Weight: ptr.To[int32](3)},
					},
				},
			},
			defaultPlugins: &v1.Plugins{
				Filter: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "Plugin1"},
						{Name: "Plugin2"},
						{Name: "Plugin3"},
					},
				},
			},
			expectedPlugins: &v1.Plugins{
				Filter: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "Plugin1", Weight: ptr.To[int32](2)},
						{Name: "Plugin2"},
						{Name: "Plugin3", Weight: ptr.To[int32](3)},
					},
				},
			},
		},
		{
			name: "OrderPreserveAfterOverride",
			customPlugins: &v1.Plugins{
				Filter: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "Plugin2", Weight: ptr.To[int32](2)},
						{Name: "Plugin1", Weight: ptr.To[int32](1)},
					},
				},
			},
			defaultPlugins: &v1.Plugins{
				Filter: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "Plugin1"},
						{Name: "Plugin2"},
						{Name: "Plugin3"},
					},
				},
			},
			expectedPlugins: &v1.Plugins{
				Filter: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "Plugin1", Weight: ptr.To[int32](1)},
						{Name: "Plugin2", Weight: ptr.To[int32](2)},
						{Name: "Plugin3"},
					},
				},
			},
		},
		{
			name: "RepeatedCustomPlugin",
			customPlugins: &v1.Plugins{
				Filter: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "Plugin1"},
						{Name: "Plugin2", Weight: ptr.To[int32](2)},
						{Name: "Plugin3"},
						{Name: "Plugin2", Weight: ptr.To[int32](4)},
					},
				},
			},
			defaultPlugins: &v1.Plugins{
				Filter: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "Plugin1"},
						{Name: "Plugin2"},
						{Name: "Plugin3"},
					},
				},
			},
			expectedPlugins: &v1.Plugins{
				Filter: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "Plugin1"},
						{Name: "Plugin2", Weight: ptr.To[int32](4)},
						{Name: "Plugin3"},
						{Name: "Plugin2", Weight: ptr.To[int32](2)},
					},
				},
			},
		},
		{
			name: "Append custom MultiPoint plugin",
			customPlugins: &v1.Plugins{
				MultiPoint: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "CustomPlugin"},
					},
				},
			},
			defaultPlugins: &v1.Plugins{
				MultiPoint: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
					},
				},
			},
			expectedPlugins: &v1.Plugins{
				MultiPoint: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
						{Name: "CustomPlugin"},
					},
				},
			},
		},
		{
			name: "Append disabled Multipoint plugins",
			customPlugins: &v1.Plugins{
				MultiPoint: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "CustomPlugin"},
						{Name: "CustomPlugin2"},
					},
					Disabled: []v1.Plugin{
						{Name: "DefaultPlugin2"},
					},
				},
				Score: v1.PluginSet{
					Disabled: []v1.Plugin{
						{Name: "CustomPlugin2"},
					},
				},
			},
			defaultPlugins: &v1.Plugins{
				MultiPoint: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
					},
				},
			},
			expectedPlugins: &v1.Plugins{
				MultiPoint: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "DefaultPlugin1"},
						{Name: "CustomPlugin"},
						{Name: "CustomPlugin2"},
					},
					Disabled: []v1.Plugin{
						{Name: "DefaultPlugin2"},
					},
				},
				Score: v1.PluginSet{
					Disabled: []v1.Plugin{
						{Name: "CustomPlugin2"},
					},
				},
			},
		},
		{
			name: "override default MultiPoint plugins with custom value",
			customPlugins: &v1.Plugins{
				MultiPoint: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "DefaultPlugin", Weight: ptr.To[int32](5)},
					},
				},
			},
			defaultPlugins: &v1.Plugins{
				MultiPoint: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "DefaultPlugin"},
					},
				},
			},
			expectedPlugins: &v1.Plugins{
				MultiPoint: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "DefaultPlugin", Weight: ptr.To[int32](5)},
					},
				},
			},
		},
		{
			name: "disabled MultiPoint plugin in default set",
			defaultPlugins: &v1.Plugins{
				MultiPoint: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "DefaultPlugin"},
					},
					Disabled: []v1.Plugin{
						{Name: "DefaultPlugin2"},
					},
				},
			},
			customPlugins: &v1.Plugins{
				MultiPoint: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "CustomPlugin"},
					},
				},
			},
			expectedPlugins: &v1.Plugins{
				MultiPoint: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "DefaultPlugin"},
						{Name: "CustomPlugin"},
					},
					Disabled: []v1.Plugin{
						{Name: "DefaultPlugin2"},
					},
				},
			},
		},
		{
			name: "disabled MultiPoint plugin in default set for specific extension point",
			defaultPlugins: &v1.Plugins{
				MultiPoint: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "DefaultPlugin"},
					},
				},
				Score: v1.PluginSet{
					Disabled: []v1.Plugin{
						{Name: "DefaultPlugin2"},
					},
				},
			},
			customPlugins: &v1.Plugins{
				MultiPoint: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "CustomPlugin"},
					},
				},
			},
			expectedPlugins: &v1.Plugins{
				MultiPoint: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "DefaultPlugin"},
						{Name: "CustomPlugin"},
					},
				},
				Score: v1.PluginSet{
					Disabled: []v1.Plugin{
						{Name: "DefaultPlugin2"},
					},
				},
			},
		},
		{
			name: "multipoint with only disabled gets merged",
			defaultPlugins: &v1.Plugins{
				MultiPoint: v1.PluginSet{
					Enabled: []v1.Plugin{
						{Name: "DefaultPlugin"},
					},
				},
			},
			customPlugins: &v1.Plugins{
				MultiPoint: v1.PluginSet{
					Disabled: []v1.Plugin{
						{Name: "DefaultPlugin"},
					},
				},
			},
			expectedPlugins: &v1.Plugins{
				MultiPoint: v1.PluginSet{
					Disabled: []v1.Plugin{
						{Name: "DefaultPlugin"},
					},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			gotPlugins := mergePlugins(logger, test.defaultPlugins, test.customPlugins)
			if d := cmp.Diff(test.expectedPlugins, gotPlugins); d != "" {
				t.Fatalf("plugins mismatch (-want +got):\n%s", d)
			}
		})
	}
}
