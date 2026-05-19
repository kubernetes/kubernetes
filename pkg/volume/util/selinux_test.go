/*
Copyright 2024 The Kubernetes Authors.

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

package util

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/utils/ptr"
)

func TestGetMountSELinuxLabel(t *testing.T) {
	pvRWOP := &v1.PersistentVolume{
		Spec: v1.PersistentVolumeSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOncePod},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				HostPath: &v1.HostPathVolumeSource{},
			},
		},
	}
	pvRWX := &v1.PersistentVolume{
		Spec: v1.PersistentVolumeSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteMany},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				HostPath: &v1.HostPathVolumeSource{},
			},
		},
	}

	seLinuxOpts1 := v1.SELinuxOptions{
		Level: "s0:c123,c456",
	}
	seLinuxOpts2 := v1.SELinuxOptions{
		Level: "s0:c234,c567",
	}
	seLinuxOpts3 := v1.SELinuxOptions{
		Level: "s0:c345,c678",
	}
	label1 := "system_u:object_r:container_file_t:s0:c123,c456"

	tests := []struct {
		name                  string
		featureGates          []featuregate.Feature // SELinuxMountReadWriteOncePod is always enabled
		pluginSupportsSELinux bool
		volume                *volume.Spec
		podSecurityContext    *v1.PodSecurityContext
		seLinuxOptions        []*v1.SELinuxOptions
		expectError           bool
		expectedInfo          SELinuxLabelInfo
	}{
		// Tests with no labels
		{
			name:                  "no label, no changePolicy",
			featureGates:          nil,
			pluginSupportsSELinux: true,
			volume:                &volume.Spec{PersistentVolume: pvRWOP},
			podSecurityContext:    nil,
			seLinuxOptions:        nil,
			expectError:           false,
			expectedInfo: SELinuxLabelInfo{
				SELinuxMountLabel:                 "", // no SELinuxOptions + the default policy is recursive
				SELinuxProcessLabel:               "", // no SELinuxOptions
				PluginSupportsSELinuxContextMount: true,
			},
		},
		{
			name:                  "no label, Recursive change policy",
			featureGates:          nil,
			pluginSupportsSELinux: true,
			volume:                &volume.Spec{PersistentVolume: pvRWOP},
			podSecurityContext:    &v1.PodSecurityContext{SELinuxChangePolicy: ptr.To(v1.SELinuxChangePolicyRecursive)},
			seLinuxOptions:        nil,
			expectError:           false,
			expectedInfo: SELinuxLabelInfo{
				SELinuxMountLabel:                 "", // no SELinuxOptions + recursive policy
				SELinuxProcessLabel:               "", // SELinuxOptions
				PluginSupportsSELinuxContextMount: true,
			},
		},
		// Tests with one label and RWOP volume
		{
			name:                  "one label, Recursive change policy, no feature gate",
			featureGates:          nil,
			pluginSupportsSELinux: true,
			volume:                &volume.Spec{PersistentVolume: pvRWOP},
			podSecurityContext:    &v1.PodSecurityContext{SELinuxChangePolicy: ptr.To(v1.SELinuxChangePolicyRecursive)},
			seLinuxOptions:        []*v1.SELinuxOptions{&seLinuxOpts1},
			expectError:           false,
			expectedInfo: SELinuxLabelInfo{
				SELinuxMountLabel:                 label1, // Recursive policy is not observed when SELinuxChangePolicy is off
				SELinuxProcessLabel:               label1, // Pod has a label assigned
				PluginSupportsSELinuxContextMount: true,
			},
		},
		{
			name:                  "one label, Recursive change policy, SELinuxChangePolicy",
			featureGates:          []featuregate.Feature{features.SELinuxChangePolicy},
			pluginSupportsSELinux: true,
			volume:                &volume.Spec{PersistentVolume: pvRWOP},
			podSecurityContext:    &v1.PodSecurityContext{SELinuxChangePolicy: ptr.To(v1.SELinuxChangePolicyRecursive)},
			seLinuxOptions:        []*v1.SELinuxOptions{&seLinuxOpts1},
			expectError:           false,
			expectedInfo: SELinuxLabelInfo{
				SELinuxMountLabel:                 "",     // Recursive policy is effective with SELinuxChangePolicy, affects RWOP too.
				SELinuxProcessLabel:               label1, // Pod has a label assigned
				PluginSupportsSELinuxContextMount: true,
			},
		},
		{
			name:                  "one label, no policy",
			featureGates:          nil,
			pluginSupportsSELinux: true,
			volume:                &volume.Spec{PersistentVolume: pvRWOP},
			podSecurityContext:    nil,
			seLinuxOptions:        []*v1.SELinuxOptions{&seLinuxOpts1},
			expectError:           false,
			expectedInfo: SELinuxLabelInfo{
				SELinuxMountLabel:                 label1, // The default policy is MountOption
				SELinuxProcessLabel:               label1, // Pod has a label assigned
				PluginSupportsSELinuxContextMount: true,
			},
		},
		{
			name:                  "one label, MountOption policy",
			featureGates:          nil,
			pluginSupportsSELinux: true,
			volume:                &volume.Spec{PersistentVolume: pvRWOP},
			podSecurityContext:    &v1.PodSecurityContext{SELinuxChangePolicy: ptr.To(v1.SELinuxChangePolicyMountOption)},
			seLinuxOptions:        []*v1.SELinuxOptions{&seLinuxOpts1},
			expectError:           false,
			expectedInfo: SELinuxLabelInfo{
				SELinuxMountLabel:                 label1, // SELinuxChangePolicy feature is disabled, but the default policy is MountOption anyway
				SELinuxProcessLabel:               label1, // Pod has a label assigned
				PluginSupportsSELinuxContextMount: true,
			},
		},
		// Tests with RWX volume
		{
			name:                  "one label, no policy, RWX",
			featureGates:          nil,
			pluginSupportsSELinux: true,
			volume:                &volume.Spec{PersistentVolume: pvRWX},
			podSecurityContext:    nil,
			seLinuxOptions:        []*v1.SELinuxOptions{&seLinuxOpts1},
			expectError:           false,
			expectedInfo: SELinuxLabelInfo{
				SELinuxMountLabel:                 label1, // GetMountSELinuxLabel() does not check the access mode, it's up to the caller
				SELinuxProcessLabel:               label1, // Pod has a label assigned
				PluginSupportsSELinuxContextMount: true,
			},
		},
		{
			name:                  "one label, no policy, RWX, SELinuxChangePolicy",
			featureGates:          []featuregate.Feature{features.SELinuxChangePolicy},
			pluginSupportsSELinux: true,
			volume:                &volume.Spec{PersistentVolume: pvRWX},
			podSecurityContext:    nil,
			seLinuxOptions:        []*v1.SELinuxOptions{&seLinuxOpts1},
			expectError:           false,
			expectedInfo: SELinuxLabelInfo{
				SELinuxMountLabel:                 label1, //  GetMountSELinuxLabel() does not check the access mode, it's up to the caller
				SELinuxProcessLabel:               label1, // Pod has a label assigned
				PluginSupportsSELinuxContextMount: true,
			},
		},
		{
			name:                  "one label, MountOption policy, RWX, SELinuxChangePolicy",
			featureGates:          []featuregate.Feature{features.SELinuxChangePolicy},
			pluginSupportsSELinux: true,
			volume:                &volume.Spec{PersistentVolume: pvRWX},
			podSecurityContext:    &v1.PodSecurityContext{SELinuxChangePolicy: ptr.To(v1.SELinuxChangePolicyMountOption)},
			seLinuxOptions:        []*v1.SELinuxOptions{&seLinuxOpts1},
			expectError:           false,
			expectedInfo: SELinuxLabelInfo{
				SELinuxMountLabel:                 label1, // GetMountSELinuxLabel() does not check the access mode, it's up to the caller
				SELinuxProcessLabel:               label1, // Pod has a label assigned
				PluginSupportsSELinuxContextMount: true,
			},
		},
		{
			name:                  "one label, no policy, RWX, SELinuxMount",
			featureGates:          []featuregate.Feature{features.SELinuxChangePolicy, features.SELinuxMount},
			pluginSupportsSELinux: true,
			volume:                &volume.Spec{PersistentVolume: pvRWX},
			podSecurityContext:    nil,
			seLinuxOptions:        []*v1.SELinuxOptions{&seLinuxOpts1},
			expectError:           false,
			expectedInfo: SELinuxLabelInfo{
				SELinuxMountLabel:                 label1, // SELinuxMount FG + MountOption policy
				SELinuxProcessLabel:               label1, // Pod has a label assigned
				PluginSupportsSELinuxContextMount: true,
			},
		},
		// No plugin support
		{
			name:                  "one label, Recursive change policy, SELinuxChangePolicy, no plugin support",
			featureGates:          []featuregate.Feature{features.SELinuxChangePolicy},
			pluginSupportsSELinux: false,
			volume:                &volume.Spec{PersistentVolume: pvRWOP},
			podSecurityContext:    &v1.PodSecurityContext{SELinuxChangePolicy: ptr.To(v1.SELinuxChangePolicyRecursive)},
			seLinuxOptions:        []*v1.SELinuxOptions{&seLinuxOpts1},
			expectError:           false,
			expectedInfo: SELinuxLabelInfo{
				SELinuxMountLabel:                 "",     // No plugin support
				SELinuxProcessLabel:               label1, // Pod has a label assigned
				PluginSupportsSELinuxContextMount: false,
			},
		},
		{
			name:                  "one label, no policy, no plugin support",
			featureGates:          nil,
			pluginSupportsSELinux: false,
			volume:                &volume.Spec{PersistentVolume: pvRWOP},
			podSecurityContext:    nil,
			seLinuxOptions:        []*v1.SELinuxOptions{&seLinuxOpts1},
			expectError:           false,
			expectedInfo: SELinuxLabelInfo{
				SELinuxMountLabel:                 "",     // No plugin support
				SELinuxProcessLabel:               label1, // Pod has a label assigned
				PluginSupportsSELinuxContextMount: false,
			},
		},
		{
			name:                  "one label, MountOption policy, no plugin support",
			featureGates:          nil,
			pluginSupportsSELinux: false,
			volume:                &volume.Spec{PersistentVolume: pvRWOP},
			podSecurityContext:    &v1.PodSecurityContext{SELinuxChangePolicy: ptr.To(v1.SELinuxChangePolicyMountOption)},
			seLinuxOptions:        []*v1.SELinuxOptions{&seLinuxOpts1},
			expectError:           false,
			expectedInfo: SELinuxLabelInfo{
				SELinuxMountLabel:                 "",     // No plugin support
				SELinuxProcessLabel:               label1, // Pod has a label assigned
				PluginSupportsSELinuxContextMount: false,
			},
		},
		// Corner cases
		{
			name:                  "multiple same labels, no policy",
			featureGates:          nil,
			pluginSupportsSELinux: true,
			volume:                &volume.Spec{PersistentVolume: pvRWOP},
			podSecurityContext:    nil,
			seLinuxOptions:        []*v1.SELinuxOptions{&seLinuxOpts1, &seLinuxOpts1, &seLinuxOpts1, &seLinuxOpts1},
			expectError:           false,
			expectedInfo: SELinuxLabelInfo{
				SELinuxMountLabel:                 label1, // The default policy is MountOption
				SELinuxProcessLabel:               label1, // Pod has a label assigned
				PluginSupportsSELinuxContextMount: true,
			},
		},
		// Error cases
		{
			name:                  "multiple different labels, no policy",
			featureGates:          nil,
			pluginSupportsSELinux: true,
			volume:                &volume.Spec{PersistentVolume: pvRWOP},
			podSecurityContext:    nil,
			seLinuxOptions:        []*v1.SELinuxOptions{&seLinuxOpts1, &seLinuxOpts2, &seLinuxOpts3},
			expectError:           true,
			expectedInfo: SELinuxLabelInfo{
				SELinuxMountLabel:                 "",
				SELinuxProcessLabel:               "",
				PluginSupportsSELinuxContextMount: true,
			},
		},
		{
			name:                  "multiple different labels, Recursive policy",
			featureGates:          nil,
			pluginSupportsSELinux: true,
			volume:                &volume.Spec{PersistentVolume: pvRWOP},
			podSecurityContext:    &v1.PodSecurityContext{SELinuxChangePolicy: ptr.To(v1.SELinuxChangePolicyRecursive)},
			seLinuxOptions:        []*v1.SELinuxOptions{&seLinuxOpts1, &seLinuxOpts2, &seLinuxOpts3},
			expectError:           true,
			expectedInfo: SELinuxLabelInfo{
				SELinuxMountLabel:                 "",
				SELinuxProcessLabel:               "",
				PluginSupportsSELinuxContextMount: true,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			// Set feature gates for the test. *Disable* those that are not in tt.featureGates.
			allGates := []featuregate.Feature{features.SELinuxChangePolicy, features.SELinuxMount}
			enabledGates := sets.New(tt.featureGates...)
			for _, fg := range allGates {
				enable := enabledGates.Has(fg)
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, fg, enable)
			}
			seLinuxTranslator := NewFakeSELinuxLabelTranslator()
			pluginMgr, plugin := volumetesting.GetTestKubeletVolumePluginMgr(t)
			plugin.SupportsSELinux = tt.pluginSupportsSELinux

			// Act
			info, err := GetMountSELinuxLabel(tt.volume, tt.seLinuxOptions, tt.podSecurityContext, pluginMgr, seLinuxTranslator)

			// Assert
			if err != nil {
				if !tt.expectError {
					t.Errorf("GetMountSELinuxLabel() unexpected error: %v", err)
				}
				return
			}
			if tt.expectError {
				t.Errorf("GetMountSELinuxLabel() expected error, got none")
				return
			}

			if info != tt.expectedInfo {
				t.Errorf("GetMountSELinuxLabel() expected %+v, got %+v", tt.expectedInfo, info)
			}
		})
	}
}
