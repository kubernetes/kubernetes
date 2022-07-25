/*
Copyright 2020 The Kubernetes Authors.

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
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/volume"
)

func TestGetFullQualifiedPluginNameForVolume(t *testing.T) {
	var (
		fakePluginName          = "kubernetes.io/fakePlugin"
		fakeInlineCSIDriverName = "fake.inline.csi.driver"
		fakeCSIDriverName       = "fake.csi.driver"
	)

	testCase := []struct {
		name         string
		pluginName   string
		spec         *volume.Spec
		wantFullName string
	}{
		{
			name:         "get full qualified plugin name without volume spec",
			pluginName:   fakePluginName,
			spec:         nil,
			wantFullName: fakePluginName,
		},
		{
			name:         "get full qualified plugin name without using CSI plugin",
			pluginName:   fakePluginName,
			spec:         &volume.Spec{},
			wantFullName: fakePluginName,
		},
		{
			name:       "get full qualified plugin name with CSI ephemeral volume",
			pluginName: fakePluginName,
			spec: &volume.Spec{
				Volume: &v1.Volume{
					VolumeSource: v1.VolumeSource{
						CSI: &v1.CSIVolumeSource{
							Driver: fakeInlineCSIDriverName,
						},
					},
				},
			},
			wantFullName: fmt.Sprintf("%s:%s", fakePluginName, fakeInlineCSIDriverName),
		},
		{
			name:       "get full qualified plugin name with CSI PV",
			pluginName: fakePluginName,
			spec: &volume.Spec{
				PersistentVolume: &v1.PersistentVolume{
					Spec: v1.PersistentVolumeSpec{
						PersistentVolumeSource: v1.PersistentVolumeSource{
							CSI: &v1.CSIPersistentVolumeSource{
								Driver: fakeCSIDriverName,
							},
						},
					},
				},
			},
			wantFullName: fmt.Sprintf("%s:%s", fakePluginName, fakeCSIDriverName),
		},
	}

	for _, test := range testCase {
		t.Run(test.name, func(t *testing.T) {
			if fullPluginName := GetFullQualifiedPluginNameForVolume(test.pluginName, test.spec); fullPluginName != test.wantFullName {
				t.Errorf("Case name: %s, GetFullQualifiedPluginNameForVolume, pluginName:%s, spec: %v, return:%s, want:%s", test.name, test.pluginName, test.spec, fullPluginName, test.wantFullName)
			}
		})
	}
}
