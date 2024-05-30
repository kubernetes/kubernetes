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

package server

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/version"
	utilversion "k8s.io/apiserver/pkg/util/version"
	"k8s.io/sample-apiserver/pkg/apiserver"

	"github.com/stretchr/testify/assert"
)

func TestMapBinaryEffectiveVersionToKubeEffectiveVersion(t *testing.T) {
	wardleEffectiveVersion := utilversion.NewEffectiveVersion("1.2")
	defaultKubeEffectiveVersion := utilversion.DefaultKubeEffectiveVersion()

	testCases := []struct {
		desc                     string
		wardleEmulationVer       *version.Version
		expectedKubeEmulationVer *version.Version
	}{
		{
			desc:                     "1 version higher than kube binary",
			wardleEmulationVer:       version.MajorMinor(1, 2),
			expectedKubeEmulationVer: defaultKubeEffectiveVersion.BinaryVersion().AddMinor(1),
		},
		{
			desc:               "no mapping",
			wardleEmulationVer: version.MajorMinor(1, 10),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			registry := utilversion.NewComponentGlobalsRegistry()
			_ = registry.Register(apiserver.WardleComponentName, wardleEffectiveVersion, nil, true)
			_ = registry.Register(utilversion.DefaultKubeComponent, defaultKubeEffectiveVersion, nil, true)

			wardleEffectiveVersion.SetEmulationVersion(tc.wardleEmulationVer)
			err := mapWardleEffectiveVersionToKubeEffectiveVersion(registry)
			if tc.expectedKubeEmulationVer == nil {
				if err == nil {
					t.Fatal("expected error, no error found")
				}
			} else {
				assert.True(t, registry.EffectiveVersionFor(utilversion.DefaultKubeComponent).EmulationVersion().EqualTo(tc.expectedKubeEmulationVer))
			}
		})
	}
}
