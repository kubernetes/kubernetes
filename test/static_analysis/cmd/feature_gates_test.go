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

package cmd

import (
	"go/token"
	"log"
	"os"
	"testing"

	"k8s.io/apimachinery/pkg/util/errors"
)

func TestVerifyAlphabeticOrderInFeatureSpecMap(t *testing.T) {
	tests := []struct {
		name        string
		fileContent string
		pkgPrefix   string
		expectErr   bool
	}{
		{
			name: "ordered versioned specs",
			fileContent: `
package features

import (
	"k8s.io/apimachinery/pkg/util/version"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/component-base/featuregate"
)
var defaultVersionedKubernetesFeatureGates = map[featuregate.Feature]featuregate.VersionedSpecs{
	SELinuxMount: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
	},
	SupplementalGroupsPolicy: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
	},
	genericfeatures.AdmissionWebhookMatchConditions: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
	},
	genericfeatures.AggregatedDiscoveryEndpoint: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
	},
}
			`,
		},
		{
			name: "unordered versioned specs",
			fileContent: `
package features

import (
	"k8s.io/apimachinery/pkg/util/version"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/component-base/featuregate"
)
var defaultVersionedKubernetesFeatureGates = map[featuregate.Feature]featuregate.VersionedSpecs{
	SupplementalGroupsPolicy: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
	},
	SELinuxMount: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
	},
}
			`,
			expectErr: true,
		},
		{
			name: "package prefix versioned specs",
			fileContent: `
package features

import (
	"k8s.io/apimachinery/pkg/util/version"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/component-base/featuregate"
)
var defaultVersionedKubernetesFeatureGates = map[featuregate.Feature]featuregate.VersionedSpecs{
	SupplementalGroupsPolicy: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
	},
	SELinuxMount: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
	},
	genericfeatures.AdmissionWebhookMatchConditions: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
	},
	genericfeatures.AggregatedDiscoveryEndpoint: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
	},
}
			`,
			pkgPrefix: "genericfeatures",
		},
		{
			name: "split package",
			fileContent: `
package features

import (
	"k8s.io/apimachinery/pkg/util/version"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/component-base/featuregate"
)
var defaultVersionedKubernetesFeatureGates = map[featuregate.Feature]featuregate.VersionedSpecs{
	genericfeatures.AdmissionWebhookMatchConditions: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
	},
	SELinuxMount: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
	},
	SupplementalGroupsPolicy: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
	},
	genericfeatures.AggregatedDiscoveryEndpoint: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
	},
}
			`,
			expectErr: true,
		},
		{
			name: "ordered unversioned specs",
			fileContent: `
package features

import (
	"k8s.io/apimachinery/pkg/util/version"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/component-base/featuregate"
)
var defaultVersionedKubernetesFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{
	AppArmorFields: {Default: true, PreRelease: featuregate.Beta},
	ClusterTrustBundleProjection: {Default: false, PreRelease: featuregate.Alpha},
	CPUCFSQuotaPeriod: {Default: false, PreRelease: featuregate.Alpha},
}
			`,
		},
		{
			name: "unordered unversioned specs",
			fileContent: `
package features

import (
	"k8s.io/apimachinery/pkg/util/version"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/component-base/featuregate"
)
var defaultVersionedKubernetesFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{
	AppArmorFields: {Default: true, PreRelease: featuregate.Beta},
	CPUCFSQuotaPeriod: {Default: false, PreRelease: featuregate.Alpha},
	ClusterTrustBundleProjection: {Default: false, PreRelease: featuregate.Alpha},
}
			`,
			expectErr: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tmpfile := writeContentToTmpFile(t, "features.go", tc.fileContent)
			fset := token.NewFileSet()
			errs := []error{}
			err := verifyAlphabeticOrderInFeatureSpecMap(fset, tmpfile.Name(), tc.pkgPrefix, false)
			if err != nil {
				errs = append(errs, err)
			}
			err = verifyAlphabeticOrderInFeatureSpecMap(fset, tmpfile.Name(), tc.pkgPrefix, true)
			if err != nil {
				errs = append(errs, err)
			}
			if tc.expectErr {
				if len(errs) == 0 {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if len(errs) > 0 {
				t.Fatal(errors.NewAggregate(errs))
			}
		})
	}
}

func TestVerifyNoNewUnversionedFeatureSpec(t *testing.T) {
	oldFileContent := `
package features

import (
	"k8s.io/apimachinery/pkg/util/version"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/component-base/featuregate"
)
var defaultVersionedKubernetesFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{
	AppArmorFields: {Default: true, PreRelease: featuregate.Beta},
	CPUCFSQuotaPeriod: {Default: false, PreRelease: featuregate.Alpha},
	ClusterTrustBundleProjection: {Default: false, PreRelease: featuregate.Alpha},
}
			`
	tests := []struct {
		name        string
		fileContent string
		expectErr   bool
	}{
		{
			name:        "no change",
			fileContent: oldFileContent,
		},
		{
			name: "new feature added",
			fileContent: `
package features

import (
	"k8s.io/apimachinery/pkg/util/version"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/component-base/featuregate"
)
var defaultVersionedKubernetesFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{
	AppArmorFields: {Default: true, PreRelease: featuregate.Beta},
	CPUCFSQuotaPeriod: {Default: false, PreRelease: featuregate.Alpha},
	ClusterTrustBundleProjection: {Default: false, PreRelease: featuregate.Alpha},
	SELinuxMount: {Default: false, PreRelease: featuregate.Alpha},
}
			`,
			expectErr: true,
		},
		{
			name: "lifecycle of existing feature ok",
			fileContent: `
package features

import (
	"k8s.io/apimachinery/pkg/util/version"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/component-base/featuregate"
)
var defaultVersionedKubernetesFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{
	AppArmorFields: {Default: true, PreRelease: featuregate.Beta},
	CPUCFSQuotaPeriod: {Default: false, PreRelease: featuregate.Beta},
	ClusterTrustBundleProjection: {Default: false, PreRelease: featuregate.Alpha},
}
			`,
		},
		{
			name: "deleting existing feature ok",
			fileContent: `
package features

import (
	"k8s.io/apimachinery/pkg/util/version"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/component-base/featuregate"
)
var defaultVersionedKubernetesFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{
	AppArmorFields: {Default: true, PreRelease: featuregate.Beta},
	ClusterTrustBundleProjection: {Default: false, PreRelease: featuregate.Alpha},
}
			`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			oldFile := writeContentToTmpFile(t, "old_features.go", oldFileContent)
			newFile := writeContentToTmpFile(t, "new_features.go", tc.fileContent)
			err := verifyNoNewUnversionedFeatureSpec(newFile.Name(), oldFile.Name())
			if tc.expectErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}

func writeContentToTmpFile(t *testing.T, fileName, fileContent string) *os.File {
	tmpfile, err := os.CreateTemp("", fileName)
	if err != nil {
		log.Fatal(err)
	}
	_, err = tmpfile.WriteString(fileContent)
	if err != nil {
		t.Fatal(err)
	}
	err = tmpfile.Close()
	if err != nil {
		t.Fatal(err)
	}
	return tmpfile
}
