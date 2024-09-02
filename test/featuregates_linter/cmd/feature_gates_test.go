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
	"fmt"
	"go/ast"
	"go/token"
	"log"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestVerifyAlphabeticOrder(t *testing.T) {
	tests := []struct {
		name      string
		keys      []string
		expectErr bool
	}{
		{
			name: "ordered versioned specs",
			keys: []string{
				"SchedulerQueueingHints", "SELinuxMount", "ServiceAccountTokenJTI",
				"genericfeatures.AdmissionWebhookMatchConditions",
				"genericfeatures.AggregatedDiscoveryEndpoint",
			},
		},
		{
			name: "unordered versioned specs",
			keys: []string{
				"SELinuxMount", "SchedulerQueueingHints", "ServiceAccountTokenJTI",
				"genericfeatures.AdmissionWebhookMatchConditions",
				"genericfeatures.AggregatedDiscoveryEndpoint",
			},
			expectErr: true,
		},
		{
			name: "unordered versioned specs with mixed pkg prefix",
			keys: []string{
				"genericfeatures.AdmissionWebhookMatchConditions",
				"SchedulerQueueingHints", "SELinuxMount", "ServiceAccountTokenJTI",
				"genericfeatures.AggregatedDiscoveryEndpoint",
			},
			expectErr: true,
		},
		{
			name: "unordered versioned specs with pkg prefix",
			keys: []string{
				"SchedulerQueueingHints", "SELinuxMount", "ServiceAccountTokenJTI",
				"genericfeatures.AggregatedDiscoveryEndpoint",
				"genericfeatures.AdmissionWebhookMatchConditions",
			},
			expectErr: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := verifyAlphabeticOrder(tc.keys, "")
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

func TestVerifyOrUpdateFeatureListUnversioned(t *testing.T) {
	featureListFileContent := `- name: AppArmorFields
  versionedSpecs:
  - default: true
    lockToDefault: false
    preRelease: Beta
    version: ""
- name: ClusterTrustBundleProjection
  versionedSpecs:
  - default: false
    lockToDefault: false
    preRelease: Alpha
    version: ""
- name: CPUCFSQuotaPeriod
  versionedSpecs:
  - default: false
    lockToDefault: false
    preRelease: Alpha
    version: ""
`
	tests := []struct {
		name                          string
		goFileContent                 string
		updatedFeatureListFileContent string
		expectVerifyErr               bool
		expectUpdateErr               bool
	}{
		{
			name: "no change",
			goFileContent: `
package features

import (
	"k8s.io/component-base/featuregate"
)
var defaultKubernetesFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{
	AppArmorFields: {Default: true, PreRelease: featuregate.Beta},
	CPUCFSQuotaPeriod: {Default: false, PreRelease: featuregate.Alpha},
	ClusterTrustBundleProjection: {Default: false, PreRelease: featuregate.Alpha},
}
var otherFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{
	AppArmorFields: {Default: true, PreRelease: featuregate.Beta},
}
`,
			updatedFeatureListFileContent: featureListFileContent,
		},
		{
			name: "same feature added twice with different lifecycle",
			goFileContent: `
package features

import (
	"k8s.io/component-base/featuregate"
)
var defaultKubernetesFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{
	AppArmorFields: {Default: true, PreRelease: featuregate.Beta},
	CPUCFSQuotaPeriod: {Default: false, PreRelease: featuregate.Alpha},
	ClusterTrustBundleProjection: {Default: false, PreRelease: featuregate.Alpha},
}
	var otherFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{
	AppArmorFields: {Default: true, PreRelease: featuregate.Alpha},
}
`,
			expectVerifyErr: true,
			expectUpdateErr: true,
		},
		{
			name: "new feature added",
			goFileContent: `
package features

import (
	"k8s.io/component-base/featuregate"
)
var defaultKubernetesFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{
	AppArmorFields: {Default: true, PreRelease: featuregate.Beta},
	CPUCFSQuotaPeriod: {Default: false, PreRelease: featuregate.Alpha},
	ClusterTrustBundleProjection: {Default: false, PreRelease: featuregate.Alpha},
	SELinuxMount: {Default: false, PreRelease: featuregate.Alpha},
}
`,
			expectVerifyErr: true,
			expectUpdateErr: true,
		},
		{
			name: "delete feature",
			goFileContent: `
package features

import (
	"k8s.io/component-base/featuregate"
)
var defaultKubernetesFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{
	AppArmorFields: {Default: true, PreRelease: featuregate.Beta},
	ClusterTrustBundleProjection: {Default: false, PreRelease: featuregate.Alpha},
}
`,
			expectVerifyErr: true,
			updatedFeatureListFileContent: `- name: AppArmorFields
  versionedSpecs:
  - default: true
    lockToDefault: false
    preRelease: Beta
    version: ""
- name: ClusterTrustBundleProjection
  versionedSpecs:
  - default: false
    lockToDefault: false
    preRelease: Alpha
    version: ""
`,
		},
		{
			name: "update feature",
			goFileContent: `
package features

import (
	"k8s.io/component-base/featuregate"
)
var defaultKubernetesFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{
	AppArmorFields: {Default: true, PreRelease: featuregate.GA},
	CPUCFSQuotaPeriod: {Default: false, PreRelease: featuregate.Alpha},
	ClusterTrustBundleProjection: {Default: false, PreRelease: featuregate.Alpha},
}
			`,
			expectVerifyErr: true,
			expectUpdateErr: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featureListFile := writeContentToTmpFile(t, "", "feature_list.yaml", strings.TrimSpace(featureListFileContent))
			tmpDir := filepath.Dir(featureListFile.Name())
			_ = writeContentToTmpFile(t, tmpDir, "pkg/new_features.go", tc.goFileContent)
			err := verifyOrUpdateFeatureList(tmpDir, filepath.Base(featureListFile.Name()), false, false)
			if tc.expectVerifyErr != (err != nil) {
				t.Errorf("expectVerifyErr=%v, got err: %s", tc.expectVerifyErr, err)
			}
			err = verifyOrUpdateFeatureList(tmpDir, filepath.Base(featureListFile.Name()), true, false)
			if tc.expectUpdateErr != (err != nil) {
				t.Errorf("expectVerifyErr=%v, got err: %s", tc.expectVerifyErr, err)
			}
			if tc.expectUpdateErr {
				return
			}
			updatedFeatureListFileContent, err := os.ReadFile(featureListFile.Name())
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(string(updatedFeatureListFileContent), tc.updatedFeatureListFileContent); diff != "" {
				t.Errorf("updatedFeatureListFileContent does not match expected, diff=%s", diff)
			}
		})
	}
}

func TestVerifyOrUpdateFeatureListVersioned(t *testing.T) {
	featureListFileContent := `- name: APIListChunking
  versionedSpecs:
  - default: true
    lockToDefault: true
    preRelease: GA
    version: "1.30"
- name: AppArmorFields
  versionedSpecs:
  - default: true
    lockToDefault: false
    preRelease: Beta
    version: "1.30"
- name: CPUCFSQuotaPeriod
  versionedSpecs:
  - default: false
    lockToDefault: false
    preRelease: Alpha
    version: "1.30"
  - default: true
    lockToDefault: false
    preRelease: Beta
    version: "1.31"
`
	tests := []struct {
		name                          string
		goFileContent                 string
		updatedFeatureListFileContent string
		expectVerifyErr               bool
		expectUpdateErr               bool
	}{
		{
			name: "no change",
			goFileContent: `
package features

import (
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-base/featuregate"
)
var defaultVersionedKubernetesFeatureGates = map[featuregate.Feature]featuregate.VersionedSpecs{
	AppArmorFields: {
		{Version: version.MajorMinor(1, 30), Default: true, PreRelease: featuregate.Beta},
	},
	CPUCFSQuotaPeriod: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
	},
	genericfeatures.APIListChunking: {
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},
}
var otherFeatureGates = map[featuregate.Feature]featuregate.VersionedSpecs{
	AppArmorFields: {
		{Version: version.MajorMinor(1, 30), Default: true, PreRelease: featuregate.Beta},
	},
}
`,
			updatedFeatureListFileContent: featureListFileContent,
		},
		{
			name: "same feature added twice with different lifecycle",
			goFileContent: `
package features

import (
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-base/featuregate"
)
var defaultVersionedKubernetesFeatureGates = map[featuregate.Feature]featuregate.VersionedSpecs{
	AppArmorFields: {
		{Version: version.MajorMinor(1, 30), Default: true, PreRelease: featuregate.Beta},
	},
	CPUCFSQuotaPeriod: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
	},
	genericfeatures.APIListChunking: {
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},
}
var otherFeatureGates = map[featuregate.Feature]featuregate.VersionedSpecs{
	AppArmorFields: {
		{Version: version.MajorMinor(1, 30), Default: true, PreRelease: featuregate.Alpha},
	},
}
`,
			expectVerifyErr: true,
			expectUpdateErr: true,
		},
		{
			name: "VersionedSpecs not ordered by version",
			goFileContent: `
package features

import (
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-base/featuregate"
)
var defaultVersionedKubernetesFeatureGates = map[featuregate.Feature]featuregate.VersionedSpecs{
	AppArmorFields: {
		{Version: version.MajorMinor(1, 30), Default: true, PreRelease: featuregate.Beta},
	},
	CPUCFSQuotaPeriod: {
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.Beta},
	},
	genericfeatures.APIListChunking: {
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},
}
`,
			expectVerifyErr: true,
			expectUpdateErr: true,
		},
		{
			name: "add new feature",
			goFileContent: `
package features

import (
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-base/featuregate"
)
var defaultVersionedKubernetesFeatureGates = map[featuregate.Feature]featuregate.VersionedSpecs{
	AppArmorFields: {
		{Version: version.MajorMinor(1, 30), Default: true, PreRelease: featuregate.Beta},
	},
	ClusterTrustBundleProjection: {
		{Version: version.MajorMinor(1, 30), Default: true, PreRelease: featuregate.Beta},
	},
	CPUCFSQuotaPeriod: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
	},
	genericfeatures.APIListChunking: {
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},
}
`,
			expectVerifyErr: true,
			updatedFeatureListFileContent: `- name: APIListChunking
  versionedSpecs:
  - default: true
    lockToDefault: true
    preRelease: GA
    version: "1.30"
- name: AppArmorFields
  versionedSpecs:
  - default: true
    lockToDefault: false
    preRelease: Beta
    version: "1.30"
- name: ClusterTrustBundleProjection
  versionedSpecs:
  - default: true
    lockToDefault: false
    preRelease: Beta
    version: "1.30"
- name: CPUCFSQuotaPeriod
  versionedSpecs:
  - default: false
    lockToDefault: false
    preRelease: Alpha
    version: "1.30"
  - default: true
    lockToDefault: false
    preRelease: Beta
    version: "1.31"
`,
		},
		{
			name: "remove feature",
			goFileContent: `
package features

import (
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-base/featuregate"
)
var defaultVersionedKubernetesFeatureGates = map[featuregate.Feature]featuregate.VersionedSpecs{
	CPUCFSQuotaPeriod: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
	},
	genericfeatures.APIListChunking: {
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},
}
`,
			expectVerifyErr: true,
			updatedFeatureListFileContent: `- name: APIListChunking
  versionedSpecs:
  - default: true
    lockToDefault: true
    preRelease: GA
    version: "1.30"
- name: CPUCFSQuotaPeriod
  versionedSpecs:
  - default: false
    lockToDefault: false
    preRelease: Alpha
    version: "1.30"
  - default: true
    lockToDefault: false
    preRelease: Beta
    version: "1.31"
`,
		},
		{
			name: "update feature",
			goFileContent: `
package features

import (
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-base/featuregate"
)
var defaultVersionedKubernetesFeatureGates = map[featuregate.Feature]featuregate.VersionedSpecs{
	AppArmorFields: {
		{Version: version.MajorMinor(1, 30), Default: true, PreRelease: featuregate.Beta},
	},
	CPUCFSQuotaPeriod: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.32"), Default: true, PreRelease: featuregate.GA},
	},
	genericfeatures.APIListChunking: {
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},
}
`,
			expectVerifyErr: true,
			updatedFeatureListFileContent: `- name: APIListChunking
  versionedSpecs:
  - default: true
    lockToDefault: true
    preRelease: GA
    version: "1.30"
- name: AppArmorFields
  versionedSpecs:
  - default: true
    lockToDefault: false
    preRelease: Beta
    version: "1.30"
- name: CPUCFSQuotaPeriod
  versionedSpecs:
  - default: false
    lockToDefault: false
    preRelease: Alpha
    version: "1.30"
  - default: true
    lockToDefault: false
    preRelease: Beta
    version: "1.31"
  - default: true
    lockToDefault: false
    preRelease: GA
    version: "1.32"
`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featureListFile := writeContentToTmpFile(t, "", "feature_list.yaml", strings.TrimSpace(featureListFileContent))
			tmpDir := filepath.Dir(featureListFile.Name())
			_ = writeContentToTmpFile(t, tmpDir, "pkg/new_features.go", tc.goFileContent)
			err := verifyOrUpdateFeatureList(tmpDir, filepath.Base(featureListFile.Name()), false, true)
			if tc.expectVerifyErr != (err != nil) {
				t.Errorf("expectVerifyErr=%v, got err: %s", tc.expectVerifyErr, err)
			}
			err = verifyOrUpdateFeatureList(tmpDir, filepath.Base(featureListFile.Name()), true, true)
			if tc.expectUpdateErr != (err != nil) {
				t.Errorf("expectVerifyErr=%v, got err: %s", tc.expectVerifyErr, err)
			}
			if tc.expectUpdateErr {
				return
			}
			updatedFeatureListFileContent, err := os.ReadFile(featureListFile.Name())
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(string(updatedFeatureListFileContent), tc.updatedFeatureListFileContent); diff != "" {
				t.Errorf("updatedFeatureListFileContent does not match expected, diff=%s", diff)
			}
		})
	}
}

func TestExtractFeatureInfoListFromFile(t *testing.T) {
	tests := []struct {
		name             string
		fileContent      string
		expectedFeatures []featureInfo
	}{
		{
			name: "map in var",
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
	genericfeatures.AggregatedDiscoveryEndpoint: {Default: false, PreRelease: featuregate.Alpha},
}
			`,
			expectedFeatures: []featureInfo{
				{
					Name:     "AppArmorFields",
					FullName: "AppArmorFields",
					VersionedSpecs: []featureSpec{
						{Default: true, PreRelease: "Beta"},
					},
				},
				{
					Name:     "CPUCFSQuotaPeriod",
					FullName: "CPUCFSQuotaPeriod",
					VersionedSpecs: []featureSpec{
						{Default: false, PreRelease: "Alpha"},
					},
				},
				{
					Name:     "AggregatedDiscoveryEndpoint",
					FullName: "genericfeatures.AggregatedDiscoveryEndpoint",
					VersionedSpecs: []featureSpec{
						{Default: false, PreRelease: "Alpha"},
					},
				},
			},
		},
		{
			name: "map in var with alias",
			fileContent: `
package features

import (
	"k8s.io/apimachinery/pkg/util/version"
	genericfeatures "k8s.io/apiserver/pkg/features"
	fg "k8s.io/component-base/featuregate"
)
const (
    CPUCFSQuotaPeriodDefault = false
)
var defaultVersionedKubernetesFeatureGates = map[fg.Feature]fg.FeatureSpec{
	AppArmorFields: {Default: true, PreRelease: fg.Beta},
	CPUCFSQuotaPeriod: {Default: CPUCFSQuotaPeriodDefault, PreRelease: fg.Alpha},
	genericfeatures.AggregatedDiscoveryEndpoint: {Default: false, PreRelease: fg.Alpha},
}
			`,
			expectedFeatures: []featureInfo{
				{
					Name:     "AppArmorFields",
					FullName: "AppArmorFields",
					VersionedSpecs: []featureSpec{
						{Default: true, PreRelease: "Beta"},
					},
				},
				{
					Name:     "CPUCFSQuotaPeriod",
					FullName: "CPUCFSQuotaPeriod",
					VersionedSpecs: []featureSpec{
						{Default: false, PreRelease: "Alpha"},
					},
				},
				{
					Name:     "AggregatedDiscoveryEndpoint",
					FullName: "genericfeatures.AggregatedDiscoveryEndpoint",
					VersionedSpecs: []featureSpec{
						{Default: false, PreRelease: "Alpha"},
					},
				},
			},
		},
		{
			name: "map in function return statement",
			fileContent: `
package features

import (
	"k8s.io/component-base/featuregate"
)

const (
	ComponentSLIs featuregate.Feature = "ComponentSLIs"
)

func featureGates() map[featuregate.Feature]featuregate.FeatureSpec {
	return map[featuregate.Feature]featuregate.FeatureSpec{
		ComponentSLIs: {Default: true, PreRelease: featuregate.Beta},
	}
}
			`,
			expectedFeatures: []featureInfo{
				{
					Name:     "ComponentSLIs",
					FullName: "ComponentSLIs",
					VersionedSpecs: []featureSpec{
						{Default: true, PreRelease: "Beta"},
					},
				},
			},
		},
		// 		{
		// 			name: "map in function call",
		// 			fileContent: `
		// package features

		// import (
		// 	"k8s.io/component-base/featuregate"
		// )

		// const (
		// 	ComponentSLIs featuregate.Feature = "ComponentSLIs"
		// )

		// func featureGates() featuregate.FeatureGate {
		// 	featureGate := featuregate.NewFeatureGate()
		// 	_ = featureGate.Add(map[featuregate.Feature]featuregate.FeatureSpec{
		// 		ComponentSLIs: {
		// 			Default: true, PreRelease: featuregate.Beta}})
		// 	return featureGate
		// }
		// 			`,
		// 		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			newFile := writeContentToTmpFile(t, "", "new_features.go", tc.fileContent)
			fset := token.NewFileSet()
			features, err := extractFeatureInfoListFromFile(fset, newFile.Name(), false)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(features, tc.expectedFeatures); diff != "" {
				t.Errorf("File contents: got=%v, want=%v, diff=%s", features, tc.expectedFeatures, diff)
			}
		})
	}
}

func TestExtractFeatureInfoListFromFileVersioned(t *testing.T) {
	tests := []struct {
		name             string
		fileContent      string
		expectedFeatures []featureInfo
		expectErr        bool
	}{
		{
			name: "map in var",
			fileContent: `
package features

import (
	"k8s.io/apimachinery/pkg/util/version"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/component-base/featuregate"
)
var defaultVersionedKubernetesFeatureGates = map[featuregate.Feature]featuregate.VersionedSpecs{
	AppArmorFields: {
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
	},
	CPUCFSQuotaPeriod: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
	},
	genericfeatures.AggregatedDiscoveryEndpoint: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
	},
}
			`,
			expectedFeatures: []featureInfo{
				{
					Name:     "AppArmorFields",
					FullName: "AppArmorFields",
					VersionedSpecs: []featureSpec{
						{Default: true, PreRelease: "Beta", Version: "1.31"},
					},
				},
				{
					Name:     "CPUCFSQuotaPeriod",
					FullName: "CPUCFSQuotaPeriod",
					VersionedSpecs: []featureSpec{
						{Default: false, PreRelease: "Alpha", Version: "1.29"},
					},
				},
				{
					Name:     "AggregatedDiscoveryEndpoint",
					FullName: "genericfeatures.AggregatedDiscoveryEndpoint",
					VersionedSpecs: []featureSpec{
						{Default: false, PreRelease: "Alpha", Version: "1.30"},
					},
				},
			},
		},
		{
			name: "map in var with alias",
			fileContent: `
package features

import (
	"k8s.io/apimachinery/pkg/util/version"
	genericfeatures "k8s.io/apiserver/pkg/features"
	fg "k8s.io/component-base/featuregate"
)
const (
    CPUCFSQuotaPeriodDefault = false
)
var defaultVersionedKubernetesFeatureGates = map[fg.Feature]fg.VersionedSpecs{
	AppArmorFields: {
		{Version: version.MustParse("1.31"), Default: true, PreRelease: fg.Beta},
	},
	CPUCFSQuotaPeriod: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: fg.Alpha},
	},
	genericfeatures.AggregatedDiscoveryEndpoint: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: fg.Alpha},
	},
}
			`,
			expectedFeatures: []featureInfo{
				{
					Name:     "AppArmorFields",
					FullName: "AppArmorFields",
					VersionedSpecs: []featureSpec{
						{Default: true, PreRelease: "Beta", Version: "1.31"},
					},
				},
				{
					Name:     "CPUCFSQuotaPeriod",
					FullName: "CPUCFSQuotaPeriod",
					VersionedSpecs: []featureSpec{
						{Default: false, PreRelease: "Alpha", Version: "1.29"},
					},
				},
				{
					Name:     "AggregatedDiscoveryEndpoint",
					FullName: "genericfeatures.AggregatedDiscoveryEndpoint",
					VersionedSpecs: []featureSpec{
						{Default: false, PreRelease: "Alpha", Version: "1.30"},
					},
				},
			},
		},
		{
			name: "map in function return statement",
			fileContent: `
package features

import (
	"k8s.io/component-base/featuregate"
)

const (
	ComponentSLIs featuregate.Feature = "ComponentSLIs"
)

func featureGates() map[featuregate.Feature]featuregate.VersionedSpecs {
	return map[featuregate.Feature]featuregate.VersionedSpecs{
		ComponentSLIs: {
			{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
			{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
			{Version: version.MustParse("1.32"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
		},
	}
}
			`,
			expectedFeatures: []featureInfo{
				{
					Name:     "ComponentSLIs",
					FullName: "ComponentSLIs",
					VersionedSpecs: []featureSpec{
						{Default: false, PreRelease: "Alpha", Version: "1.30"},
						{Default: true, PreRelease: "Beta", Version: "1.31"},
						{Default: true, PreRelease: "GA", Version: "1.32", LockToDefault: true},
					},
				},
			},
		},
		{
			name: "error when VersionedSpecs not ordered by version",
			fileContent: `
package features

import (
	"k8s.io/component-base/featuregate"
)

const (
	ComponentSLIs featuregate.Feature = "ComponentSLIs"
)

func featureGates() map[featuregate.Feature]featuregate.VersionedSpecs {
	return map[featuregate.Feature]featuregate.VersionedSpecs{
		ComponentSLIs: {
			{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
			{Version: version.MustParse("1.32"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
			{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
		},
	}
}
			`,
			expectErr: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			newFile := writeContentToTmpFile(t, "", "new_features.go", tc.fileContent)
			fset := token.NewFileSet()
			features, err := extractFeatureInfoListFromFile(fset, newFile.Name(), true)
			if tc.expectErr {
				if err == nil {
					t.Fatal("expect err")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(features, tc.expectedFeatures); diff != "" {
				t.Errorf("File contents: got=%v, want=%v, diff=%s", features, tc.expectedFeatures, diff)
			}
		})
	}
}

func writeContentToTmpFile(t *testing.T, tmpDir, fileName, fileContent string) *os.File {
	if tmpDir == "" {
		p, err := os.MkdirTemp("", "k8s")
		if err != nil {
			t.Fatal(err)
		}
		tmpDir = p
	}
	fullPath := filepath.Join(tmpDir, fileName)
	err := os.MkdirAll(filepath.Dir(fullPath), os.ModePerm)
	if err != nil {
		t.Fatal(err)
	}
	tmpfile, err := os.Create(fullPath)
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
	fmt.Printf("sizhangDebug: Written tmp file %s\n", tmpfile.Name())
	return tmpfile
}

func TestParseFeatureSpec(t *testing.T) {
	tests := []struct {
		name                string
		val                 ast.Expr
		expectedFeatureSpec featureSpec
	}{
		{
			name: "spec by field name",
			expectedFeatureSpec: featureSpec{
				Default: true, LockToDefault: true, PreRelease: "Beta", Version: "1.31",
			},
			val: &ast.CompositeLit{
				Elts: []ast.Expr{
					&ast.KeyValueExpr{
						Key: &ast.Ident{
							Name: "Version",
						},
						Value: &ast.CallExpr{
							Fun: &ast.SelectorExpr{
								X: &ast.Ident{
									Name: "version",
								},
								Sel: &ast.Ident{
									Name: "MustParse",
								},
							},
							Args: []ast.Expr{
								&ast.BasicLit{
									Kind:  token.STRING,
									Value: "\"1.31\"",
								},
							},
						},
					},
					&ast.KeyValueExpr{
						Key: &ast.Ident{
							Name: "Default",
						},
						Value: &ast.Ident{
							Name: "true",
						},
					},
					&ast.KeyValueExpr{
						Key: &ast.Ident{
							Name: "LockToDefault",
						},
						Value: &ast.Ident{
							Name: "true",
						},
					},
					&ast.KeyValueExpr{
						Key: &ast.Ident{
							Name: "PreRelease",
						},
						Value: &ast.SelectorExpr{
							X: &ast.Ident{
								Name: "featuregate",
							},
							Sel: &ast.Ident{
								Name: "Beta",
							},
						},
					},
				},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			variables := map[string]ast.Expr{}
			spec, err := parseFeatureSpec(variables, tc.val)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(tc.expectedFeatureSpec, spec) {
				t.Errorf("expected: %#v, got %#v", tc.expectedFeatureSpec, spec)
			}
		})
	}
}
