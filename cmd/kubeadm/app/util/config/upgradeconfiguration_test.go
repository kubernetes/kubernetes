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

package config

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/yaml"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

func TestDocMapToUpgradeConfiguration(t *testing.T) {
	tests := []struct {
		name        string
		cfg         kubeadmapiv1.UpgradeConfiguration
		expectedCfg kubeadmapi.UpgradeConfiguration
	}{
		{
			name: "default config is set correctly",
			cfg: kubeadmapiv1.UpgradeConfiguration{
				TypeMeta: metav1.TypeMeta{
					APIVersion: kubeadmapiv1.SchemeGroupVersion.String(),
					Kind:       constants.UpgradeConfigurationKind,
				},
			},
			expectedCfg: kubeadmapi.UpgradeConfiguration{
				Apply: kubeadmapi.UpgradeApplyConfiguration{
					CertificateRenewal: ptr.To(true),
					EtcdUpgrade:        ptr.To(true),
				},
				Node: kubeadmapi.UpgradeNodeConfiguration{
					CertificateRenewal: ptr.To(true),
					EtcdUpgrade:        ptr.To(true),
				},
			},
		},
		{
			name: "cfg has part of fields configured",
			cfg: kubeadmapiv1.UpgradeConfiguration{
				Apply: kubeadmapiv1.UpgradeApplyConfiguration{
					CertificateRenewal: ptr.To(false),
				},
				Node: kubeadmapiv1.UpgradeNodeConfiguration{
					EtcdUpgrade: ptr.To(false),
				},
				TypeMeta: metav1.TypeMeta{
					APIVersion: kubeadmapiv1.SchemeGroupVersion.String(),
					Kind:       constants.UpgradeConfigurationKind,
				},
			},
			expectedCfg: kubeadmapi.UpgradeConfiguration{
				Apply: kubeadmapi.UpgradeApplyConfiguration{
					CertificateRenewal: ptr.To(false),
					EtcdUpgrade:        ptr.To(true),
				},
				Node: kubeadmapi.UpgradeNodeConfiguration{
					CertificateRenewal: ptr.To(true),
					EtcdUpgrade:        ptr.To(false),
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			b, err := yaml.Marshal(tc.cfg)
			if err != nil {
				t.Fatalf("unexpected error while marshalling to YAML: %v", err)
			}
			docmap, err := kubeadmutil.SplitYAMLDocuments(b)
			if err != nil {
				t.Fatalf("Unexpect error of SplitYAMLDocuments: %v", err)
			}
			cfg, err := DocMapToUpgradeConfiguration(docmap)
			if err != nil {
				t.Fatalf("unexpected error of DocMapToUpgradeConfiguration: %v\nconfig: %s", err, string(b))
			}
			if diff := cmp.Diff(*cfg, tc.expectedCfg, cmpopts.IgnoreFields(kubeadmapi.UpgradeConfiguration{}, "Timeouts")); diff != "" {
				t.Fatalf("DocMapToUpgradeConfiguration returned unexpected diff (-want,+got):\n%s", diff)
			}
		})
	}
}
