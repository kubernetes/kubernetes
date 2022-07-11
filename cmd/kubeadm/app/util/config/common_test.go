/*
Copyright 2018 The Kubernetes Authors.

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
	"fmt"
	"reflect"
	"testing"

	"github.com/lithammer/dedent"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/version"
	apimachineryversion "k8s.io/apimachinery/pkg/version"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1old "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta2"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta3"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

const KubeadmGroupName = "kubeadm.k8s.io"

func TestValidateSupportedVersion(t *testing.T) {
	tests := []struct {
		gv              schema.GroupVersion
		allowDeprecated bool
		expectedErr     bool
	}{
		{
			gv: schema.GroupVersion{
				Group:   KubeadmGroupName,
				Version: "v1alpha1",
			},
			expectedErr: true,
		},
		{
			gv: schema.GroupVersion{
				Group:   KubeadmGroupName,
				Version: "v1alpha2",
			},
			expectedErr: true,
		},
		{
			gv: schema.GroupVersion{
				Group:   KubeadmGroupName,
				Version: "v1alpha3",
			},
			expectedErr: true,
		},
		{
			gv: schema.GroupVersion{
				Group:   KubeadmGroupName,
				Version: "v1beta1",
			},
			expectedErr: true,
		},
		{
			gv: schema.GroupVersion{
				Group:   KubeadmGroupName,
				Version: "v1beta2",
			},
			allowDeprecated: true,
		},
		{
			gv: schema.GroupVersion{
				Group:   KubeadmGroupName,
				Version: "v1beta3",
			},
		},
		{
			gv: schema.GroupVersion{
				Group:   "foo.k8s.io",
				Version: "v1",
			},
		},
	}

	for _, rt := range tests {
		t.Run(fmt.Sprintf("%s/allowDeprecated:%t", rt.gv, rt.allowDeprecated), func(t *testing.T) {
			err := validateSupportedVersion(rt.gv, rt.allowDeprecated)
			if rt.expectedErr && err == nil {
				t.Error("unexpected success")
			} else if !rt.expectedErr && err != nil {
				t.Errorf("unexpected failure: %v", err)
			}
		})
	}
}

func TestLowercaseSANs(t *testing.T) {
	tests := []struct {
		name string
		in   []string
		out  []string
	}{
		{
			name: "empty struct",
		},
		{
			name: "already lowercase",
			in:   []string{"example.k8s.io"},
			out:  []string{"example.k8s.io"},
		},
		{
			name: "ip addresses and uppercase",
			in:   []string{"EXAMPLE.k8s.io", "10.100.0.1"},
			out:  []string{"example.k8s.io", "10.100.0.1"},
		},
		{
			name: "punycode and uppercase",
			in:   []string{"xn--7gq663byk9a.xn--fiqz9s", "ANOTHEREXAMPLE.k8s.io"},
			out:  []string{"xn--7gq663byk9a.xn--fiqz9s", "anotherexample.k8s.io"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := &kubeadmapiv1.ClusterConfiguration{
				APIServer: kubeadmapiv1.APIServer{
					CertSANs: test.in,
				},
			}

			LowercaseSANs(cfg.APIServer.CertSANs)

			if len(cfg.APIServer.CertSANs) != len(test.out) {
				t.Fatalf("expected %d elements, got %d", len(test.out), len(cfg.APIServer.CertSANs))
			}

			for i, expected := range test.out {
				if cfg.APIServer.CertSANs[i] != expected {
					t.Errorf("expected element %d to be %q, got %q", i, expected, cfg.APIServer.CertSANs[i])
				}
			}
		})
	}
}

func TestVerifyAPIServerBindAddress(t *testing.T) {
	tests := []struct {
		name          string
		address       string
		expectedError bool
	}{
		{
			name:    "valid address: IPV4",
			address: "192.168.0.1",
		},
		{
			name:    "valid address: IPV6",
			address: "2001:db8:85a3::8a2e:370:7334",
		},
		{
			name:          "valid address 127.0.0.1",
			address:       "127.0.0.1",
			expectedError: false,
		},
		{
			name:          "invalid address: not a global unicast 0.0.0.0",
			address:       "0.0.0.0",
			expectedError: true,
		},
		{
			name:          "invalid address: not a global unicast ::",
			address:       "::",
			expectedError: true,
		},
		{
			name:          "invalid address: cannot parse IPV4",
			address:       "255.255.255.255.255",
			expectedError: true,
		},
		{
			name:          "invalid address: cannot parse IPV6",
			address:       "2a00:800::2a00:800:10102a00",
			expectedError: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := VerifyAPIServerBindAddress(test.address); (err != nil) != test.expectedError {
				t.Errorf("expected error: %v, got %v, error: %v", test.expectedError, (err != nil), err)
			}
		})
	}
}

func TestMigrateOldConfigFromFile(t *testing.T) {
	tests := []struct {
		desc          string
		oldCfg        string
		expectedKinds []string
		expectErr     bool
	}{
		{
			desc:      "empty file produces empty result",
			oldCfg:    "",
			expectErr: false,
		},
		{
			desc: "bad config produces error",
			oldCfg: dedent.Dedent(fmt.Sprintf(`
			apiVersion: %s
			`, kubeadmapiv1old.SchemeGroupVersion.String())),
			expectErr: true,
		},
		{
			desc: "InitConfiguration only gets migrated",
			oldCfg: dedent.Dedent(fmt.Sprintf(`
			apiVersion: %s
			kind: InitConfiguration
			`, kubeadmapiv1old.SchemeGroupVersion.String())),
			expectedKinds: []string{
				constants.InitConfigurationKind,
				constants.ClusterConfigurationKind,
			},
			expectErr: false,
		},
		{
			desc: "ClusterConfiguration only gets migrated",
			oldCfg: dedent.Dedent(fmt.Sprintf(`
			apiVersion: %s
			kind: ClusterConfiguration
			`, kubeadmapiv1old.SchemeGroupVersion.String())),
			expectedKinds: []string{
				constants.InitConfigurationKind,
				constants.ClusterConfigurationKind,
			},
			expectErr: false,
		},
		{
			desc: "JoinConfiguration only gets migrated",
			oldCfg: dedent.Dedent(fmt.Sprintf(`
			apiVersion: %s
			kind: JoinConfiguration
			discovery:
			  bootstrapToken:
			    token: abcdef.0123456789abcdef
			    apiServerEndpoint: kube-apiserver:6443
			    unsafeSkipCAVerification: true
			`, kubeadmapiv1old.SchemeGroupVersion.String())),
			expectedKinds: []string{
				constants.JoinConfigurationKind,
			},
			expectErr: false,
		},
		{
			desc: "Init + Cluster Configurations are migrated",
			oldCfg: dedent.Dedent(fmt.Sprintf(`
			apiVersion: %s
			kind: InitConfiguration
			---
			apiVersion: %[1]s
			kind: ClusterConfiguration
			`, kubeadmapiv1old.SchemeGroupVersion.String())),
			expectedKinds: []string{
				constants.InitConfigurationKind,
				constants.ClusterConfigurationKind,
			},
			expectErr: false,
		},
		{
			desc: "Init + Join Configurations are migrated",
			oldCfg: dedent.Dedent(fmt.Sprintf(`
			apiVersion: %s
			kind: InitConfiguration
			---
			apiVersion: %[1]s
			kind: JoinConfiguration
			discovery:
			  bootstrapToken:
			    token: abcdef.0123456789abcdef
			    apiServerEndpoint: kube-apiserver:6443
			    unsafeSkipCAVerification: true
			`, kubeadmapiv1old.SchemeGroupVersion.String())),
			expectedKinds: []string{
				constants.InitConfigurationKind,
				constants.ClusterConfigurationKind,
				constants.JoinConfigurationKind,
			},
			expectErr: false,
		},
		{
			desc: "Cluster + Join Configurations are migrated",
			oldCfg: dedent.Dedent(fmt.Sprintf(`
			apiVersion: %s
			kind: ClusterConfiguration
			---
			apiVersion: %[1]s
			kind: JoinConfiguration
			discovery:
			  bootstrapToken:
			    token: abcdef.0123456789abcdef
			    apiServerEndpoint: kube-apiserver:6443
			    unsafeSkipCAVerification: true
			`, kubeadmapiv1old.SchemeGroupVersion.String())),
			expectedKinds: []string{
				constants.InitConfigurationKind,
				constants.ClusterConfigurationKind,
				constants.JoinConfigurationKind,
			},
			expectErr: false,
		},
		{
			desc: "Init + Cluster + Join Configurations are migrated",
			oldCfg: dedent.Dedent(fmt.Sprintf(`
			apiVersion: %s
			kind: InitConfiguration
			---
			apiVersion: %[1]s
			kind: ClusterConfiguration
			---
			apiVersion: %[1]s
			kind: JoinConfiguration
			discovery:
			  bootstrapToken:
			    token: abcdef.0123456789abcdef
			    apiServerEndpoint: kube-apiserver:6443
			    unsafeSkipCAVerification: true
			`, kubeadmapiv1old.SchemeGroupVersion.String())),
			expectedKinds: []string{
				constants.InitConfigurationKind,
				constants.ClusterConfigurationKind,
				constants.JoinConfigurationKind,
			},
			expectErr: false,
		},
		{
			desc: "component configs are not migrated",
			oldCfg: dedent.Dedent(fmt.Sprintf(`
			apiVersion: %s
			kind: InitConfiguration
			---
			apiVersion: %[1]s
			kind: ClusterConfiguration
			---
			apiVersion: %[1]s
			kind: JoinConfiguration
			discovery:
			  bootstrapToken:
			    token: abcdef.0123456789abcdef
			    apiServerEndpoint: kube-apiserver:6443
			    unsafeSkipCAVerification: true
			---
			apiVersion: kubeproxy.config.k8s.io/v1alpha1
			kind: KubeProxyConfiguration
			---
			apiVersion: kubelet.config.k8s.io/v1beta1
			kind: KubeletConfiguration
			`, kubeadmapiv1old.SchemeGroupVersion.String())),
			expectedKinds: []string{
				constants.InitConfigurationKind,
				constants.ClusterConfigurationKind,
				constants.JoinConfigurationKind,
			},
			expectErr: false,
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			b, err := MigrateOldConfig([]byte(test.oldCfg))
			if test.expectErr {
				if err == nil {
					t.Fatalf("unexpected success:\n%s", b)
				}
			} else {
				if err != nil {
					t.Fatalf("unexpected failure: %v", err)
				}
				gvks, err := kubeadmutil.GroupVersionKindsFromBytes(b)
				if err != nil {
					t.Fatalf("unexpected error returned by GroupVersionKindsFromBytes: %v", err)
				}
				if len(gvks) != len(test.expectedKinds) {
					t.Fatalf("length mismatch between resulting gvks and expected kinds:\n\tlen(gvks)=%d\n\tlen(expectedKinds)=%d",
						len(gvks), len(test.expectedKinds))
				}
				for _, expectedKind := range test.expectedKinds {
					if !kubeadmutil.GroupVersionKindsHasKind(gvks, expectedKind) {
						t.Fatalf("migration failed to produce config kind: %s", expectedKind)
					}
				}
			}
		})
	}
}

func TestIsKubeadmPrereleaseVersion(t *testing.T) {
	validVersionInfo := &apimachineryversion.Info{Major: "1", GitVersion: "v1.23.0-alpha.1"}
	tests := []struct {
		name           string
		versionInfo    *apimachineryversion.Info
		k8sVersion     *version.Version
		mcpVersion     *version.Version
		expectedResult bool
	}{
		{
			name:           "invalid versionInfo",
			versionInfo:    &apimachineryversion.Info{},
			expectedResult: false,
		},
		{
			name:           "kubeadm is not a prerelease version",
			versionInfo:    &apimachineryversion.Info{Major: "1", GitVersion: "v1.23.0"},
			expectedResult: false,
		},
		{
			name:           "mcpVersion is equal to k8sVersion",
			versionInfo:    validVersionInfo,
			k8sVersion:     version.MustParseSemantic("v1.21.0"),
			mcpVersion:     version.MustParseSemantic("v1.21.0"),
			expectedResult: true,
		},
		{
			name:           "k8sVersion is 1 MINOR version older than mcpVersion",
			versionInfo:    validVersionInfo,
			k8sVersion:     version.MustParseSemantic("v1.21.0"),
			mcpVersion:     version.MustParseSemantic("v1.22.0"),
			expectedResult: true,
		},
		{
			name:           "k8sVersion is 2 MINOR versions older than mcpVersion",
			versionInfo:    validVersionInfo,
			k8sVersion:     version.MustParseSemantic("v1.21.0"),
			mcpVersion:     version.MustParseSemantic("v1.23.0"),
			expectedResult: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := isKubeadmPrereleaseVersion(tc.versionInfo, tc.k8sVersion, tc.mcpVersion)
			if result != tc.expectedResult {
				t.Errorf("expected result: %v, got %v", tc.expectedResult, result)
			}
		})
	}
}

func TestNormalizeKubernetesVersion(t *testing.T) {
	validVersion := fmt.Sprintf("v%v", constants.MinimumControlPlaneVersion)
	validCIVersion := fmt.Sprintf("%s%s", constants.CIKubernetesVersionPrefix, validVersion)
	tests := []struct {
		name        string
		cfg         *kubeadmapi.ClusterConfiguration
		expectedCfg *kubeadmapi.ClusterConfiguration
		expectErr   bool
	}{
		{
			name: "normal version, default image repository",
			cfg: &kubeadmapi.ClusterConfiguration{
				KubernetesVersion: validVersion,
				ImageRepository:   kubeadmapiv1.DefaultImageRepository,
			},
			expectedCfg: &kubeadmapi.ClusterConfiguration{
				KubernetesVersion:   validVersion,
				CIKubernetesVersion: "",
				ImageRepository:     kubeadmapiv1.DefaultImageRepository,
				CIImageRepository:   "",
			},
			expectErr: false,
		},
		{
			name: "normal version, custom image repository",
			cfg: &kubeadmapi.ClusterConfiguration{
				KubernetesVersion: validVersion,
				ImageRepository:   "custom.repository",
			},
			expectedCfg: &kubeadmapi.ClusterConfiguration{
				KubernetesVersion:   validVersion,
				CIKubernetesVersion: "",
				ImageRepository:     "custom.repository",
				CIImageRepository:   "",
			},
			expectErr: false,
		},
		{
			name: "ci version, default image repository",
			cfg: &kubeadmapi.ClusterConfiguration{
				KubernetesVersion: validCIVersion,
				ImageRepository:   kubeadmapiv1.DefaultImageRepository,
			},
			expectedCfg: &kubeadmapi.ClusterConfiguration{
				KubernetesVersion:   validVersion,
				CIKubernetesVersion: validCIVersion,
				ImageRepository:     kubeadmapiv1.DefaultImageRepository,
				CIImageRepository:   constants.DefaultCIImageRepository,
			},
			expectErr: false,
		},
		{
			name: "ci version, custom image repository",
			cfg: &kubeadmapi.ClusterConfiguration{
				KubernetesVersion: validCIVersion,
				ImageRepository:   "custom.repository",
			},
			expectedCfg: &kubeadmapi.ClusterConfiguration{
				KubernetesVersion:   validVersion,
				CIKubernetesVersion: validCIVersion,
				ImageRepository:     "custom.repository",
				CIImageRepository:   "",
			},
			expectErr: false,
		},
		{
			name: "unsupported old version",
			cfg: &kubeadmapi.ClusterConfiguration{
				KubernetesVersion: "v0.0.0",
				ImageRepository:   kubeadmapiv1.DefaultImageRepository,
			},
			expectedCfg: &kubeadmapi.ClusterConfiguration{
				KubernetesVersion:   "v0.0.0",
				CIKubernetesVersion: "",
				ImageRepository:     kubeadmapiv1.DefaultImageRepository,
				CIImageRepository:   "",
			},
			expectErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := NormalizeKubernetesVersion(tc.cfg)
			if !reflect.DeepEqual(tc.cfg, tc.expectedCfg) {
				t.Errorf("expected ClusterConfiguration: %#v, got %#v", tc.expectedCfg, tc.cfg)
			}
			if !tc.expectErr && err != nil {
				t.Errorf("unexpected failure: %v", err)
			}
		})
	}
}
