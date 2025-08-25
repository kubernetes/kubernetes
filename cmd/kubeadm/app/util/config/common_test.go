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
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/lithammer/dedent"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/version"
	apimachineryversion "k8s.io/apimachinery/pkg/version"
	"sigs.k8s.io/yaml"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1old "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta3"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

const KubeadmGroupName = "kubeadm.k8s.io"

var formats = []struct {
	name    string
	marshal func(interface{}) ([]byte, error)
}{
	{name: "JSON", marshal: json.Marshal},
	{name: "YAML", marshal: yaml.Marshal},
}

func TestValidateSupportedVersion(t *testing.T) {
	tests := []struct {
		gvk               schema.GroupVersionKind
		allowDeprecated   bool
		allowExperimental bool
		expectedErr       bool
	}{
		{
			gvk: schema.GroupVersionKind{
				Group:   KubeadmGroupName,
				Version: "v1alpha1",
				Kind:    "InitConfiguration",
			},
			expectedErr: true,
		},
		{
			gvk: schema.GroupVersionKind{
				Group:   KubeadmGroupName,
				Version: "v1alpha2",
				Kind:    "InitConfiguration",
			},
			expectedErr: true,
		},
		{
			gvk: schema.GroupVersionKind{
				Group:   KubeadmGroupName,
				Version: "v1alpha3",
				Kind:    "InitConfiguration",
			},
			expectedErr: true,
		},
		{
			gvk: schema.GroupVersionKind{
				Group:   KubeadmGroupName,
				Version: "v1beta1",
				Kind:    "InitConfiguration",
			},
			expectedErr: true,
		},
		{
			gvk: schema.GroupVersionKind{
				Group:   KubeadmGroupName,
				Version: "v1beta2",
				Kind:    "InitConfiguration",
			},
			expectedErr: true,
		},
		{
			gvk: schema.GroupVersionKind{
				Group:   KubeadmGroupName,
				Version: "v1beta3",
				Kind:    "ClusterConfiguration",
			},
		},
		{
			gvk: schema.GroupVersionKind{
				Group:   "foo.k8s.io",
				Version: "v1",
				Kind:    "InitConfiguration",
			},
		},
		{
			gvk: schema.GroupVersionKind{
				Group:   KubeadmGroupName,
				Version: "v1beta4",
				Kind:    "ResetConfiguration",
			},
		},
	}

	for _, rt := range tests {
		t.Run(fmt.Sprintf("%s/allowDeprecated:%t", rt.gvk.GroupVersion(), rt.allowDeprecated), func(t *testing.T) {
			err := validateSupportedVersion(rt.gvk, rt.allowDeprecated, rt.allowExperimental)
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

// NOTE: do not delete this test once an older API is removed and there is only one API left.
// Update the inline "gv" and "gvNew" variables, to have the GroupVersion String of
// the API to be tested. If there are no new APIs make "gvNew" point to the old API.
// If an experimental API has to be tested, use the 'allowExperimental' option
// and add negative and positive test cases for the experimental API.
func TestMigrateOldConfig(t *testing.T) {
	var (
		gv    = kubeadmapiv1old.SchemeGroupVersion.String()
		gvNew = kubeadmapiv1.SchemeGroupVersion.String()
	)
	tests := []struct {
		name              string
		oldCfg            string
		expectedKinds     []string
		expectErr         bool
		allowExperimental bool
	}{
		{
			name:      "empty file produces empty result",
			oldCfg:    "",
			expectErr: false,
		},
		{
			name: "bad config produces error",
			oldCfg: dedent.Dedent(fmt.Sprintf(`
			apiVersion: %s
			`, gv)),
			expectErr: true,
		},
		{
			name: "unknown API produces error",
			oldCfg: dedent.Dedent(fmt.Sprintf(`
			apiVersion: %s
			kind: Foo
			`, gv)),
			expectErr: true,
		},
		{
			name: "InitConfiguration only gets migrated",
			oldCfg: dedent.Dedent(fmt.Sprintf(`
			apiVersion: %s
			kind: InitConfiguration
			`, gv)),
			expectedKinds: []string{
				constants.InitConfigurationKind,
				constants.ClusterConfigurationKind,
			},
			expectErr: false,
		},
		{
			name: "ClusterConfiguration only gets migrated",
			oldCfg: dedent.Dedent(fmt.Sprintf(`
			apiVersion: %s
			kind: ClusterConfiguration
			kubernetesVersion: v1.10.0
			`, gv)),
			expectedKinds: []string{
				constants.InitConfigurationKind,
				constants.ClusterConfigurationKind,
			},
			expectErr: false,
		},
		{
			name: "JoinConfiguration only gets migrated",
			oldCfg: dedent.Dedent(fmt.Sprintf(`
			apiVersion: %s
			kind: JoinConfiguration
			discovery:
			  bootstrapToken:
			    token: abcdef.0123456789abcdef
			    apiServerEndpoint: kube-apiserver:6443
			    unsafeSkipCAVerification: true
			`, gv)),
			expectedKinds: []string{
				constants.JoinConfigurationKind,
			},
			expectErr: false,
		},
		{
			name: "Init + Cluster Configurations are migrated",
			oldCfg: dedent.Dedent(fmt.Sprintf(`
			apiVersion: %s
			kind: InitConfiguration
			---
			apiVersion: %[1]s
			kind: ClusterConfiguration
			kubernetesVersion: v1.10.0
			`, gv)),
			expectedKinds: []string{
				constants.InitConfigurationKind,
				constants.ClusterConfigurationKind,
			},
			expectErr: false,
		},
		{
			name: "Init + Join Configurations are migrated",
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
			`, gv)),
			expectedKinds: []string{
				constants.InitConfigurationKind,
				constants.ClusterConfigurationKind,
				constants.JoinConfigurationKind,
			},
			expectErr: false,
		},
		{
			name: "Cluster + Join Configurations are migrated",
			oldCfg: dedent.Dedent(fmt.Sprintf(`
			apiVersion: %s
			kind: ClusterConfiguration
			kubernetesVersion: v1.10.0
			---
			apiVersion: %[1]s
			kind: JoinConfiguration
			discovery:
			  bootstrapToken:
			    token: abcdef.0123456789abcdef
			    apiServerEndpoint: kube-apiserver:6443
			    unsafeSkipCAVerification: true
			`, gv)),
			expectedKinds: []string{
				constants.InitConfigurationKind,
				constants.ClusterConfigurationKind,
				constants.JoinConfigurationKind,
			},
			expectErr: false,
		},
		{
			name: "Init + Cluster + Join Configurations are migrated",
			oldCfg: dedent.Dedent(fmt.Sprintf(`
			apiVersion: %s
			kind: InitConfiguration
			---
			apiVersion: %[1]s
			kind: ClusterConfiguration
			kubernetesVersion: v1.10.0
			---
			apiVersion: %[1]s
			kind: JoinConfiguration
			discovery:
			  bootstrapToken:
			    token: abcdef.0123456789abcdef
			    apiServerEndpoint: kube-apiserver:6443
			    unsafeSkipCAVerification: true
			`, gv)),
			expectedKinds: []string{
				constants.InitConfigurationKind,
				constants.ClusterConfigurationKind,
				constants.JoinConfigurationKind,
			},
			expectErr: false,
		},
		{
			name: "component configs are not migrated",
			oldCfg: dedent.Dedent(fmt.Sprintf(`
			apiVersion: %s
			kind: InitConfiguration
			---
			apiVersion: %[1]s
			kind: ClusterConfiguration
			kubernetesVersion: v1.10.0
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
			`, gv)),
			expectedKinds: []string{
				constants.InitConfigurationKind,
				constants.ClusterConfigurationKind,
				constants.JoinConfigurationKind,
			},
			expectErr: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			b, err := MigrateOldConfig([]byte(test.oldCfg), test.allowExperimental, defaultEmptyMigrateMutators())
			if test.expectErr != (err != nil) {
				t.Fatalf("expected error: %v, got: %v", test.expectErr, err != nil)
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
			expectedGV := gvNew
			if test.allowExperimental {
				expectedGV = gvNew
			}
			for _, gvk := range gvks {
				if gvk.GroupVersion().String() != expectedGV {
					t.Errorf("GV mismatch, expected GV: %s, got GV: %s", expectedGV, gvk.GroupVersion().String())
				}
			}
		})
	}
}

// Test the migration of all breaking changes in v1beta4, marked as "MIGRATED" in the YAML below:
// - ExtraArgs
// - ClusterConfiguration.APIServer.TimeoutForControlPlane -> {Init|Join}Configuration.Timeout.ControlPlaneComponentHealthCheck
// - JoinConfiguration.Discovery.Timeout -> JoinConfiguration.Timeout.Discovery
func TestMigrateV1Beta3WithBreakingChanges(t *testing.T) {
	var (
		gv         = kubeadmapiv1old.SchemeGroupVersion.String()
		gvNew      = kubeadmapiv1.SchemeGroupVersion.String()
		criSocket  = fmt.Sprintf("%s:///some-socket-path", kubeadmapiv1.DefaultContainerRuntimeURLScheme)
		caCertPath = kubeadmapiv1.DefaultCACertPath

		input = dedent.Dedent(fmt.Sprintf(`
		apiVersion: %s
		bootstrapTokens:
		- groups:
		  - system:bootstrappers:kubeadm:default-node-token
		  token: n32eo4.cci2j99rnn8fmv42
		  ttl: 24h0m0s
		  usages:
		  - signing
		  - authentication
		kind: InitConfiguration
		localAPIEndpoint:
		  advertiseAddress: 1.2.3.4
		  bindPort: 6443
		nodeRegistration:
		  criSocket: %[2]s
		  kubeletExtraArgs: # MIGRATED
		    foo: bar
		  name: node
		---
		apiServer:
		  timeoutForControlPlane: 2m32s # MIGRATED
		  extraArgs: # MIGRATED
		    foo: bar
		apiVersion: %[1]s
		controllerManager:
		  extraArgs: # MIGRATED
		    foo: bar
		etcd:
		  local:
		    extraArgs: # MIGRATED
		      foo: bar
		kind: ClusterConfiguration
		kubernetesVersion: v1.10.0
		scheduler:
		  extraArgs: # MIGRATED
		    foo: bar
		---
		apiVersion: %[1]s
		kind: JoinConfiguration
		nodeRegistration:
		  criSocket: %[2]s
		  imagePullPolicy: IfNotPresent
		  kubeletExtraArgs: # MIGRATED
		    foo: baz
		  name: foo
		  taints: null
		discovery:
		  bootstrapToken:
		    apiServerEndpoint: some-address:6443
		    token: abcdef.0123456789abcdef
		    unsafeSkipCAVerification: true
		  tlsBootstrapToken: abcdef.0123456789abcdef
		  timeout: 2m10s # MIGRATED
		`, gv, criSocket))

		expectedOutput = dedent.Dedent(fmt.Sprintf(`
		apiVersion: %s
		bootstrapTokens:
		- groups:
		  - system:bootstrappers:kubeadm:default-node-token
		  token: n32eo4.cci2j99rnn8fmv42
		  ttl: 24h0m0s
		  usages:
		  - signing
		  - authentication
		kind: InitConfiguration
		localAPIEndpoint:
		  advertiseAddress: 1.2.3.4
		  bindPort: 6443
		nodeRegistration:
		  criSocket: %[2]s
		  imagePullPolicy: IfNotPresent
		  imagePullSerial: true
		  kubeletExtraArgs:
		  - name: foo
		    value: bar
		  name: node
		  taints:
		  - effect: NoSchedule
		    key: node-role.kubernetes.io/control-plane
		timeouts:
		  controlPlaneComponentHealthCheck: 2m32s
		  discovery: 5m0s
		  etcdAPICall: 2m0s
		  kubeletHealthCheck: 4m0s
		  kubernetesAPICall: 1m0s
		  tlsBootstrap: 5m0s
		  upgradeManifests: 5m0s
		---
		apiServer:
		  extraArgs:
		  - name: foo
		    value: bar
		apiVersion: %[1]s
		caCertificateValidityPeriod: 87600h0m0s
		certificateValidityPeriod: 8760h0m0s
		certificatesDir: /etc/kubernetes/pki
		clusterName: kubernetes
		controllerManager:
		  extraArgs:
		  - name: foo
		    value: bar
		dns: {}
		encryptionAlgorithm: RSA-2048
		etcd:
		  local:
		    dataDir: /var/lib/etcd
		    extraArgs:
		    - name: foo
		      value: bar
		imageRepository: registry.k8s.io
		kind: ClusterConfiguration
		kubernetesVersion: v1.10.0
		networking:
		  dnsDomain: cluster.local
		  serviceSubnet: 10.96.0.0/12
		proxy: {}
		scheduler:
		  extraArgs:
		  - name: foo
		    value: bar
		---
		apiVersion: %[1]s
		caCertPath: %[3]s
		discovery:
		  bootstrapToken:
		    apiServerEndpoint: some-address:6443
		    token: abcdef.0123456789abcdef
		    unsafeSkipCAVerification: true
		  tlsBootstrapToken: abcdef.0123456789abcdef
		kind: JoinConfiguration
		nodeRegistration:
		  criSocket: %[2]s
		  imagePullPolicy: IfNotPresent
		  imagePullSerial: true
		  kubeletExtraArgs:
		  - name: foo
		    value: baz
		  name: foo
		  taints: null
		timeouts:
		  controlPlaneComponentHealthCheck: 2m32s
		  discovery: 2m10s
		  etcdAPICall: 2m0s
		  kubeletHealthCheck: 4m0s
		  kubernetesAPICall: 1m0s
		  tlsBootstrap: 5m0s
		  upgradeManifests: 5m0s
		`, gvNew, criSocket, caCertPath))
	)

	b, err := MigrateOldConfig([]byte(input), false, defaultEmptyMigrateMutators())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Trim one leading new line as MigrateOldConfig does the same
	expectedOutput = strings.TrimLeft(expectedOutput, "\n")

	// Split string lines in the diff
	diff := cmp.Diff(expectedOutput, string(b), cmpopts.AcyclicTransformer("multiline", func(s string) []string {
		return strings.Split(s, "\n")
	}))
	if len(diff) > 0 {
		t.Fatalf("unexpected diff (-want,+got):\n%s", diff)
	}
}

// NOTE: do not delete this test once an older API is removed and there is only one API left.
// Update the inline "gv" and "gvNew" variables, to have the GroupVersion String of
// the API to be tested. If there are no experimental APIs make "gvNew" point to
// an non-experimental API.
func TestValidateConfig(t *testing.T) {
	var (
		gv    = kubeadmapiv1old.SchemeGroupVersion.String()
		gvNew = kubeadmapiv1.SchemeGroupVersion.String()
	)
	tests := []struct {
		name              string
		cfg               string
		expectedError     bool
		allowExperimental bool
	}{
		{
			name: "invalid subdomain",
			cfg: dedent.Dedent(fmt.Sprintf(`
			apiVersion: %s
			kind: InitConfiguration
			  name: foo bar # not a valid subdomain
			`, gv)),
			expectedError: true,
		},
		{
			name: "unknown API GVK",
			cfg: dedent.Dedent(`
			apiVersion: foo/bar # not a valid GroupVersion
			kind: zzz # not a valid Kind
			`),
			expectedError: true,
		},
		{
			name: "legacy API GVK",
			cfg: dedent.Dedent(`
			apiVersion: kubeadm.k8s.io/v1beta1 # legacy API
			kind: InitConfiguration
			`),
			expectedError: true,
		},
		{
			name: "unknown field",
			cfg: dedent.Dedent(fmt.Sprintf(`
			apiVersion: %s
			kind: InitConfiguration
			foo: bar
			`, gv)),
			expectedError: true,
		},
		{
			name: "valid",
			cfg: dedent.Dedent(fmt.Sprintf(`
			apiVersion: %s
			kind: InitConfiguration
			`, gv)),
			expectedError: false,
		},
		{
			name: "valid: experimental API",
			cfg: dedent.Dedent(fmt.Sprintf(`
			apiVersion: %s
			kind: InitConfiguration
			`, gvNew)),
			expectedError: false,
		},
		{
			name: "valid ResetConfiguration",
			cfg: dedent.Dedent(fmt.Sprintf(`
			apiVersion: %s
			kind: ResetConfiguration
			force: true
			`, gvNew)),
			expectedError: false,
		},
		{
			name: "invalid field in ResetConfiguration",
			cfg: dedent.Dedent(fmt.Sprintf(`
			apiVersion: %s
			kind: ResetConfiguration
			foo: bar
			`, gvNew)),
			expectedError: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := ValidateConfig([]byte(test.cfg), test.allowExperimental)
			if (err != nil) != test.expectedError {
				t.Fatalf("expected error: %v, got: %v, error: %v", test.expectedError, (err != nil), err)
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

// TODO: update the test cases for this test once v1beta3 is removed.
func TestDefaultMigrateMutators(t *testing.T) {
	tests := []struct {
		name          string
		mutators      migrateMutators
		input         []any
		expected      []any
		expectedDiff  bool
		expectedError bool
	}{
		{
			name:     "mutate InitConfiguration",
			mutators: defaultMigrateMutators(),
			input: []any{&kubeadmapi.InitConfiguration{
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					APIServer: kubeadmapi.APIServer{
						TimeoutForControlPlane: &metav1.Duration{
							Duration: 1234 * time.Millisecond,
						},
					},
				},
				Timeouts: &kubeadmapi.Timeouts{
					ControlPlaneComponentHealthCheck: &metav1.Duration{},
				},
			}},
			expected: []any{&kubeadmapi.InitConfiguration{
				Timeouts: &kubeadmapi.Timeouts{
					ControlPlaneComponentHealthCheck: &metav1.Duration{
						Duration: 1234 * time.Millisecond,
					},
				},
			}},
		},
		{
			name:     "mutate JoinConfiguration",
			mutators: defaultMigrateMutators(),
			input: []any{&kubeadmapi.JoinConfiguration{
				Discovery: kubeadmapi.Discovery{
					Timeout: &metav1.Duration{
						Duration: 1234 * time.Microsecond,
					},
				},
				Timeouts: &kubeadmapi.Timeouts{
					Discovery: &metav1.Duration{},
				},
			}},
			expected: []any{&kubeadmapi.JoinConfiguration{
				Timeouts: &kubeadmapi.Timeouts{
					Discovery: &metav1.Duration{
						Duration: 1234 * time.Microsecond,
					},
				},
			}},
		},
		{
			name:     "diff when mutating InitConfiguration",
			mutators: defaultMigrateMutators(),
			input: []any{&kubeadmapi.InitConfiguration{
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					APIServer: kubeadmapi.APIServer{
						TimeoutForControlPlane: &metav1.Duration{
							Duration: 1234 * time.Millisecond,
						},
					},
				},
				Timeouts: &kubeadmapi.Timeouts{
					ControlPlaneComponentHealthCheck: &metav1.Duration{},
				},
			}},
			expected: []any{&kubeadmapi.InitConfiguration{
				Timeouts: &kubeadmapi.Timeouts{
					ControlPlaneComponentHealthCheck: &metav1.Duration{
						Duration: 1 * time.Millisecond, // a different value
					},
				},
			}},
			expectedDiff: true,
		},
		{
			name:          "expect an error for a missing mutator",
			mutators:      migrateMutators{}, // empty list of mutators
			input:         []any{&kubeadmapi.ResetConfiguration{}},
			expectedError: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.mutators.mutate(tc.input)
			if (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, got: %v, error: %v", tc.expectedError, (err != nil), err)
			}
			if err != nil {
				return
			}
			diff := cmp.Diff(tc.expected, tc.input)
			if (len(diff) > 0) != tc.expectedDiff {
				t.Fatalf("got a diff with the expected config (-want,+got):\n%s", diff)
			}
		})
	}
}
