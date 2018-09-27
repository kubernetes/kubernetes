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
	"bytes"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/version"
	clientset "k8s.io/client-go/kubernetes"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

var k8sVersionString = "v1.12.0"
var k8sVersion = version.MustParseGeneric(k8sVersionString)
var nodeName = "mynode"
var cfgFiles = map[string][]byte{
	"InitConfiguration_v1alpha3": []byte(`
apiVersion: kubeadm.k8s.io/v1alpha3
kind: InitConfiguration
`),
	"ClusterConfiguration_v1alpha3": []byte(`
apiVersion: kubeadm.k8s.io/v1alpha3
kind: ClusterConfiguration
kubernetesVersion: ` + k8sVersionString + `
`),
	"ClusterStatus_v1alpha3": []byte(`
apiVersion: kubeadm.k8s.io/v1alpha3
kind: ClusterStatus
apiEndpoints: 
  ` + nodeName + `: 
    advertiseAddress: 1.2.3.4
    bindPort: 1234
`),
	"ClusterStatus_v1alpha3_Without_APIEndpoints": []byte(`
apiVersion: kubeadm.k8s.io/v1alpha3
kind: ClusterStatus
`),
	"Kube-proxy_componentconfig": []byte(`
apiVersion: kubeproxy.config.k8s.io/v1alpha1
kind: KubeProxyConfiguration
`),
	"Kubelet_componentconfig": []byte(`
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
`),
}

var kubeletConfFiles = map[string][]byte{
	"withoutX509Cert": []byte(`
apiVersion: v1
clusters:
- cluster:
    server: https://10.0.2.15:6443
    name: kubernetes
contexts:
- context:
    cluster: kubernetes
    user: system:node:mynode
  name: system:node:mynode@kubernetes
current-context: system:node:mynode@kubernetes
kind: Config
preferences: {}
users:
- name: system:node:mynode
  user:
`),
	"configWithEmbeddedCert": []byte(`
apiVersion: v1
clusters:
- cluster:
    server: https://10.0.2.15:6443
  name: kubernetes
contexts:
- context:
    cluster: kubernetes
    user: system:node:mynode
  name: system:node:mynode@kubernetes
current-context: system:node:mynode@kubernetes
kind: Config
preferences: {}
users:
- name: system:node:mynode
  user:
      client-certificate-data: LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUM4akNDQWRxZ0F3SUJBZ0lJQWl3VURhYk5vZ1F3RFFZSktvWklodmNOQVFFTEJRQXdGVEVUTUJFR0ExVUUKQXhNS2EzVmlaWEp1WlhSbGN6QWVGdzB4T0RBNU1ERXhOVE14TWpaYUZ3MHhPVEE1TURFeE5qQXhOVGxhTURReApGVEFUQmdOVkJBb1RESE41YzNSbGJUcHViMlJsY3pFYk1Ca0dBMVVFQXhNU2MzbHpkR1Z0T201dlpHVTZiWGx1CmIyUmxNSUlCSWpBTkJna3Foa2lHOXcwQkFRRUZBQU9DQVE4QU1JSUJDZ0tDQVFFQWs2UXUzeStyNEZYUzZ4VkoKWU1vNE9kSkt3R1d1MHY4TEJIUnhhOUhvVHo1RXZLQnB1OVJoemt5dStUaFczb0xta2ZTRmNJcitHa0M5MW0zOApFelRmVE5JY2dsL0V5YkpmR1QvdGdUazZYd1kxY1UrUUdmSEFNNTBCVzFXTFVHc25CSllJZjA5eENnZTVoTkxLCnREeUJOWWNQZzg1bUJpOU9CNFJ2UlgyQVFRMjJwZ0xrQUpJWklOU0FEdUFrODN2b051SXM2YVY2bHBkR2Vva3YKdDlpTFdNR3p3a3ZTZUZQTlNGeWZ3Q055eENjb1FDQUNtSnJRS3NoeUE2bWNzdVhORWVXdlRQdVVFSWZPVFl4dwpxdkszRVBOK0xUYlA2anhUMWtTcFJUOSt4Z29uSlFhU0RsbUNBd20zRGJkSVppWUt3R2ppMkxKL0kvYWc0cTlzCjNLb0J2UUlEQVFBQm95Y3dKVEFPQmdOVkhROEJBZjhFQkFNQ0JhQXdFd1lEVlIwbEJBd3dDZ1lJS3dZQkJRVUgKQXdJd0RRWUpLb1pJaHZjTkFRRUxCUUFEZ2dFQkFLcVVrU21jdW85OG5EK015b005VFdEV0pyTndySXpQTUNqRQpCSkdyREhVaHIwcEZlRjc0RHViODNzRXlaNjFxNUVQd2Y0enNLSzdzdDRUTzZhcE9pZWJYVmN3dnZoa09HQ2dFCmFVdGNOMjFhUGxtU0tOd0c4ai8yK3ZhbU80bGplK1NnZzRUUVB0eDZWejh5VXN2RFhxSUZycjNNd1gzSDA1RW4KWXAzN05JYkhKbGxHUW5LVHA5aTg5aXF4WXVhSERqZldiVHlEY3B5NldNVjdVaFYvY1plc3lGL0NBamNHd1V6YgowRlo5bW5tMnFONlBGWHZ4RmdMSGFWZzN2SVVCbkNmVVVyY1BDNE94VFNPK21aUmUxazh3eUFpVWovSk0rZllvCkcrMi9sbThUYVZqb1U3Rmk1S2E1RzVIWTJHTGFSN1ArSXhZY3JNSENsNjJZN1JxY3JuYz0KLS0tLS1FTkQgQ0VSVElGSUNBVEUtLS0tLQo=
`),
	"configWithLinkedCert": []byte(`
apiVersion: v1
clusters:
- cluster:
    server: https://10.0.2.15:6443
  name: kubernetes
contexts:
- context:
    cluster: kubernetes
    user: system:node:mynode
  name: system:node:mynode@kubernetes
current-context: system:node:mynode@kubernetes
kind: Config
preferences: {}
users:
- name: system:node:mynode
  user:
      client-certificate: kubelet.pem
`),
}

var pemFiles = map[string][]byte{
	"mynode.pem": []byte(`
-----BEGIN CERTIFICATE-----
MIIC8jCCAdqgAwIBAgIIAiwUDabNogQwDQYJKoZIhvcNAQELBQAwFTETMBEGA1UE
AxMKa3ViZXJuZXRlczAeFw0xODA5MDExNTMxMjZaFw0xOTA5MDExNjAxNTlaMDQx
FTATBgNVBAoTDHN5c3RlbTpub2RlczEbMBkGA1UEAxMSc3lzdGVtOm5vZGU6bXlu
b2RlMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAk6Qu3y+r4FXS6xVJ
YMo4OdJKwGWu0v8LBHRxa9HoTz5EvKBpu9Rhzkyu+ThW3oLmkfSFcIr+GkC91m38
EzTfTNIcgl/EybJfGT/tgTk6XwY1cU+QGfHAM50BW1WLUGsnBJYIf09xCge5hNLK
tDyBNYcPg85mBi9OB4RvRX2AQQ22pgLkAJIZINSADuAk83voNuIs6aV6lpdGeokv
t9iLWMGzwkvSeFPNSFyfwCNyxCcoQCACmJrQKshyA6mcsuXNEeWvTPuUEIfOTYxw
qvK3EPN+LTbP6jxT1kSpRT9+xgonJQaSDlmCAwm3DbdIZiYKwGji2LJ/I/ag4q9s
3KoBvQIDAQABoycwJTAOBgNVHQ8BAf8EBAMCBaAwEwYDVR0lBAwwCgYIKwYBBQUH
AwIwDQYJKoZIhvcNAQELBQADggEBAKqUkSmcuo98nD+MyoM9TWDWJrNwrIzPMCjE
BJGrDHUhr0pFeF74Dub83sEyZ61q5EPwf4zsKK7st4TO6apOiebXVcwvvhkOGCgE
aUtcN21aPlmSKNwG8j/2+vamO4lje+Sgg4TQPtx6Vz8yUsvDXqIFrr3MwX3H05En
Yp37NIbHJllGQnKTp9i89iqxYuaHDjfWbTyDcpy6WMV7UhV/cZesyF/CAjcGwUzb
0FZ9mnm2qN6PFXvxFgLHaVg3vIUBnCfUUrcPC4OxTSO+mZRe1k8wyAiUj/JM+fYo
G+2/lm8TaVjoU7Fi5Ka5G5HY2GLaR7P+IxYcrMHCl62Y7Rqcrnc=
-----END CERTIFICATE-----
`),
}

func TestLoadInitConfigurationFromFile(t *testing.T) {
	// Create temp folder for the test case
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	var tests = []struct {
		name         string
		fileContents []byte
	}{
		// TODO: implemen v1beta1 tests after introducing v1beta1
		{
			name:         "v1alpha3.partial1",
			fileContents: cfgFiles["InitConfiguration_v1alpha3"],
		},
		{
			name:         "v1alpha3.partial2",
			fileContents: cfgFiles["ClusterConfiguration_v1alpha3"],
		},
		{
			name: "v1alpha3.full",
			fileContents: bytes.Join([][]byte{
				cfgFiles["InitConfiguration_v1alpha3"],
				cfgFiles["ClusterConfiguration_v1alpha3"],
				cfgFiles["Kube-proxy_componentconfig"],
				cfgFiles["Kubelet_componentconfig"],
			}, []byte(kubeadmconstants.YAMLDocumentSeparator)),
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t2 *testing.T) {
			cfgPath := filepath.Join(tmpdir, rt.name)
			err := ioutil.WriteFile(cfgPath, rt.fileContents, 0644)
			if err != nil {
				t.Errorf("Couldn't create file")
				return
			}

			obj, err := loadInitConfigurationFromFile(cfgPath)
			if err != nil {
				t.Errorf("Error reading file: %v", err)
				return
			}

			if obj == nil {
				t.Errorf("Unexpected nil return value")
			}
		})
	}
}

func TestGetNodeNameFromKubeletConfig(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	var tests = []struct {
		name              string
		kubeconfigContent []byte
		pemContent        []byte
		expectedError     bool
	}{
		{
			name:              "valid - with embedded cert",
			kubeconfigContent: kubeletConfFiles["configWithEmbeddedCert"],
		},
		{
			name:              "invalid - linked cert missing",
			kubeconfigContent: kubeletConfFiles["configWithLinkedCert"],
			expectedError:     true,
		},
		{
			name:              "valid - with linked cert",
			kubeconfigContent: kubeletConfFiles["configWithLinkedCert"],
			pemContent:        pemFiles["mynode.pem"],
		},
		{
			name:              "invalid - without embedded or linked X509Cert",
			kubeconfigContent: kubeletConfFiles["withoutX509Cert"],
			expectedError:     true,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t2 *testing.T) {
			if len(rt.pemContent) > 0 {
				pemPath := filepath.Join(tmpdir, "kubelet.pem")
				err := ioutil.WriteFile(pemPath, rt.pemContent, 0644)
				if err != nil {
					t.Errorf("Couldn't create pem file: %v", err)
					return
				}
				rt.kubeconfigContent = []byte(strings.Replace(string(rt.kubeconfigContent), "kubelet.pem", pemPath, -1))
			}

			kubeconfigPath := filepath.Join(tmpdir, kubeadmconstants.KubeletKubeConfigFileName)
			err := ioutil.WriteFile(kubeconfigPath, rt.kubeconfigContent, 0644)
			if err != nil {
				t.Errorf("Couldn't create kubeconfig: %v", err)
				return
			}

			name, err := getNodeNameFromKubeletConfig(tmpdir)
			if rt.expectedError != (err != nil) {
				t.Errorf("unexpected return err from getNodeRegistration: %v", err)
				return
			}
			if rt.expectedError {
				return
			}

			if name != nodeName {
				t.Errorf("invalid name")
			}
		})
	}
}

func TestGetNodeRegistration(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	var tests = []struct {
		name          string
		fileContents  []byte
		node          *v1.Node
		expectedError bool
	}{
		{
			name:          "invalid - no kubelet.conf",
			expectedError: true,
		},
		{
			name:         "valid",
			fileContents: kubeletConfFiles["configWithEmbeddedCert"],
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: nodeName,
					Annotations: map[string]string{
						kubeadmconstants.AnnotationKubeadmCRISocket: "myCRIsocket",
					},
				},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{kubeadmconstants.MasterTaint},
				},
			},
		},
		{
			name:          "invalid - no node",
			fileContents:  kubeletConfFiles["configWithEmbeddedCert"],
			expectedError: true,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t2 *testing.T) {
			cfgPath := filepath.Join(tmpdir, kubeadmconstants.KubeletKubeConfigFileName)
			if len(rt.fileContents) > 0 {
				err := ioutil.WriteFile(cfgPath, rt.fileContents, 0644)
				if err != nil {
					t.Errorf("Couldn't create file")
					return
				}
			}

			client := clientsetfake.NewSimpleClientset()

			if rt.node != nil {
				_, err := client.CoreV1().Nodes().Create(rt.node)
				if err != nil {
					t.Errorf("couldn't create Node")
					return
				}
			}

			cfg := &kubeadmapi.InitConfiguration{}
			err = getNodeRegistration(tmpdir, client, &cfg.NodeRegistration)
			if rt.expectedError != (err != nil) {
				t.Errorf("unexpected return err from getNodeRegistration: %v", err)
				return
			}
			if rt.expectedError {
				return
			}

			if cfg.NodeRegistration.Name != nodeName {
				t.Errorf("invalid cfg.NodeRegistration.Name")
			}
			if cfg.NodeRegistration.CRISocket != "myCRIsocket" {
				t.Errorf("invalid cfg.NodeRegistration.CRISocket")
			}
			if len(cfg.NodeRegistration.Taints) != 1 {
				t.Errorf("invalid cfg.NodeRegistration.Taints")
			}
		})
	}
}

func TestGetAPIEndpoint(t *testing.T) {
	var tests = []struct {
		name          string
		configMap     fakeConfigMap
		expectedError bool
	}{
		{
			name: "valid",
			configMap: fakeConfigMap{
				name: kubeadmconstants.InitConfigurationConfigMap, // ClusterConfiguration from kubeadm-config.
				data: map[string]string{
					kubeadmconstants.ClusterStatusConfigMapKey: string(cfgFiles["ClusterStatus_v1alpha3"]),
				},
			},
		},
		{
			name: "invalid - No CLusterStatus in kubeadm-config ConfigMap",
			configMap: fakeConfigMap{
				name: kubeadmconstants.InitConfigurationConfigMap, // ClusterConfiguration from kubeadm-config.
				data: map[string]string{},
			},
			expectedError: true,
		},
		{
			name: "invalid - CLusterStatus without APIEndopoints",
			configMap: fakeConfigMap{
				name: kubeadmconstants.InitConfigurationConfigMap, // ClusterConfiguration from kubeadm-config.
				data: map[string]string{
					kubeadmconstants.ClusterStatusConfigMapKey: string(cfgFiles["ClusterStatus_v1alpha3_Without_APIEndpoints"]),
				},
			},
			expectedError: true,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			cfg := &kubeadmapi.InitConfiguration{}
			err := getAPIEndpoint(rt.configMap.data, nodeName, &cfg.APIEndpoint)
			if rt.expectedError != (err != nil) {
				t.Errorf("unexpected return err from getInitConfigurationFromCluster: %v", err)
				return
			}
			if rt.expectedError {
				return
			}

			if cfg.APIEndpoint.AdvertiseAddress != "1.2.3.4" || cfg.APIEndpoint.BindPort != 1234 {
				t.Errorf("invalid cfg.APIEndpoint")
			}
		})
	}
}

func TestGetComponentConfigs(t *testing.T) {
	var tests = []struct {
		name          string
		configMaps    []fakeConfigMap
		expectedError bool
	}{
		{
			name: "valid",
			configMaps: []fakeConfigMap{
				{
					name: kubeadmconstants.KubeProxyConfigMap, // Kube-proxy component config from corresponding ConfigMap.
					data: map[string]string{
						kubeadmconstants.KubeProxyConfigMapKey: string(cfgFiles["Kube-proxy_componentconfig"]),
					},
				},
				{
					name: kubeadmconstants.GetKubeletConfigMapName(k8sVersion), // Kubelet component config from corresponding ConfigMap.
					data: map[string]string{
						kubeadmconstants.KubeletBaseConfigurationConfigMapKey: string(cfgFiles["Kubelet_componentconfig"]),
					},
				},
			},
		},
		{
			name: "invalid - No kubelet component config ConfigMap",
			configMaps: []fakeConfigMap{
				{
					name: kubeadmconstants.KubeProxyConfigMap,
					data: map[string]string{
						kubeadmconstants.KubeProxyConfigMapKey: string(cfgFiles["Kube-proxy_componentconfig"]),
					},
				},
			},
			expectedError: true,
		},
		{
			name: "invalid - No kube-proxy component config ConfigMap",
			configMaps: []fakeConfigMap{
				{
					name: kubeadmconstants.GetKubeletConfigMapName(k8sVersion),
					data: map[string]string{
						kubeadmconstants.KubeletBaseConfigurationConfigMapKey: string(cfgFiles["Kubelet_componentconfig"]),
					},
				},
			},
			expectedError: true,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset()

			for _, c := range rt.configMaps {
				err := c.create(client)
				if err != nil {
					t.Errorf("couldn't create ConfigMap %s", c.name)
					return
				}
			}

			cfg := &kubeadmapi.InitConfiguration{
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					KubernetesVersion: k8sVersionString,
				},
			}
			err := getComponentConfigs(client, &cfg.ClusterConfiguration)
			if rt.expectedError != (err != nil) {
				t.Errorf("unexpected return err from getInitConfigurationFromCluster: %v", err)
				return
			}
			if rt.expectedError {
				return
			}

			// Test expected values in InitConfiguration
			if cfg.ComponentConfigs.Kubelet == nil {
				t.Errorf("invalid cfg.ComponentConfigs.Kubelet")
			}
			if cfg.ComponentConfigs.KubeProxy == nil {
				t.Errorf("invalid cfg.ComponentConfigs.KubeProxy")
			}
		})
	}
}

func TestGetInitConfigurationFromCluster(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	var tests = []struct {
		name            string
		fileContents    []byte
		node            *v1.Node
		configMaps      []fakeConfigMap
		newControlPlane bool
		expectedError   bool
	}{
		{
			name:          "invalid - No kubeadm-config ConfigMap",
			expectedError: true,
		},
		{
			name: "invalid - No CLusterConfiguration in kubeadm-config ConfigMap",
			configMaps: []fakeConfigMap{
				{
					name: kubeadmconstants.InitConfigurationConfigMap, // ClusterConfiguration from kubeadm-config.
					data: map[string]string{},
				},
			},
			expectedError: true,
		},
		{
			name: "valid - new control plane == false", // InitConfiguration composed with data from different places, with also node specific information from ClusterStatus and node
			configMaps: []fakeConfigMap{
				{
					name: kubeadmconstants.InitConfigurationConfigMap, // ClusterConfiguration from kubeadm-config.
					data: map[string]string{
						kubeadmconstants.ClusterConfigurationConfigMapKey: string(cfgFiles["ClusterConfiguration_v1alpha3"]),
						kubeadmconstants.ClusterStatusConfigMapKey:        string(cfgFiles["ClusterStatus_v1alpha3"]),
					},
				},
				{
					name: kubeadmconstants.KubeProxyConfigMap, // Kube-proxy component config from corresponding ConfigMap.
					data: map[string]string{
						kubeadmconstants.KubeProxyConfigMapKey: string(cfgFiles["Kube-proxy_componentconfig"]),
					},
				},
				{
					name: kubeadmconstants.GetKubeletConfigMapName(k8sVersion), // Kubelet component config from corresponding ConfigMap.
					data: map[string]string{
						kubeadmconstants.KubeletBaseConfigurationConfigMapKey: string(cfgFiles["Kubelet_componentconfig"]),
					},
				},
			},
			fileContents: kubeletConfFiles["configWithEmbeddedCert"],
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: nodeName,
					Annotations: map[string]string{
						kubeadmconstants.AnnotationKubeadmCRISocket: "myCRIsocket",
					},
				},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{kubeadmconstants.MasterTaint},
				},
			},
		},
		{
			name: "valid - new control plane == true", // InitConfiguration composed with data from different places, without node specific information
			configMaps: []fakeConfigMap{
				{
					name: kubeadmconstants.InitConfigurationConfigMap, // ClusterConfiguration from kubeadm-config.
					data: map[string]string{
						kubeadmconstants.ClusterConfigurationConfigMapKey: string(cfgFiles["ClusterConfiguration_v1alpha3"]),
					},
				},
				{
					name: kubeadmconstants.KubeProxyConfigMap, // Kube-proxy component config from corresponding ConfigMap.
					data: map[string]string{
						kubeadmconstants.KubeProxyConfigMapKey: string(cfgFiles["Kube-proxy_componentconfig"]),
					},
				},
				{
					name: kubeadmconstants.GetKubeletConfigMapName(k8sVersion), // Kubelet component config from corresponding ConfigMap.
					data: map[string]string{
						kubeadmconstants.KubeletBaseConfigurationConfigMapKey: string(cfgFiles["Kubelet_componentconfig"]),
					},
				},
			},
			newControlPlane: true,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			cfgPath := filepath.Join(tmpdir, kubeadmconstants.KubeletKubeConfigFileName)
			if len(rt.fileContents) > 0 {
				err := ioutil.WriteFile(cfgPath, rt.fileContents, 0644)
				if err != nil {
					t.Errorf("Couldn't create file")
					return
				}
			}

			client := clientsetfake.NewSimpleClientset()

			if rt.node != nil {
				_, err := client.CoreV1().Nodes().Create(rt.node)
				if err != nil {
					t.Errorf("couldn't create Node")
					return
				}
			}

			for _, c := range rt.configMaps {
				err := c.create(client)
				if err != nil {
					t.Errorf("couldn't create ConfigMap %s", c.name)
					return
				}
			}

			cfg, err := getInitConfigurationFromCluster(tmpdir, client, rt.newControlPlane)
			if rt.expectedError != (err != nil) {
				t.Errorf("unexpected return err from getInitConfigurationFromCluster: %v", err)
				return
			}
			if rt.expectedError {
				return
			}

			// Test expected values in InitConfiguration
			if cfg == nil {
				t.Errorf("unexpected nil return value")
			}
			if cfg.ClusterConfiguration.KubernetesVersion != k8sVersionString {
				t.Errorf("invalid ClusterConfiguration.KubernetesVersion")
			}
			if !rt.newControlPlane && (cfg.APIEndpoint.AdvertiseAddress != "1.2.3.4" || cfg.APIEndpoint.BindPort != 1234) {
				t.Errorf("invalid cfg.APIEndpoint")
			}
			if cfg.ComponentConfigs.Kubelet == nil {
				t.Errorf("invalid cfg.ComponentConfigs.Kubelet")
			}
			if cfg.ComponentConfigs.KubeProxy == nil {
				t.Errorf("invalid cfg.ComponentConfigs.KubeProxy")
			}
		})
	}
}

type fakeConfigMap struct {
	name string
	data map[string]string
}

func (c *fakeConfigMap) create(client clientset.Interface) error {
	return apiclient.CreateOrUpdateConfigMap(client, &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      c.name,
			Namespace: metav1.NamespaceSystem,
		},
		Data: c.data,
	})
}
