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
	"context"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/pkg/errors"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apimachinery/pkg/util/wait"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	testresources "k8s.io/kubernetes/cmd/kubeadm/test/resources"
)

var k8sVersionString = kubeadmconstants.MinimumControlPlaneVersion.String()
var k8sVersion = version.MustParseGeneric(k8sVersionString)
var nodeName = "mynode"
var cfgFiles = map[string][]byte{
	"InitConfiguration_v1beta1": []byte(`
apiVersion: kubeadm.k8s.io/v1beta1
kind: InitConfiguration
`),
	"ClusterConfiguration_v1beta1": []byte(`
apiVersion: kubeadm.k8s.io/v1beta1
kind: ClusterConfiguration
kubernetesVersion: ` + k8sVersionString + `
`),
	"InitConfiguration_v1beta2": []byte(`
apiVersion: kubeadm.k8s.io/v1beta2
kind: InitConfiguration
`),
	"ClusterConfiguration_v1beta2": []byte(`
apiVersion: kubeadm.k8s.io/v1beta2
kind: ClusterConfiguration
kubernetesVersion: ` + k8sVersionString + `
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
	"configWithInvalidContext": []byte(`
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
current-context: invalidContext
kind: Config
preferences: {}
users:
- name: system:node:mynode
  user:
      client-certificate: kubelet.pem
`),
	"configWithInvalidUser": []byte(`
apiVersion: v1
clusters:
- cluster:
    server: https://10.0.2.15:6443
  name: kubernetes
contexts:
- context:
    cluster: kubernetes
    user: invalidUser
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
		{
			name:              "invalid - the current context is invalid",
			kubeconfigContent: kubeletConfFiles["configWithInvalidContext"],
			expectedError:     true,
		},
		{
			name:              "invalid - the user of the current context is invalid",
			kubeconfigContent: kubeletConfFiles["configWithInvalidUser"],
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
					Taints: []v1.Taint{kubeadmconstants.ControlPlaneTaint},
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
				_, err := client.CoreV1().Nodes().Create(context.TODO(), rt.node, metav1.CreateOptions{})
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

func TestGetAPIEndpointWithBackoff(t *testing.T) {
	var tests = []struct {
		name             string
		nodeName         string
		staticPod        *testresources.FakeStaticPod
		configMap        *testresources.FakeConfigMap
		expectedEndpoint *kubeadmapi.APIEndpoint
		expectedErr      bool
	}{
		{
			name:        "no pod annotations; no ClusterStatus",
			nodeName:    nodeName,
			expectedErr: true,
		},
		{
			name:     "valid ipv4 endpoint in pod annotation; no ClusterStatus",
			nodeName: nodeName,
			staticPod: &testresources.FakeStaticPod{
				Component: kubeadmconstants.KubeAPIServer,
				Annotations: map[string]string{
					kubeadmconstants.KubeAPIServerAdvertiseAddressEndpointAnnotationKey: "1.2.3.4:1234",
				},
			},
			expectedEndpoint: &kubeadmapi.APIEndpoint{AdvertiseAddress: "1.2.3.4", BindPort: 1234},
		},
		{
			name:     "invalid ipv4 endpoint in pod annotation; no ClusterStatus",
			nodeName: nodeName,
			staticPod: &testresources.FakeStaticPod{
				Component: kubeadmconstants.KubeAPIServer,
				Annotations: map[string]string{
					kubeadmconstants.KubeAPIServerAdvertiseAddressEndpointAnnotationKey: "1.2.3::1234",
				},
			},
			expectedErr: true,
		},
		{
			name:     "invalid negative port with ipv4 address in pod annotation; no ClusterStatus",
			nodeName: nodeName,
			staticPod: &testresources.FakeStaticPod{
				Component: kubeadmconstants.KubeAPIServer,
				Annotations: map[string]string{
					kubeadmconstants.KubeAPIServerAdvertiseAddressEndpointAnnotationKey: "1.2.3.4:-1234",
				},
			},
			expectedErr: true,
		},
		{
			name:     "invalid high port with ipv4 address in pod annotation; no ClusterStatus",
			nodeName: nodeName,
			staticPod: &testresources.FakeStaticPod{
				Component: kubeadmconstants.KubeAPIServer,
				Annotations: map[string]string{
					kubeadmconstants.KubeAPIServerAdvertiseAddressEndpointAnnotationKey: "1.2.3.4:65536",
				},
			},
			expectedErr: true,
		},
		{
			name:     "valid ipv6 endpoint in pod annotation; no ClusterStatus",
			nodeName: nodeName,
			staticPod: &testresources.FakeStaticPod{
				Component: kubeadmconstants.KubeAPIServer,
				Annotations: map[string]string{
					kubeadmconstants.KubeAPIServerAdvertiseAddressEndpointAnnotationKey: "[::1]:1234",
				},
			},
			expectedEndpoint: &kubeadmapi.APIEndpoint{AdvertiseAddress: "::1", BindPort: 1234},
		},
		{
			name:     "invalid ipv6 endpoint in pod annotation; no ClusterStatus",
			nodeName: nodeName,
			staticPod: &testresources.FakeStaticPod{
				Component: kubeadmconstants.KubeAPIServer,
				Annotations: map[string]string{
					kubeadmconstants.KubeAPIServerAdvertiseAddressEndpointAnnotationKey: "[::1:1234",
				},
			},
			expectedErr: true,
		},
		{
			name:     "invalid negative port with ipv6 address in pod annotation; no ClusterStatus",
			nodeName: nodeName,
			staticPod: &testresources.FakeStaticPod{
				Component: kubeadmconstants.KubeAPIServer,
				Annotations: map[string]string{
					kubeadmconstants.KubeAPIServerAdvertiseAddressEndpointAnnotationKey: "[::1]:-1234",
				},
			},
			expectedErr: true,
		},
		{
			name:     "invalid high port with ipv6 address in pod annotation",
			nodeName: nodeName,
			staticPod: &testresources.FakeStaticPod{
				Component: kubeadmconstants.KubeAPIServer,
				Annotations: map[string]string{
					kubeadmconstants.KubeAPIServerAdvertiseAddressEndpointAnnotationKey: "[::1]:65536",
				},
			},
			expectedErr: true,
		},
		{
			name:             "no pod annotations; ClusterStatus with valid ipv4 endpoint",
			nodeName:         nodeName,
			configMap:        testresources.ClusterStatusWithAPIEndpoint(nodeName, kubeadmapi.APIEndpoint{AdvertiseAddress: "1.2.3.4", BindPort: 1234}),
			expectedEndpoint: &kubeadmapi.APIEndpoint{AdvertiseAddress: "1.2.3.4", BindPort: 1234},
		},
		{
			name:     "invalid ipv4 endpoint in pod annotation; ClusterStatus with valid ipv4 endpoint",
			nodeName: nodeName,
			staticPod: &testresources.FakeStaticPod{
				Component: kubeadmconstants.KubeAPIServer,
				Annotations: map[string]string{
					kubeadmconstants.KubeAPIServerAdvertiseAddressEndpointAnnotationKey: "1.2.3::1234",
				},
			},
			configMap:        testresources.ClusterStatusWithAPIEndpoint(nodeName, kubeadmapi.APIEndpoint{AdvertiseAddress: "1.2.3.4", BindPort: 1234}),
			expectedEndpoint: &kubeadmapi.APIEndpoint{AdvertiseAddress: "1.2.3.4", BindPort: 1234},
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset()
			if rt.staticPod != nil {
				rt.staticPod.NodeName = rt.nodeName
				if err := rt.staticPod.Create(client); err != nil {
					t.Error("could not create static pod")
					return
				}
			}
			if rt.configMap != nil {
				if err := rt.configMap.Create(client); err != nil {
					t.Error("could not create ConfigMap")
					return
				}
			}
			apiEndpoint := kubeadm.APIEndpoint{}
			err := getAPIEndpointWithBackoff(client, rt.nodeName, &apiEndpoint, wait.Backoff{Duration: 0, Jitter: 0, Steps: 1})
			if err != nil && !rt.expectedErr {
				t.Errorf("got error %q; was expecting no errors", err)
				return
			} else if err == nil && rt.expectedErr {
				t.Error("got no error; was expecting an error")
				return
			}

			if rt.expectedEndpoint != nil && !reflect.DeepEqual(apiEndpoint, *rt.expectedEndpoint) {
				t.Errorf("expected API endpoint: %v; got %v", rt.expectedEndpoint, apiEndpoint)
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
		staticPods      []testresources.FakeStaticPod
		configMaps      []testresources.FakeConfigMap
		newControlPlane bool
		expectedError   bool
	}{
		{
			name:          "invalid - No kubeadm-config ConfigMap",
			expectedError: true,
		},
		{
			name: "invalid - No ClusterConfiguration in kubeadm-config ConfigMap",
			configMaps: []testresources.FakeConfigMap{
				{
					Name: kubeadmconstants.KubeadmConfigConfigMap, // ClusterConfiguration from kubeadm-config.
					Data: map[string]string{},
				},
			},
			expectedError: true,
		},
		{
			name: "valid v1beta1 - new control plane == false", // InitConfiguration composed with data from different places, with also node specific information from ClusterStatus and node
			staticPods: []testresources.FakeStaticPod{
				{
					NodeName:  nodeName,
					Component: kubeadmconstants.KubeAPIServer,
					Annotations: map[string]string{
						kubeadmconstants.KubeAPIServerAdvertiseAddressEndpointAnnotationKey: "1.2.3.4:1234",
					},
				},
			},
			configMaps: []testresources.FakeConfigMap{
				{
					Name: kubeadmconstants.KubeadmConfigConfigMap, // ClusterConfiguration from kubeadm-config.
					Data: map[string]string{
						kubeadmconstants.ClusterConfigurationConfigMapKey: string(cfgFiles["ClusterConfiguration_v1beta1"]),
					},
				},
				{
					Name: kubeadmconstants.KubeProxyConfigMap, // Kube-proxy component config from corresponding ConfigMap.
					Data: map[string]string{
						kubeadmconstants.KubeProxyConfigMapKey: string(cfgFiles["Kube-proxy_componentconfig"]),
					},
				},
				{
					Name: kubeadmconstants.GetKubeletConfigMapName(k8sVersion), // Kubelet component config from corresponding ConfigMap.
					Data: map[string]string{
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
					Taints: []v1.Taint{kubeadmconstants.ControlPlaneTaint},
				},
			},
		},
		{
			name: "valid v1beta1 - new control plane == true", // InitConfiguration composed with data from different places, without node specific information
			staticPods: []testresources.FakeStaticPod{
				{
					NodeName:  nodeName,
					Component: kubeadmconstants.KubeAPIServer,
					Annotations: map[string]string{
						kubeadmconstants.KubeAPIServerAdvertiseAddressEndpointAnnotationKey: "1.2.3.4:1234",
					},
				},
			},
			configMaps: []testresources.FakeConfigMap{
				{
					Name: kubeadmconstants.KubeadmConfigConfigMap, // ClusterConfiguration from kubeadm-config.
					Data: map[string]string{
						kubeadmconstants.ClusterConfigurationConfigMapKey: string(cfgFiles["ClusterConfiguration_v1beta1"]),
					},
				},
				{
					Name: kubeadmconstants.KubeProxyConfigMap, // Kube-proxy component config from corresponding ConfigMap.
					Data: map[string]string{
						kubeadmconstants.KubeProxyConfigMapKey: string(cfgFiles["Kube-proxy_componentconfig"]),
					},
				},
				{
					Name: kubeadmconstants.GetKubeletConfigMapName(k8sVersion), // Kubelet component config from corresponding ConfigMap.
					Data: map[string]string{
						kubeadmconstants.KubeletBaseConfigurationConfigMapKey: string(cfgFiles["Kubelet_componentconfig"]),
					},
				},
			},
			newControlPlane: true,
		},
		{
			name: "valid v1beta2 - new control plane == false", // InitConfiguration composed with data from different places, with also node specific information from ClusterStatus and node
			staticPods: []testresources.FakeStaticPod{
				{
					NodeName:  nodeName,
					Component: kubeadmconstants.KubeAPIServer,
					Annotations: map[string]string{
						kubeadmconstants.KubeAPIServerAdvertiseAddressEndpointAnnotationKey: "1.2.3.4:1234",
					},
				},
			},
			configMaps: []testresources.FakeConfigMap{
				{
					Name: kubeadmconstants.KubeadmConfigConfigMap, // ClusterConfiguration from kubeadm-config.
					Data: map[string]string{
						kubeadmconstants.ClusterConfigurationConfigMapKey: string(cfgFiles["ClusterConfiguration_v1beta2"]),
					},
				},
				{
					Name: kubeadmconstants.KubeProxyConfigMap, // Kube-proxy component config from corresponding ConfigMap.
					Data: map[string]string{
						kubeadmconstants.KubeProxyConfigMapKey: string(cfgFiles["Kube-proxy_componentconfig"]),
					},
				},
				{
					Name: kubeadmconstants.GetKubeletConfigMapName(k8sVersion), // Kubelet component config from corresponding ConfigMap.
					Data: map[string]string{
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
					Taints: []v1.Taint{kubeadmconstants.ControlPlaneTaint},
				},
			},
		},
		{
			name: "valid v1beta2 - new control plane == true", // InitConfiguration composed with data from different places, without node specific information
			staticPods: []testresources.FakeStaticPod{
				{
					NodeName:  nodeName,
					Component: kubeadmconstants.KubeAPIServer,
					Annotations: map[string]string{
						kubeadmconstants.KubeAPIServerAdvertiseAddressEndpointAnnotationKey: "1.2.3.4:1234",
					},
				},
			},
			configMaps: []testresources.FakeConfigMap{
				{
					Name: kubeadmconstants.KubeadmConfigConfigMap, // ClusterConfiguration from kubeadm-config.
					Data: map[string]string{
						kubeadmconstants.ClusterConfigurationConfigMapKey: string(cfgFiles["ClusterConfiguration_v1beta2"]),
					},
				},
				{
					Name: kubeadmconstants.KubeProxyConfigMap, // Kube-proxy component config from corresponding ConfigMap.
					Data: map[string]string{
						kubeadmconstants.KubeProxyConfigMapKey: string(cfgFiles["Kube-proxy_componentconfig"]),
					},
				},
				{
					Name: kubeadmconstants.GetKubeletConfigMapName(k8sVersion), // Kubelet component config from corresponding ConfigMap.
					Data: map[string]string{
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
				_, err := client.CoreV1().Nodes().Create(context.TODO(), rt.node, metav1.CreateOptions{})
				if err != nil {
					t.Errorf("couldn't create Node")
					return
				}
			}

			for _, p := range rt.staticPods {
				err := p.Create(client)
				if err != nil {
					t.Errorf("couldn't create pod for nodename %s", p.NodeName)
					return
				}
			}

			for _, c := range rt.configMaps {
				err := c.Create(client)
				if err != nil {
					t.Errorf("couldn't create ConfigMap %s", c.Name)
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
				return
			}
			if cfg.ClusterConfiguration.KubernetesVersion != k8sVersionString {
				t.Errorf("invalid ClusterConfiguration.KubernetesVersion")
			}
			if !rt.newControlPlane && (cfg.LocalAPIEndpoint.AdvertiseAddress != "1.2.3.4" || cfg.LocalAPIEndpoint.BindPort != 1234) {
				t.Errorf("invalid cfg.LocalAPIEndpoint: %v", cfg.LocalAPIEndpoint)
			}
			if _, ok := cfg.ComponentConfigs[componentconfigs.KubeletGroup]; !ok {
				t.Errorf("no cfg.ComponentConfigs[%q]", componentconfigs.KubeletGroup)
			}
			if _, ok := cfg.ComponentConfigs[componentconfigs.KubeProxyGroup]; !ok {
				t.Errorf("no cfg.ComponentConfigs[%q]", componentconfigs.KubeProxyGroup)
			}
		})
	}
}

func TestGetGetClusterStatus(t *testing.T) {
	var tests = []struct {
		name          string
		configMaps    []testresources.FakeConfigMap
		expectedError bool
	}{
		{
			name: "invalid missing config map",
		},
		{
			name: "valid v1beta1",
			configMaps: []testresources.FakeConfigMap{
				{
					Name: kubeadmconstants.KubeadmConfigConfigMap,
					Data: map[string]string{
						kubeadmconstants.ClusterStatusConfigMapKey: string(cfgFiles["ClusterStatus_v1beta1"]),
					},
				},
			},
		},
		{
			name: "valid v1beta2",
			configMaps: []testresources.FakeConfigMap{
				{
					Name: kubeadmconstants.KubeadmConfigConfigMap,
					Data: map[string]string{
						kubeadmconstants.ClusterStatusConfigMapKey: string(cfgFiles["ClusterStatus_v1beta2"]),
					},
				},
			},
		},
		{
			name: "invalid missing ClusterStatusConfigMapKey in the config map",
			configMaps: []testresources.FakeConfigMap{
				{
					Name: kubeadmconstants.KubeadmConfigConfigMap,
					Data: map[string]string{},
				},
			},
			expectedError: true,
		},
		{
			name: "invalid wrong value in the config map",
			configMaps: []testresources.FakeConfigMap{
				{
					Name: kubeadmconstants.KubeadmConfigConfigMap,
					Data: map[string]string{
						kubeadmconstants.ClusterStatusConfigMapKey: "not a kubeadm type",
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
				err := c.Create(client)
				if err != nil {
					t.Errorf("couldn't create ConfigMap %s", c.Name)
					return
				}
			}

			_, err := GetClusterStatus(client)
			if rt.expectedError != (err != nil) {
				t.Errorf("unexpected return err from GetClusterStatus: %v", err)
				return
			}
			if rt.expectedError {
				return
			}
		})
	}
}

func TestGetAPIEndpointFromPodAnnotation(t *testing.T) {
	var tests = []struct {
		name             string
		nodeName         string
		pods             []testresources.FakeStaticPod
		clientSetup      func(*clientsetfake.Clientset)
		expectedEndpoint kubeadmapi.APIEndpoint
		expectedErr      bool
	}{
		{
			name:     "exactly one pod with annotation",
			nodeName: nodeName,
			pods: []testresources.FakeStaticPod{
				{
					Component:   constants.KubeAPIServer,
					Annotations: map[string]string{kubeadmconstants.KubeAPIServerAdvertiseAddressEndpointAnnotationKey: "1.2.3.4:1234"},
				},
			},
			expectedEndpoint: kubeadmapi.APIEndpoint{AdvertiseAddress: "1.2.3.4", BindPort: 1234},
		},
		{
			name:        "no pods with annotation",
			nodeName:    nodeName,
			expectedErr: true,
		},
		{
			name:     "exactly one pod with annotation; all requests fail",
			nodeName: nodeName,
			pods: []testresources.FakeStaticPod{
				{
					Component:   constants.KubeAPIServer,
					Annotations: map[string]string{kubeadmconstants.KubeAPIServerAdvertiseAddressEndpointAnnotationKey: "1.2.3.4:1234"},
				},
			},
			clientSetup: func(clientset *clientsetfake.Clientset) {
				clientset.PrependReactor("list", "pods", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, nil, apierrors.NewInternalError(errors.New("API server down"))
				})
			},
			expectedErr: true,
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset()
			for i, pod := range rt.pods {
				pod.NodeName = rt.nodeName
				if err := pod.CreateWithPodSuffix(client, strconv.Itoa(i)); err != nil {
					t.Errorf("error setting up test creating pod for node %q", pod.NodeName)
					return
				}
			}
			if rt.clientSetup != nil {
				rt.clientSetup(client)
			}
			apiEndpoint := kubeadmapi.APIEndpoint{}
			err := getAPIEndpointFromPodAnnotation(client, rt.nodeName, &apiEndpoint, wait.Backoff{Duration: 0, Jitter: 0, Steps: 1})
			if err != nil && !rt.expectedErr {
				t.Errorf("got error %v, but wasn't expecting any error", err)
				return
			} else if err == nil && rt.expectedErr {
				t.Error("didn't get any error; but was expecting an error")
				return
			} else if err != nil && rt.expectedErr {
				return
			}
			if !reflect.DeepEqual(apiEndpoint, rt.expectedEndpoint) {
				t.Errorf("expected API endpoint: %v; got %v", rt.expectedEndpoint, apiEndpoint)
			}
		})
	}
}

func TestGetRawAPIEndpointFromPodAnnotationWithoutRetry(t *testing.T) {
	var tests = []struct {
		name             string
		nodeName         string
		pods             []testresources.FakeStaticPod
		clientSetup      func(*clientsetfake.Clientset)
		expectedEndpoint string
		expectedErr      bool
	}{
		{
			name:        "no pods",
			nodeName:    nodeName,
			expectedErr: true,
		},
		{
			name:     "exactly one pod with annotation",
			nodeName: nodeName,
			pods: []testresources.FakeStaticPod{
				{
					Component:   constants.KubeAPIServer,
					Annotations: map[string]string{kubeadmconstants.KubeAPIServerAdvertiseAddressEndpointAnnotationKey: "1.2.3.4:1234"},
				},
			},
			expectedEndpoint: "1.2.3.4:1234",
		},
		{
			name:     "two pods: one with annotation, one missing annotation",
			nodeName: nodeName,
			pods: []testresources.FakeStaticPod{
				{
					Component:   constants.KubeAPIServer,
					Annotations: map[string]string{kubeadmconstants.KubeAPIServerAdvertiseAddressEndpointAnnotationKey: "1.2.3.4:1234"},
				},
				{
					Component: constants.KubeAPIServer,
				},
			},
			expectedErr: true,
		},
		{
			name:     "two pods: different annotations",
			nodeName: nodeName,
			pods: []testresources.FakeStaticPod{
				{
					Component:   constants.KubeAPIServer,
					Annotations: map[string]string{kubeadmconstants.KubeAPIServerAdvertiseAddressEndpointAnnotationKey: "1.2.3.4:1234"},
				},
				{
					Component:   constants.KubeAPIServer,
					Annotations: map[string]string{kubeadmconstants.KubeAPIServerAdvertiseAddressEndpointAnnotationKey: "1.2.3.5:1234"},
				},
			},
			expectedErr: true,
		},
		{
			name:     "two pods: both missing annotation",
			nodeName: nodeName,
			pods: []testresources.FakeStaticPod{
				{
					Component: constants.KubeAPIServer,
				},
				{
					Component: constants.KubeAPIServer,
				},
			},
			expectedErr: true,
		},
		{
			name:     "exactly one pod with annotation; request fails",
			nodeName: nodeName,
			pods: []testresources.FakeStaticPod{
				{
					Component:   constants.KubeAPIServer,
					Annotations: map[string]string{kubeadmconstants.KubeAPIServerAdvertiseAddressEndpointAnnotationKey: "1.2.3.4:1234"},
				},
			},
			clientSetup: func(clientset *clientsetfake.Clientset) {
				clientset.PrependReactor("list", "pods", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, nil, apierrors.NewInternalError(errors.New("API server down"))
				})
			},
			expectedErr: true,
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset()
			for i, pod := range rt.pods {
				pod.NodeName = rt.nodeName
				if err := pod.CreateWithPodSuffix(client, strconv.Itoa(i)); err != nil {
					t.Errorf("error setting up test creating pod for node %q", pod.NodeName)
					return
				}
			}
			if rt.clientSetup != nil {
				rt.clientSetup(client)
			}
			endpoint, err := getRawAPIEndpointFromPodAnnotationWithoutRetry(client, rt.nodeName)
			if err != nil && !rt.expectedErr {
				t.Errorf("got error %v, but wasn't expecting any error", err)
				return
			} else if err == nil && rt.expectedErr {
				t.Error("didn't get any error; but was expecting an error")
				return
			} else if err != nil && rt.expectedErr {
				return
			}
			if endpoint != rt.expectedEndpoint {
				t.Errorf("expected API endpoint: %v; got: %v", rt.expectedEndpoint, endpoint)
			}
		})
	}
}
