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
	_ "embed"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	authv1 "k8s.io/api/authentication/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1old "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta3"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
	testresources "k8s.io/kubernetes/cmd/kubeadm/test/resources"
)

var k8sVersionString = kubeadmconstants.MinimumControlPlaneVersion.String()
var nodeName = "mynode"
var cfgFiles = map[string][]byte{
	"InitConfiguration_v1beta3": []byte(fmt.Sprintf(`
apiVersion: %s
kind: InitConfiguration
`, kubeadmapiv1old.SchemeGroupVersion.String())),
	"ClusterConfiguration_v1beta3": []byte(fmt.Sprintf(`
apiVersion: %s
kind: ClusterConfiguration
kubernetesVersion: %s
`, kubeadmapiv1old.SchemeGroupVersion.String(), k8sVersionString)),
	"InitConfiguration_v1beta4": []byte(fmt.Sprintf(`
apiVersion: %s
kind: InitConfiguration
`, kubeadmapiv1.SchemeGroupVersion.String())),
	"ClusterConfiguration_v1beta4": []byte(fmt.Sprintf(`
apiVersion: %s
kind: ClusterConfiguration
kubernetesVersion: %s
`, kubeadmapiv1.SchemeGroupVersion.String(), k8sVersionString)),
	"Kube-proxy_componentconfig": []byte(`
apiVersion: kubeproxy.config.k8s.io/v1alpha1
kind: KubeProxyConfiguration
`),
	"Kubelet_componentconfig": []byte(`
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
`),
}

var (
	//go:embed testdata/kubelet-without-cert.yaml
	configWithoutCert []byte

	//go:embed testdata/kubelet-with-embedded-cert.yaml
	configWithEmbeddedCert []byte

	//go:embed testdata/kubelet-with-linked-cert.yaml
	configWithLinkedCert []byte

	//go:embed testdata/kubelet-with-invalid-context.yaml
	configWithInvalidContext []byte

	//go:embed testdata/kubelet-with-invalid-user.yaml
	configWithInvalidUser []byte
)

//go:embed testdata/mynode.pem
var mynodePem []byte

func TestGetNodeNameFromKubeletConfig(t *testing.T) {
	tmpdir, err := os.MkdirTemp("", "")
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
			kubeconfigContent: configWithEmbeddedCert,
		},
		{
			name:              "invalid - linked cert missing",
			kubeconfigContent: configWithLinkedCert,
			expectedError:     true,
		},
		{
			name:              "valid - with linked cert",
			kubeconfigContent: configWithLinkedCert,
			pemContent:        mynodePem,
		},
		{
			name:              "invalid - without embedded or linked X509Cert",
			kubeconfigContent: configWithoutCert,
			expectedError:     true,
		},
		{
			name:              "invalid - the current context is invalid",
			kubeconfigContent: configWithInvalidContext,
			expectedError:     true,
		},
		{
			name:              "invalid - the user of the current context is invalid",
			kubeconfigContent: configWithInvalidUser,
			expectedError:     true,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t2 *testing.T) {
			if len(rt.pemContent) > 0 {
				pemPath := filepath.Join(tmpdir, "kubelet.pem")
				err := os.WriteFile(pemPath, rt.pemContent, 0644)
				if err != nil {
					t.Errorf("Couldn't create pem file: %v", err)
					return
				}
				rt.kubeconfigContent = []byte(strings.Replace(string(rt.kubeconfigContent), "kubelet.pem", pemPath, -1))
			}

			kubeconfigPath := filepath.Join(tmpdir, kubeadmconstants.KubeletKubeConfigFileName)
			err := os.WriteFile(kubeconfigPath, rt.kubeconfigContent, 0644)
			if err != nil {
				t.Errorf("Couldn't create kubeconfig: %v", err)
				return
			}

			name, err := getNodeNameFromKubeletConfig(kubeconfigPath)
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
	tmpdir, err := os.MkdirTemp("", "")
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
			fileContents: configWithEmbeddedCert,
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
			fileContents:  configWithEmbeddedCert,
			expectedError: true,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t2 *testing.T) {
			cfgPath := filepath.Join(tmpdir, kubeadmconstants.KubeletKubeConfigFileName)
			if len(rt.fileContents) > 0 {
				err := os.WriteFile(cfgPath, rt.fileContents, 0644)
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
			err = GetNodeRegistration(cfgPath, client, &cfg.NodeRegistration, &cfg.ClusterConfiguration)
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
		expectedEndpoint *kubeadmapi.APIEndpoint
		expectedErr      bool
	}{
		{
			name:        "no pod annotations",
			nodeName:    nodeName,
			expectedErr: true,
		},
		{
			name:     "valid ipv4 endpoint in pod annotation",
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
			name:     "invalid ipv4 endpoint in pod annotation",
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
			name:     "invalid negative port with ipv4 address in pod annotation",
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
			name:     "invalid high port with ipv4 address in pod annotation",
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
			name:     "valid ipv6 endpoint in pod annotation",
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
			name:     "invalid ipv6 endpoint in pod annotation",
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
			name:     "invalid negative port with ipv6 address in pod annotation",
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
			apiEndpoint := kubeadmapi.APIEndpoint{}
			err := getAPIEndpointWithRetry(client, rt.nodeName, &apiEndpoint,
				time.Millisecond*10, time.Millisecond*100)
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
	tmpdir, err := os.MkdirTemp("", "")
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
			name: "valid v1beta4 - new control plane == false", // InitConfiguration composed with data from different places, with also node specific information
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
						kubeadmconstants.ClusterConfigurationConfigMapKey: string(cfgFiles["ClusterConfiguration_v1beta4"]),
					},
				},
				{
					Name: kubeadmconstants.KubeProxyConfigMap, // Kube-proxy component config from corresponding ConfigMap.
					Data: map[string]string{
						kubeadmconstants.KubeProxyConfigMapKey: string(cfgFiles["Kube-proxy_componentconfig"]),
					},
				},
				{
					Name: kubeadmconstants.KubeletBaseConfigurationConfigMap, // Kubelet component config from corresponding ConfigMap.
					Data: map[string]string{
						kubeadmconstants.KubeletBaseConfigurationConfigMapKey: string(cfgFiles["Kubelet_componentconfig"]),
					},
				},
			},
			fileContents: configWithEmbeddedCert,
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
			name: "valid v1beta3 - new control plane == true", // InitConfiguration composed with data from different places, without node specific information
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
						kubeadmconstants.ClusterConfigurationConfigMapKey: string(cfgFiles["ClusterConfiguration_v1beta4"]),
					},
				},
				{
					Name: kubeadmconstants.KubeProxyConfigMap, // Kube-proxy component config from corresponding ConfigMap.
					Data: map[string]string{
						kubeadmconstants.KubeProxyConfigMapKey: string(cfgFiles["Kube-proxy_componentconfig"]),
					},
				},
				{
					Name: kubeadmconstants.KubeletBaseConfigurationConfigMap, // Kubelet component config from corresponding ConfigMap.
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
				err := os.WriteFile(cfgPath, rt.fileContents, 0644)
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

			cfg, err := getInitConfigurationFromCluster(tmpdir, client, rt.newControlPlane, false)
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
			if cfg.NodeRegistration.ImagePullPolicy != kubeadmapiv1.DefaultImagePullPolicy {
				t.Errorf("invalid cfg.NodeRegistration.ImagePullPolicy %v", cfg.NodeRegistration.ImagePullPolicy)
			}
			if !rt.newControlPlane && (cfg.LocalAPIEndpoint.AdvertiseAddress != "1.2.3.4" || cfg.LocalAPIEndpoint.BindPort != 1234) {
				t.Errorf("invalid cfg.LocalAPIEndpoint: %v", cfg.LocalAPIEndpoint)
			}
			if !rt.newControlPlane && (cfg.NodeRegistration.Name != nodeName || cfg.NodeRegistration.CRISocket != "myCRIsocket" || len(cfg.NodeRegistration.Taints) != 1) {
				t.Errorf("invalid cfg.NodeRegistration: %v", cfg.NodeRegistration)
			}
			if rt.newControlPlane && len(cfg.NodeRegistration.CRISocket) > 0 {
				t.Errorf("invalid cfg.NodeRegistration.CRISocket: expected empty CRISocket, but got %v", cfg.NodeRegistration.CRISocket)
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
					Component:   kubeadmconstants.KubeAPIServer,
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
					Component:   kubeadmconstants.KubeAPIServer,
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
			err := getAPIEndpointFromPodAnnotation(client, rt.nodeName, &apiEndpoint,
				time.Millisecond*10, time.Millisecond*100)
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
					Component:   kubeadmconstants.KubeAPIServer,
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
					Component:   kubeadmconstants.KubeAPIServer,
					Annotations: map[string]string{kubeadmconstants.KubeAPIServerAdvertiseAddressEndpointAnnotationKey: "1.2.3.4:1234"},
				},
				{
					Component: kubeadmconstants.KubeAPIServer,
				},
			},
			expectedErr: true,
		},
		{
			name:     "two pods: different annotations",
			nodeName: nodeName,
			pods: []testresources.FakeStaticPod{
				{
					Component:   kubeadmconstants.KubeAPIServer,
					Annotations: map[string]string{kubeadmconstants.KubeAPIServerAdvertiseAddressEndpointAnnotationKey: "1.2.3.4:1234"},
				},
				{
					Component:   kubeadmconstants.KubeAPIServer,
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
					Component: kubeadmconstants.KubeAPIServer,
				},
				{
					Component: kubeadmconstants.KubeAPIServer,
				},
			},
			expectedErr: true,
		},
		{
			name:     "exactly one pod with annotation; request fails",
			nodeName: nodeName,
			pods: []testresources.FakeStaticPod{
				{
					Component:   kubeadmconstants.KubeAPIServer,
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
			endpoint, err := getRawAPIEndpointFromPodAnnotationWithoutRetry(context.Background(), client, rt.nodeName)
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

func TestGetNodeNameFromSSR(t *testing.T) {
	var tests = []struct {
		name             string
		clientSetup      func(*clientsetfake.Clientset)
		expectedNodeName string
		expectedError    bool
	}{
		{
			name: "valid node name",
			clientSetup: func(clientset *clientsetfake.Clientset) {
				clientset.PrependReactor("create", "selfsubjectreviews",
					func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
						obj := &authv1.SelfSubjectReview{
							Status: authv1.SelfSubjectReviewStatus{
								UserInfo: authv1.UserInfo{
									Username: kubeadmconstants.NodesUserPrefix + "foo",
								},
							},
						}
						return true, obj, nil
					})
			},
			expectedNodeName: "foo",
			expectedError:    false,
		},
		{
			name: "SSR created but client is not a node",
			clientSetup: func(clientset *clientsetfake.Clientset) {
				clientset.PrependReactor("create", "selfsubjectreviews",
					func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
						obj := &authv1.SelfSubjectReview{
							Status: authv1.SelfSubjectReviewStatus{
								UserInfo: authv1.UserInfo{
									Username: "foo",
								},
							},
						}
						return true, obj, nil
					})
			},
			expectedNodeName: "",
			expectedError:    true,
		},
		{
			name: "error creating SSR",
			clientSetup: func(clientset *clientsetfake.Clientset) {
				clientset.PrependReactor("create", "selfsubjectreviews",
					func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
						return true, nil, errors.New("")
					})
			},
			expectedNodeName: "",
			expectedError:    true,
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset()
			rt.clientSetup(client)

			nodeName, err := getNodeNameFromSSR(client)

			if (err != nil) != rt.expectedError {
				t.Fatalf("expected error: %+v, got: %+v", rt.expectedError, err)
			}
			if rt.expectedNodeName != nodeName {
				t.Fatalf("expected nodeName: %s, got: %s", rt.expectedNodeName, nodeName)
			}
		})
	}
}
