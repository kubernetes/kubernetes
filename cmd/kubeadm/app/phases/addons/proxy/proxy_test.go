/*
Copyright 2017 The Kubernetes Authors.

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

package proxy

import (
	"bytes"
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/lithammer/dedent"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	clientsetscheme "k8s.io/client-go/kubernetes/scheme"
	core "k8s.io/client-go/testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
)

func TestCompileManifests(t *testing.T) {
	var tests = []struct {
		name     string
		manifest string
		data     interface{}
	}{
		{
			name:     "KubeProxyConfigMap19",
			manifest: KubeProxyConfigMap19,
			data: struct {
				ControlPlaneEndpoint, ProxyConfig, ProxyConfigMap, ProxyConfigMapKey string
			}{
				ControlPlaneEndpoint: "foo",
				ProxyConfig:          "  bindAddress: 0.0.0.0\n  clusterCIDR: 192.168.1.1\n  enableProfiling: false",
				ProxyConfigMap:       "bar",
				ProxyConfigMapKey:    "baz",
			},
		},
		{
			name:     "KubeProxyDaemonSet19",
			manifest: KubeProxyDaemonSet19,
			data: struct{ Image, ProxyConfigMap, ProxyConfigMapKey string }{
				Image:             "foo",
				ProxyConfigMap:    "bar",
				ProxyConfigMapKey: "baz",
			},
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			_, err := kubeadmutil.ParseTemplate(rt.manifest, rt.data)
			if err != nil {
				t.Errorf("unexpected ParseTemplate failure: %+v", err)
			}
		})
	}
}

func TestEnsureProxyAddon(t *testing.T) {
	type SimulatedError int
	const (
		NoError SimulatedError = iota
		ServiceAccountError
		InvalidControlPlaneEndpoint
		IPv6SetBindAddress
	)

	var testCases = []struct {
		name           string
		simError       SimulatedError
		expErrString   string
		expBindAddr    string
		expClusterCIDR string
	}{
		{
			name:           "Successful proxy addon",
			simError:       NoError,
			expErrString:   "",
			expBindAddr:    "0.0.0.0",
			expClusterCIDR: "5.6.7.8/24",
		}, {
			name:           "Simulated service account error",
			simError:       ServiceAccountError,
			expErrString:   "error when creating kube-proxy service account",
			expBindAddr:    "0.0.0.0",
			expClusterCIDR: "5.6.7.8/24",
		}, {
			name:           "IPv6 AdvertiseAddress address",
			simError:       IPv6SetBindAddress,
			expErrString:   "",
			expBindAddr:    "::",
			expClusterCIDR: "2001:101::/96",
		},
	}

	// Override the default timeouts to be shorter
	defaultTimeouts := kubeadmapi.GetActiveTimeouts()
	defaultAPICallTimeout := defaultTimeouts.KubernetesAPICall
	defaultTimeouts.KubernetesAPICall = &metav1.Duration{Duration: time.Microsecond * 500}
	defer func() {
		defaultTimeouts.KubernetesAPICall = defaultAPICallTimeout
	}()

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create a fake client and set up default test configuration
			client := clientsetfake.NewSimpleClientset()

			// TODO: Consider using a YAML file instead for this that makes it possible to specify YAML documents for the ComponentConfigs
			initConfiguration, err := configutil.DefaultedStaticInitConfiguration()
			if err != nil {
				t.Errorf("test failed to convert external to internal version: %v", err)
				return
			}

			initConfiguration.LocalAPIEndpoint = kubeadmapi.APIEndpoint{
				AdvertiseAddress: "1.2.3.4",
				BindPort:         1234,
			}

			initConfiguration.ClusterConfiguration.Networking.PodSubnet = "5.6.7.8/24"
			initConfiguration.ClusterConfiguration.ImageRepository = "someRepo"

			// Simulate an error if necessary
			switch tc.simError {
			case ServiceAccountError:
				client.PrependReactor("create", "serviceaccounts", func(action core.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewUnauthorized("")
				})
			case InvalidControlPlaneEndpoint:
				initConfiguration.LocalAPIEndpoint.AdvertiseAddress = "1.2.3"
			case IPv6SetBindAddress:
				initConfiguration.LocalAPIEndpoint.AdvertiseAddress = "1:2::3:4"
				initConfiguration.ClusterConfiguration.Networking.PodSubnet = "2001:101::/48"
			}

			err = EnsureProxyAddon(&initConfiguration.ClusterConfiguration, &initConfiguration.LocalAPIEndpoint, client, os.Stdout, false)

			// Compare actual to expected errors
			actErr := "No error"
			if err != nil {
				actErr = err.Error()
			}
			expErr := "No error"
			if tc.expErrString != "" {
				expErr = tc.expErrString
			}
			if !strings.Contains(actErr, expErr) {
				t.Errorf(
					"%s test failed, expected: %s, got: %s",
					tc.name,
					expErr,
					actErr)
			}
		})
	}
}

func TestDaemonSetsHaveSystemNodeCriticalPriorityClassName(t *testing.T) {
	testCases := []struct {
		name     string
		manifest string
		data     interface{}
	}{
		{
			name:     "KubeProxyDaemonSet19",
			manifest: KubeProxyDaemonSet19,
			data: struct{ Image, ProxyConfigMap, ProxyConfigMapKey string }{
				Image:             "foo",
				ProxyConfigMap:    "foo",
				ProxyConfigMapKey: "foo",
			},
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			daemonSetBytes, _ := kubeadmutil.ParseTemplate(testCase.manifest, testCase.data)
			daemonSet := &apps.DaemonSet{}
			if err := runtime.DecodeInto(clientsetscheme.Codecs.UniversalDecoder(), daemonSetBytes, daemonSet); err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if daemonSet.Spec.Template.Spec.PriorityClassName != "system-node-critical" {
				t.Errorf("expected to see system-node-critical priority class name. Got %q instead", daemonSet.Spec.Template.Spec.PriorityClassName)
			}
		})
	}
}

func TestPrintOrCreateKubeProxyObjects(t *testing.T) {
	tests := []struct {
		name          string
		printManifest bool
		wantOut       string
		wantErr       bool
	}{
		{
			name:          "do not print manifest",
			printManifest: false,
			wantOut:       "[addons] Applied essential addon: kube-proxy\n",
			wantErr:       false,
		},
		{
			name:          "print manifest",
			printManifest: true,
			wantOut: dedent.Dedent(`---
apiVersion: v1
kind: ServiceAccount
metadata:
  creationTimestamp: null
  name: kube-proxy
  namespace: kube-system
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  creationTimestamp: null
  name: kubeadm:node-proxier
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: system:node-proxier
subjects:
- kind: ServiceAccount
  name: kube-proxy
  namespace: kube-system
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  creationTimestamp: null
  name: kube-proxy
  namespace: kube-system
rules:
- apiGroups:
  - ""
  resourceNames:
  - kube-proxy
  resources:
  - configmaps
  verbs:
  - get
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  creationTimestamp: null
  name: kube-proxy
  namespace: kube-system
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: kube-proxy
subjects:
- kind: Group
  name: system:bootstrappers:kubeadm:default-node-token
---
foo
---
bar
`),
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := &bytes.Buffer{}
			client := newMockClientForTest(t)
			cmByte := []byte{'\n', 'f', 'o', 'o', '\n'}
			dsByte := []byte{'\n', 'b', 'a', 'r', '\n'}
			if err := printOrCreateKubeProxyObjects(cmByte, dsByte, client, out, tt.printManifest); (err != nil) != tt.wantErr {
				t.Fatalf("printOrCreateKubeProxyObjects() error = %v, wantErr %v", err, tt.wantErr)
			}
			if gotOut := out.String(); gotOut != tt.wantOut {
				t.Fatalf("printOrCreateKubeProxyObjects() = %v, want %v", gotOut, tt.wantOut)
			}
		})
	}
}

func newMockClientForTest(t *testing.T) *clientsetfake.Clientset {
	client := clientsetfake.NewSimpleClientset()
	_, err := client.AppsV1().DaemonSets(metav1.NamespaceSystem).Create(context.TODO(), &apps.DaemonSet{
		TypeMeta: metav1.TypeMeta{
			Kind:       "DaemonSet",
			APIVersion: "apps/v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "kube-proxy",
			Namespace: metav1.NamespaceSystem,
			Labels: map[string]string{
				"k8s-app": "kube-proxy",
			},
		},
		Spec: apps.DaemonSetSpec{
			Template: v1.PodTemplateSpec{},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("error creating Daemonset: %v", err)
	}

	return client
}
