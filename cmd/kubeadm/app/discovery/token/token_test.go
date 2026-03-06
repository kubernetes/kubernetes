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

package token

import (
	_ "embed"
	"testing"
	"time"

	"github.com/pmezard/go-difflib/difflib"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/clientcmd"
	bootstrapapi "k8s.io/cluster-bootstrap/token/api"
	tokenjws "k8s.io/cluster-bootstrap/token/jws"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

var (
	//go:embed testdata/ca-cert.pem
	caCert string

	//go:embed testdata/expected-kubeconfig.yaml
	expectedKubeconfig string
)

func TestRetrieveValidatedConfigInfo(t *testing.T) {
	const caCertHash = "sha256:98be2e6d4d8a89aa308fb15de0c07e2531ce549c68dec1687cdd5c06f0826658"

	tests := []struct {
		name                     string
		tokenID                  string
		tokenSecret              string
		cfg                      *kubeadmapi.Discovery
		configMap                *fakeConfigMap
		delayedJWSSignaturePatch bool
		expectedError            bool
	}{
		{
			// This is the default behavior. The JWS signature is patched after the cluster-info ConfigMap is created
			name:        "valid: retrieve a valid kubeconfig with CA verification and delayed JWS signature",
			tokenID:     "123456",
			tokenSecret: "abcdef1234567890",
			cfg: &kubeadmapi.Discovery{
				BootstrapToken: &kubeadmapi.BootstrapTokenDiscovery{
					Token:        "123456.abcdef1234567890",
					CACertHashes: []string{caCertHash},
				},
			},
			configMap: &fakeConfigMap{
				name: bootstrapapi.ConfigMapClusterInfo,
				data: map[string]string{},
			},
			delayedJWSSignaturePatch: true,
		},
		{
			// Same as above expect this test creates the ConfigMap with the JWS signature
			name:        "valid: retrieve a valid kubeconfig with CA verification",
			tokenID:     "123456",
			tokenSecret: "abcdef1234567890",
			cfg: &kubeadmapi.Discovery{
				BootstrapToken: &kubeadmapi.BootstrapTokenDiscovery{
					Token:        "123456.abcdef1234567890",
					CACertHashes: []string{caCertHash},
				},
			},
			configMap: &fakeConfigMap{
				name: bootstrapapi.ConfigMapClusterInfo,
				data: nil,
			},
		},
		{
			// Skipping CA verification is also supported
			name:        "valid: retrieve a valid kubeconfig without CA verification",
			tokenID:     "123456",
			tokenSecret: "abcdef1234567890",
			cfg: &kubeadmapi.Discovery{
				BootstrapToken: &kubeadmapi.BootstrapTokenDiscovery{
					Token: "123456.abcdef1234567890",
				},
			},
			configMap: &fakeConfigMap{
				name: bootstrapapi.ConfigMapClusterInfo,
				data: nil,
			},
		},
		{
			name:        "invalid: token format is invalid",
			tokenID:     "foo",
			tokenSecret: "bar",
			cfg: &kubeadmapi.Discovery{
				BootstrapToken: &kubeadmapi.BootstrapTokenDiscovery{
					Token: "foo.bar",
				},
			},
			configMap: &fakeConfigMap{
				name: bootstrapapi.ConfigMapClusterInfo,
				data: nil,
			},
			expectedError: true,
		},
		{
			name:        "invalid: missing cluster-info ConfigMap",
			tokenID:     "123456",
			tokenSecret: "abcdef1234567890",
			cfg: &kubeadmapi.Discovery{
				BootstrapToken: &kubeadmapi.BootstrapTokenDiscovery{
					Token: "123456.abcdef1234567890",
				},
			},
			configMap: &fakeConfigMap{
				name: "baz",
				data: nil,
			},
			expectedError: true,
		},
		{
			name:        "invalid: wrong JWS signature",
			tokenID:     "123456",
			tokenSecret: "abcdef1234567890",
			cfg: &kubeadmapi.Discovery{
				BootstrapToken: &kubeadmapi.BootstrapTokenDiscovery{
					Token: "123456.abcdef1234567890",
				},
			},
			configMap: &fakeConfigMap{
				name: bootstrapapi.ConfigMapClusterInfo,
				data: map[string]string{
					bootstrapapi.KubeConfigKey:                    "foo",
					bootstrapapi.JWSSignatureKeyPrefix + "123456": "bar",
				},
			},
			expectedError: true,
		},
		{
			name:        "invalid: missing key for JWSSignatureKeyPrefix",
			tokenID:     "123456",
			tokenSecret: "abcdef1234567890",
			cfg: &kubeadmapi.Discovery{
				BootstrapToken: &kubeadmapi.BootstrapTokenDiscovery{
					Token: "123456.abcdef1234567890",
				},
			},
			configMap: &fakeConfigMap{
				name: bootstrapapi.ConfigMapClusterInfo,
				data: map[string]string{
					bootstrapapi.KubeConfigKey: "foo",
				},
			},
			expectedError: true,
		},
		{
			name:        "invalid: wrong CA cert hash",
			tokenID:     "123456",
			tokenSecret: "abcdef1234567890",
			cfg: &kubeadmapi.Discovery{
				BootstrapToken: &kubeadmapi.BootstrapTokenDiscovery{
					Token:        "123456.abcdef1234567890",
					CACertHashes: []string{"foo"},
				},
			},
			configMap: &fakeConfigMap{
				name: bootstrapapi.ConfigMapClusterInfo,
				data: nil,
			},
			expectedError: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			kubeconfig := buildSecureBootstrapKubeConfig("127.0.0.1", []byte(caCert), "somecluster")
			kubeconfigBytes, err := clientcmd.Write(*kubeconfig)
			if err != nil {
				t.Fatalf("cannot marshal kubeconfig %v", err)
			}

			// Generate signature of the insecure kubeconfig
			sig, err := tokenjws.ComputeDetachedSignature(string(kubeconfigBytes), test.tokenID, test.tokenSecret)
			if err != nil {
				t.Fatalf("cannot compute detached JWS signature: %v", err)
			}

			// If the JWS signature is delayed, only add the kubeconfig
			if test.delayedJWSSignaturePatch {
				test.configMap.data = map[string]string{}
				test.configMap.data[bootstrapapi.KubeConfigKey] = string(kubeconfigBytes)
			}

			// Populate the default cluster-info data
			if test.configMap.data == nil {
				test.configMap.data = map[string]string{}
				test.configMap.data[bootstrapapi.KubeConfigKey] = string(kubeconfigBytes)
				test.configMap.data[bootstrapapi.JWSSignatureKeyPrefix+test.tokenID] = sig
			}

			// Create a fake client and create the cluster-info ConfigMap
			client := fakeclient.NewSimpleClientset()
			if err = test.configMap.createOrUpdate(client); err != nil {
				t.Fatalf("could not create ConfigMap: %v", err)
			}

			// Set arbitrary discovery timeout and retry interval
			timeout := time.Millisecond * 500
			interval := time.Millisecond * 20

			// Patch the JWS signature after a short delay
			if test.delayedJWSSignaturePatch {
				test.configMap.data[bootstrapapi.JWSSignatureKeyPrefix+test.tokenID] = sig
				go func() {
					time.Sleep(time.Millisecond * 60)
					if err := test.configMap.createOrUpdate(client); err != nil {
						t.Errorf("could not update the cluster-info ConfigMap with a JWS signature: %v", err)
					}
				}()
			}

			// Retrieve validated configuration
			kubeconfig, err = retrieveValidatedConfigInfo(client, test.cfg, interval, timeout, false, true)
			if (err != nil) != test.expectedError {
				t.Errorf("expected error %v, got %v, error: %v", test.expectedError, err != nil, err)
			}

			// Return if an error is expected
			if err != nil {
				return
			}

			// Validate the resulted kubeconfig
			kubeconfigBytes, err = clientcmd.Write(*kubeconfig)
			if err != nil {
				t.Fatalf("cannot marshal resulted kubeconfig %v", err)
			}
			if string(kubeconfigBytes) != expectedKubeconfig {
				t.Error("unexpected kubeconfig")
				diff := difflib.UnifiedDiff{
					A:        difflib.SplitLines(expectedKubeconfig),
					B:        difflib.SplitLines(string(kubeconfigBytes)),
					FromFile: "expected",
					ToFile:   "got",
					Context:  10,
				}
				diffstr, err := difflib.GetUnifiedDiffString(diff)
				if err != nil {
					t.Fatalf("error generating unified diff string: %v", err)
				}
				t.Errorf("\n%s", diffstr)
			}
		})
	}
}

type fakeConfigMap struct {
	name string
	data map[string]string
}

func (c *fakeConfigMap) createOrUpdate(client clientset.Interface) error {
	return apiclient.CreateOrUpdate(client.CoreV1().ConfigMaps(metav1.NamespacePublic), &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      c.name,
			Namespace: metav1.NamespacePublic,
		},
		Data: c.data,
	})
}
