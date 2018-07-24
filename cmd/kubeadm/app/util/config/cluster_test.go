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
	"os"
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

func TestFetchConfigFromFileOrCluster(t *testing.T) {
	var tests = []struct {
		name      string
		cfgPath   string
		testCfg   *kubeadmapi.InitConfiguration
		expectErr string
	}{
		{
			name: "fetch valid config from configMap",
			testCfg: &kubeadmapi.InitConfiguration{
				KubernetesVersion: "v1.10.3",
				API: kubeadm.API{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         6443,
				},
				Etcd: kubeadm.Etcd{
					Local: &kubeadm.LocalEtcd{
						DataDir: "/some/path",
					},
				},
				Networking: kubeadm.Networking{
					ServiceSubnet: "10.96.0.1/12",
					DNSDomain:     "cluster.local",
					PodSubnet:     "10.0.1.15/16",
				},
				CertificatesDir: "/some/other/cert/dir",
				BootstrapTokens: []kubeadm.BootstrapToken{
					{
						Token: &kubeadm.BootstrapTokenString{
							ID:     "abcdef",
							Secret: "abcdef0123456789",
						},
					},
				},
				NodeRegistration: kubeadm.NodeRegistrationOptions{
					Name:      "node-foo",
					CRISocket: "/var/run/custom-cri.sock",
				},
			},
		},
		{
			name: "fetch invalid config from configMap",
			testCfg: &kubeadmapi.InitConfiguration{
				KubernetesVersion: "v1.10.3",
				API: kubeadm.API{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         6443,
				},
				Etcd: kubeadm.Etcd{
					Local: &kubeadm.LocalEtcd{
						DataDir: "/some/path",
					},
				},
				Networking: kubeadm.Networking{
					ServiceSubnet: "10.96.0.1/12",
					DNSDomain:     "cluster.local",
					PodSubnet:     "10.0.1.15",
				},
				CertificatesDir: "/some/other/cert/dir",
				BootstrapTokens: []kubeadm.BootstrapToken{
					{
						Token: &kubeadm.BootstrapTokenString{
							ID:     "abcdef",
							Secret: "abcdef0123456789",
						},
					},
				},
				NodeRegistration: kubeadm.NodeRegistrationOptions{
					Name:      "node-foo",
					CRISocket: "/var/run/custom-cri.sock",
				},
			},
			expectErr: "couldn't parse subnet",
		},
		{
			name:    "fetch valid config from cfgPath",
			cfgPath: "testdata/conversion/master/v1alpha2.yaml",
			testCfg: &kubeadmapi.InitConfiguration{
				KubernetesVersion: "v1.10.3",
				API: kubeadm.API{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         6443,
				},
				Etcd: kubeadm.Etcd{
					Local: &kubeadm.LocalEtcd{
						DataDir: "/some/path",
					},
				},
				Networking: kubeadm.Networking{
					ServiceSubnet: "10.96.0.1/12",
					DNSDomain:     "cluster.local",
					PodSubnet:     "10.0.1.15",
				},
				CertificatesDir: "/some/other/cert/dir",
				BootstrapTokens: []kubeadm.BootstrapToken{
					{
						Token: &kubeadm.BootstrapTokenString{
							ID:     "abcdef",
							Secret: "abcdef0123456789",
						},
					},
				},
				NodeRegistration: kubeadm.NodeRegistrationOptions{
					Name:      "node-foo",
					CRISocket: "/var/run/custom-cri.sock",
				},
			},
		},
		{
			name:      "fetch invalid config from cfgPath",
			cfgPath:   "testdata/validation/invalid_mastercfg.yaml",
			expectErr: "was not of the form",
		},
		{
			name:      "fetch config from not exist cfgPath",
			cfgPath:   "testdata231/defaulting/master/defaulted.yaml",
			expectErr: "no such file or directory",
		},
		{
			name:      "fetch config when no configMap and no cfgPath",
			expectErr: "not found",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset()
			if tt.testCfg != nil {
				err := createConfigMapWithCfg(tt.testCfg, client)
				if err != nil {
					t.Errorf("UploadConfiguration failed err: %v", err)
				}
			}
			_, err := FetchConfigFromFileOrCluster(client, os.Stdout, "upgrade/config", tt.cfgPath)
			if len(tt.expectErr) == 0 {
				if err != nil {
					t.Fatalf("expected no err, but got err: %v", err)
				}
			} else if !strings.Contains(err.Error(), tt.expectErr) {
				t.Errorf("expected contain err: %v, but got err: %v", tt.expectErr, err)
			}
		})
	}
}

// createConfigMapWithCfg create a ConfigMap with InitConfiguration for TestFetchConfigFromFileOrCluster
func createConfigMapWithCfg(cfgToCreate *kubeadmapi.InitConfiguration, client clientset.Interface) error {
	cfgYaml, err := MarshalKubeadmConfigObject(cfgToCreate)
	if err != nil {
		fmt.Println("err", err.Error())
		return err
	}

	err = apiclient.CreateOrUpdateConfigMap(client, &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.InitConfigurationConfigMap,
			Namespace: metav1.NamespaceSystem,
		},
		Data: map[string]string{
			kubeadmconstants.InitConfigurationConfigMapKey: string(cfgYaml),
		},
	})
	if err != nil {
		return err
	}
	return nil
}
