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
	"strings"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha2"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/uploadconfig"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	kubeproxyconfigv1alpha1 "k8s.io/kubernetes/pkg/proxy/apis/kubeproxyconfig/v1alpha1"
	utilpointer "k8s.io/kubernetes/pkg/util/pointer"
)

func TestFetchConfigFromFileOrCluster(t *testing.T) {
	var tests = []struct {
		name      string
		cfgPath   string
		testCfg   *kubeadm.MasterConfiguration
		expectCfg string
		expectErr string
	}{
		{
			name: "fetch valid config from configMap",
			testCfg: &kubeadm.MasterConfiguration{
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
				KubeProxy: kubeadm.KubeProxy{
					Config: &kubeproxyconfigv1alpha1.KubeProxyConfiguration{
						BindAddress:        "192.168.59.103",
						HealthzBindAddress: "0.0.0.0:10256",
						MetricsBindAddress: "127.0.0.1:10249",
						ClusterCIDR:        "192.168.59.0/24",
						UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
						ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
						IPTables: kubeproxyconfigv1alpha1.KubeProxyIPTablesConfiguration{
							MasqueradeAll: true,
							SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
						},
						IPVS: kubeproxyconfigv1alpha1.KubeProxyIPVSConfiguration{
							SyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
						},
						Conntrack: kubeproxyconfigv1alpha1.KubeProxyConntrackConfiguration{
							Max:        utilpointer.Int32Ptr(2),
							MaxPerCore: utilpointer.Int32Ptr(1),
							Min:        utilpointer.Int32Ptr(1),
							TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
							TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
						},
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
			testCfg: &kubeadm.MasterConfiguration{
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
			testCfg: &kubeadm.MasterConfiguration{
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
			expectCfg: "testdata/conversion/master/v1alpha2.yaml",
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
				err := uploadconfig.UploadConfiguration(tt.testCfg, client)
				if err != nil {
					t.Errorf("UploadConfiguration failed err: %v", err)
				}
			}
			resultCfg, err := FetchConfigFromFileOrCluster(client, os.Stdout, "upgrade/config", tt.cfgPath)
			if len(tt.expectErr) == 0 {
				if err != nil {
					t.Fatalf("expected no err, but got err: %v", err)
				}
				if len(tt.expectCfg) != 0 {
					actual, err := kubeadmutil.MarshalToYamlForCodecs(resultCfg, v1alpha2.SchemeGroupVersion, scheme.Codecs)
					if err != nil {
						t.Fatalf("couldn't marshal result object: %v", err)
					}
					expected, err := ioutil.ReadFile(tt.expectCfg)
					if err != nil {
						t.Fatalf("couldn't read test data: %v", err)
					}
					if !bytes.Equal(actual, expected) {
						t.Errorf("the expected and actual output differs. %s", diff(expected, actual))
					}
				}
			} else if !strings.Contains(err.Error(), tt.expectErr) {
				t.Errorf("expected contain err: %v, but got err: %v", tt.expectErr, err)
			}
		})
	}
}
