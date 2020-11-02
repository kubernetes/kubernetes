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

package etcd

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"

	"github.com/lithammer/dedent"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	etcdutil "k8s.io/kubernetes/cmd/kubeadm/app/util/etcd"
	staticpodutil "k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
)

func TestGetEtcdPodSpec(t *testing.T) {
	// Creates a ClusterConfiguration
	cfg := &kubeadmapi.ClusterConfiguration{
		KubernetesVersion: "v1.7.0",
		Etcd: kubeadmapi.Etcd{
			Local: &kubeadmapi.LocalEtcd{
				DataDir: "/var/lib/etcd",
			},
		},
	}
	endpoint := &kubeadmapi.APIEndpoint{}

	// Executes GetEtcdPodSpec
	spec := GetEtcdPodSpec(cfg, endpoint, "", []etcdutil.Member{})

	// Assert each specs refers to the right pod
	if spec.Spec.Containers[0].Name != kubeadmconstants.Etcd {
		t.Errorf("getKubeConfigSpecs spec for etcd contains pod %s, expects %s", spec.Spec.Containers[0].Name, kubeadmconstants.Etcd)
	}
}

func TestCreateLocalEtcdStaticPodManifestFile(t *testing.T) {
	// Create temp folder for the test case
	tmpdir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpdir)

	var tests = []struct {
		cfg           *kubeadmapi.ClusterConfiguration
		expectedError bool
	}{
		{
			cfg: &kubeadmapi.ClusterConfiguration{
				KubernetesVersion: "v1.7.0",
				Etcd: kubeadmapi.Etcd{
					Local: &kubeadmapi.LocalEtcd{
						DataDir: tmpdir + "/etcd",
					},
				},
			},
			expectedError: false,
		},
		{
			cfg: &kubeadmapi.ClusterConfiguration{
				KubernetesVersion: "v1.7.0",
				Etcd: kubeadmapi.Etcd{
					External: &kubeadmapi.ExternalEtcd{
						Endpoints: []string{
							"https://etcd-instance:2379",
						},
						CAFile:   "/etc/kubernetes/pki/etcd/ca.crt",
						CertFile: "/etc/kubernetes/pki/etcd/apiserver-etcd-client.crt",
						KeyFile:  "/etc/kubernetes/pki/etcd/apiserver-etcd-client.key",
					},
				},
			},
			expectedError: true,
		},
	}

	for _, test := range tests {
		// Execute createStaticPodFunction
		manifestPath := filepath.Join(tmpdir, kubeadmconstants.ManifestsSubDirName)
		err := CreateLocalEtcdStaticPodManifestFile(manifestPath, "", "", test.cfg, &kubeadmapi.APIEndpoint{})

		if !test.expectedError {
			if err != nil {
				t.Errorf("CreateLocalEtcdStaticPodManifestFile failed when not expected: %v", err)
			}
			// Assert expected files are there
			testutil.AssertFilesCount(t, manifestPath, 1)
			testutil.AssertFileExists(t, manifestPath, kubeadmconstants.Etcd+".yaml")
		} else {
			testutil.AssertError(t, err, "etcd static pod manifest cannot be generated for cluster using external etcd")
		}
	}
}

func TestCreateLocalEtcdStaticPodManifestFileWithPatches(t *testing.T) {
	// Create temp folder for the test case
	tmpdir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpdir)

	// Creates a Cluster Configuration
	cfg := &kubeadmapi.ClusterConfiguration{
		KubernetesVersion: "v1.7.0",
		Etcd: kubeadmapi.Etcd{
			Local: &kubeadmapi.LocalEtcd{
				DataDir: tmpdir + "/etcd",
			},
		},
	}

	patchesPath := filepath.Join(tmpdir, "patch-files")
	err := os.MkdirAll(patchesPath, 0777)
	if err != nil {
		t.Fatalf("Couldn't create %s", patchesPath)
	}

	patchString := dedent.Dedent(`
	metadata:
	  annotations:
	    patched: "true"
	`)

	err = ioutil.WriteFile(filepath.Join(patchesPath, kubeadmconstants.Etcd+".yaml"), []byte(patchString), 0644)
	if err != nil {
		t.Fatalf("WriteFile returned unexpected error: %v", err)
	}

	manifestPath := filepath.Join(tmpdir, kubeadmconstants.ManifestsSubDirName)
	err = CreateLocalEtcdStaticPodManifestFile(manifestPath, patchesPath, "", cfg, &kubeadmapi.APIEndpoint{})
	if err != nil {
		t.Errorf("Error executing createStaticPodFunction: %v", err)
		return
	}

	pod, err := staticpodutil.ReadStaticPodFromDisk(filepath.Join(manifestPath, kubeadmconstants.Etcd+".yaml"))
	if err != nil {
		t.Errorf("Error executing ReadStaticPodFromDisk: %v", err)
		return
	}

	if _, ok := pod.ObjectMeta.Annotations["patched"]; !ok {
		t.Errorf("Patches were not applied to %s", kubeadmconstants.Etcd)
	}
}

func TestGetEtcdCommand(t *testing.T) {
	var tests = []struct {
		name             string
		advertiseAddress string
		nodeName         string
		extraArgs        map[string]string
		initialCluster   []etcdutil.Member
		expected         []string
	}{
		{
			name:             "Default args - with empty etcd initial cluster",
			advertiseAddress: "1.2.3.4",
			nodeName:         "foo",
			expected: []string{
				"etcd",
				"--name=foo",
				fmt.Sprintf("--listen-client-urls=https://127.0.0.1:%d,https://1.2.3.4:%d", kubeadmconstants.EtcdListenClientPort, kubeadmconstants.EtcdListenClientPort),
				fmt.Sprintf("--listen-metrics-urls=http://127.0.0.1:%d", kubeadmconstants.EtcdMetricsPort),
				fmt.Sprintf("--advertise-client-urls=https://1.2.3.4:%d", kubeadmconstants.EtcdListenClientPort),
				fmt.Sprintf("--listen-peer-urls=https://1.2.3.4:%d", kubeadmconstants.EtcdListenPeerPort),
				fmt.Sprintf("--initial-advertise-peer-urls=https://1.2.3.4:%d", kubeadmconstants.EtcdListenPeerPort),
				"--data-dir=/var/lib/etcd",
				"--cert-file=" + kubeadmconstants.EtcdServerCertName,
				"--key-file=" + kubeadmconstants.EtcdServerKeyName,
				"--trusted-ca-file=" + kubeadmconstants.EtcdCACertName,
				"--client-cert-auth=true",
				"--peer-cert-file=" + kubeadmconstants.EtcdPeerCertName,
				"--peer-key-file=" + kubeadmconstants.EtcdPeerKeyName,
				"--peer-trusted-ca-file=" + kubeadmconstants.EtcdCACertName,
				"--snapshot-count=10000",
				"--peer-client-cert-auth=true",
				fmt.Sprintf("--initial-cluster=foo=https://1.2.3.4:%d", kubeadmconstants.EtcdListenPeerPort),
			},
		},
		{
			name:             "Default args - With an existing etcd cluster",
			advertiseAddress: "1.2.3.4",
			nodeName:         "foo",
			initialCluster: []etcdutil.Member{
				{Name: "foo", PeerURL: fmt.Sprintf("https://1.2.3.4:%d", kubeadmconstants.EtcdListenPeerPort)}, // NB. the joining etcd instance should be part of the initialCluster list
				{Name: "bar", PeerURL: fmt.Sprintf("https://5.6.7.8:%d", kubeadmconstants.EtcdListenPeerPort)},
			},
			expected: []string{
				"etcd",
				"--name=foo",
				fmt.Sprintf("--listen-client-urls=https://127.0.0.1:%d,https://1.2.3.4:%d", kubeadmconstants.EtcdListenClientPort, kubeadmconstants.EtcdListenClientPort),
				fmt.Sprintf("--listen-metrics-urls=http://127.0.0.1:%d", kubeadmconstants.EtcdMetricsPort),
				fmt.Sprintf("--advertise-client-urls=https://1.2.3.4:%d", kubeadmconstants.EtcdListenClientPort),
				fmt.Sprintf("--listen-peer-urls=https://1.2.3.4:%d", kubeadmconstants.EtcdListenPeerPort),
				fmt.Sprintf("--initial-advertise-peer-urls=https://1.2.3.4:%d", kubeadmconstants.EtcdListenPeerPort),
				"--data-dir=/var/lib/etcd",
				"--cert-file=" + kubeadmconstants.EtcdServerCertName,
				"--key-file=" + kubeadmconstants.EtcdServerKeyName,
				"--trusted-ca-file=" + kubeadmconstants.EtcdCACertName,
				"--client-cert-auth=true",
				"--peer-cert-file=" + kubeadmconstants.EtcdPeerCertName,
				"--peer-key-file=" + kubeadmconstants.EtcdPeerKeyName,
				"--peer-trusted-ca-file=" + kubeadmconstants.EtcdCACertName,
				"--snapshot-count=10000",
				"--peer-client-cert-auth=true",
				"--initial-cluster-state=existing",
				fmt.Sprintf("--initial-cluster=foo=https://1.2.3.4:%d,bar=https://5.6.7.8:%d", kubeadmconstants.EtcdListenPeerPort, kubeadmconstants.EtcdListenPeerPort),
			},
		},
		{
			name:             "Extra args",
			advertiseAddress: "1.2.3.4",
			nodeName:         "bar",
			extraArgs: map[string]string{
				"listen-client-urls":    "https://10.0.1.10:2379",
				"advertise-client-urls": "https://10.0.1.10:2379",
			},
			expected: []string{
				"etcd",
				"--name=bar",
				"--listen-client-urls=https://10.0.1.10:2379",
				fmt.Sprintf("--listen-metrics-urls=http://127.0.0.1:%d", kubeadmconstants.EtcdMetricsPort),
				"--advertise-client-urls=https://10.0.1.10:2379",
				fmt.Sprintf("--listen-peer-urls=https://1.2.3.4:%d", kubeadmconstants.EtcdListenPeerPort),
				fmt.Sprintf("--initial-advertise-peer-urls=https://1.2.3.4:%d", kubeadmconstants.EtcdListenPeerPort),
				"--data-dir=/var/lib/etcd",
				"--cert-file=" + kubeadmconstants.EtcdServerCertName,
				"--key-file=" + kubeadmconstants.EtcdServerKeyName,
				"--trusted-ca-file=" + kubeadmconstants.EtcdCACertName,
				"--client-cert-auth=true",
				"--peer-cert-file=" + kubeadmconstants.EtcdPeerCertName,
				"--peer-key-file=" + kubeadmconstants.EtcdPeerKeyName,
				"--peer-trusted-ca-file=" + kubeadmconstants.EtcdCACertName,
				"--snapshot-count=10000",
				"--peer-client-cert-auth=true",
				fmt.Sprintf("--initial-cluster=bar=https://1.2.3.4:%d", kubeadmconstants.EtcdListenPeerPort),
			},
		},
		{
			name:             "IPv6 advertise address",
			advertiseAddress: "2001:db8::3",
			nodeName:         "foo",
			expected: []string{
				"etcd",
				"--name=foo",
				fmt.Sprintf("--listen-client-urls=https://[::1]:%d,https://[2001:db8::3]:%d", kubeadmconstants.EtcdListenClientPort, kubeadmconstants.EtcdListenClientPort),
				fmt.Sprintf("--listen-metrics-urls=http://[::1]:%d", kubeadmconstants.EtcdMetricsPort),
				fmt.Sprintf("--advertise-client-urls=https://[2001:db8::3]:%d", kubeadmconstants.EtcdListenClientPort),
				fmt.Sprintf("--listen-peer-urls=https://[2001:db8::3]:%d", kubeadmconstants.EtcdListenPeerPort),
				fmt.Sprintf("--initial-advertise-peer-urls=https://[2001:db8::3]:%d", kubeadmconstants.EtcdListenPeerPort),
				"--data-dir=/var/lib/etcd",
				"--cert-file=" + kubeadmconstants.EtcdServerCertName,
				"--key-file=" + kubeadmconstants.EtcdServerKeyName,
				"--trusted-ca-file=" + kubeadmconstants.EtcdCACertName,
				"--client-cert-auth=true",
				"--peer-cert-file=" + kubeadmconstants.EtcdPeerCertName,
				"--peer-key-file=" + kubeadmconstants.EtcdPeerKeyName,
				"--peer-trusted-ca-file=" + kubeadmconstants.EtcdCACertName,
				"--snapshot-count=10000",
				"--peer-client-cert-auth=true",
				fmt.Sprintf("--initial-cluster=foo=https://[2001:db8::3]:%d", kubeadmconstants.EtcdListenPeerPort),
			},
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			endpoint := &kubeadmapi.APIEndpoint{
				AdvertiseAddress: rt.advertiseAddress,
			}
			cfg := &kubeadmapi.ClusterConfiguration{
				Etcd: kubeadmapi.Etcd{
					Local: &kubeadmapi.LocalEtcd{
						DataDir:   "/var/lib/etcd",
						ExtraArgs: rt.extraArgs,
					},
				},
			}
			actual := getEtcdCommand(cfg, endpoint, rt.nodeName, rt.initialCluster)
			sort.Strings(actual)
			sort.Strings(rt.expected)
			if !reflect.DeepEqual(actual, rt.expected) {
				t.Errorf("failed getEtcdCommand:\nexpected:\n%v\nsaw:\n%v", rt.expected, actual)
			}
		})
	}
}
