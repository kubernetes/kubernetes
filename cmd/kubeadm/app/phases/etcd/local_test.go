//go:build !windows
// +build !windows

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
	"os"
	"path"
	"path/filepath"
	"reflect"
	"sort"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/lithammer/dedent"

	v1 "k8s.io/api/core/v1"
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
				ExtraEnvs: []kubeadmapi.EnvVar{
					{
						EnvVar: v1.EnvVar{Name: "Foo", Value: "Bar"},
					},
				},
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
	env := []v1.EnvVar{{Name: "Foo", Value: "Bar"}}
	if !reflect.DeepEqual(spec.Spec.Containers[0].Env, env) {
		t.Errorf("expected env: %v, got: %v", env, spec.Spec.Containers[0].Env)
	}
}

func TestCreateLocalEtcdStaticPodManifestFile(t *testing.T) {
	// Create temp folder for the test case
	tmpdir := t.TempDir()

	var tests = []struct {
		cfg              *kubeadmapi.ClusterConfiguration
		expectedError    bool
		expectedManifest string
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
			expectedManifest: fmt.Sprintf(`apiVersion: v1
kind: Pod
metadata:
  annotations:
    kubeadm.kubernetes.io/etcd.advertise-client-urls: https://:2379
  labels:
    component: etcd
    tier: control-plane
  name: etcd
  namespace: kube-system
spec:
  containers:
  - command:
    - etcd
    - --advertise-client-urls=https://:2379
    - --cert-file=etcd/server.crt
    - --client-cert-auth=true
    - --data-dir=%s/etcd
    - --experimental-initial-corrupt-check=true
    - --experimental-watch-progress-notify-interval=5s
    - --initial-advertise-peer-urls=https://:2380
    - --initial-cluster==https://:2380
    - --key-file=etcd/server.key
    - --listen-client-urls=https://127.0.0.1:2379,https://:2379
    - --listen-metrics-urls=http://127.0.0.1:2381
    - --listen-peer-urls=https://:2380
    - --name=
    - --peer-cert-file=etcd/peer.crt
    - --peer-client-cert-auth=true
    - --peer-key-file=etcd/peer.key
    - --peer-trusted-ca-file=etcd/ca.crt
    - --snapshot-count=10000
    - --trusted-ca-file=etcd/ca.crt
    image: /etcd:%s
    imagePullPolicy: IfNotPresent
    livenessProbe:
      failureThreshold: 8
      httpGet:
        host: 127.0.0.1
        path: /livez
        port: 2381
        scheme: HTTP
      initialDelaySeconds: 10
      periodSeconds: 10
      timeoutSeconds: 15
    name: etcd
    readinessProbe:
      failureThreshold: 3
      httpGet:
        host: 127.0.0.1
        path: /readyz
        port: 2381
        scheme: HTTP
      periodSeconds: 1
      timeoutSeconds: 15
    resources:
      requests:
        cpu: 100m
        memory: 100Mi
    startupProbe:
      failureThreshold: 24
      httpGet:
        host: 127.0.0.1
        path: /readyz
        port: 2381
        scheme: HTTP
      initialDelaySeconds: 10
      periodSeconds: 10
      timeoutSeconds: 15
    volumeMounts:
    - mountPath: %s/etcd
      name: etcd-data
    - mountPath: /etcd
      name: etcd-certs
  hostNetwork: true
  priority: 2000001000
  priorityClassName: system-node-critical
  securityContext:
    seccompProfile:
      type: RuntimeDefault
  volumes:
  - hostPath:
      path: /etcd
      type: DirectoryOrCreate
    name: etcd-certs
  - hostPath:
      path: %s/etcd
      type: DirectoryOrCreate
    name: etcd-data
status: {}
`, tmpdir, kubeadmconstants.DefaultEtcdVersion, tmpdir, tmpdir),
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
		err := CreateLocalEtcdStaticPodManifestFile(manifestPath, "", "", test.cfg, &kubeadmapi.APIEndpoint{}, false /* IsDryRun */)

		if !test.expectedError {
			if err != nil {
				t.Errorf("CreateLocalEtcdStaticPodManifestFile failed when not expected: %v", err)
			}
			// Assert expected files are there
			testutil.AssertFilesCount(t, manifestPath, 1)
			testutil.AssertFileExists(t, manifestPath, kubeadmconstants.Etcd+".yaml")
			manifestBytes, err := os.ReadFile(path.Join(manifestPath, kubeadmconstants.Etcd+".yaml"))
			if err != nil {
				t.Errorf("failed to load generated manifest file: %v", err)
			}
			if test.expectedManifest != string(manifestBytes) {
				t.Errorf(
					"File created by CreateLocalEtcdStaticPodManifestFile is not as expected. Diff: \n%s",
					cmp.Diff(string(manifestBytes), test.expectedManifest),
				)
			}
		} else {
			testutil.AssertError(t, err, "etcd static pod manifest cannot be generated for cluster using external etcd")
		}
	}
}

func TestCreateLocalEtcdStaticPodManifestFileWithPatches(t *testing.T) {
	// Create temp folder for the test case
	tmpdir := t.TempDir()

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

	err = os.WriteFile(filepath.Join(patchesPath, kubeadmconstants.Etcd+".yaml"), []byte(patchString), 0644)
	if err != nil {
		t.Fatalf("WriteFile returned unexpected error: %v", err)
	}

	manifestPath := filepath.Join(tmpdir, kubeadmconstants.ManifestsSubDirName)
	err = CreateLocalEtcdStaticPodManifestFile(manifestPath, patchesPath, "", cfg, &kubeadmapi.APIEndpoint{}, false /* IsDryRun */)
	if err != nil {
		t.Errorf("Error executing createStaticPodFunction: %v", err)
		return
	}

	pod, err := staticpodutil.ReadStaticPodFromDisk(filepath.Join(manifestPath, kubeadmconstants.Etcd+".yaml"))
	if err != nil {
		t.Errorf("Error executing ReadStaticPodFromDisk: %v", err)
		return
	}
	if pod.Spec.DNSPolicy != "" {
		t.Errorf("DNSPolicy should be empty but: %v", pod.Spec.DNSPolicy)
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
		extraArgs        []kubeadmapi.Arg
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
				"--experimental-initial-corrupt-check=true",
				"--experimental-watch-progress-notify-interval=5s",
				fmt.Sprintf("--listen-client-urls=https://127.0.0.1:%d,https://1.2.3.4:%d", kubeadmconstants.EtcdListenClientPort, kubeadmconstants.EtcdListenClientPort),
				fmt.Sprintf("--listen-metrics-urls=http://127.0.0.1:%d", kubeadmconstants.EtcdMetricsPort),
				fmt.Sprintf("--advertise-client-urls=https://1.2.3.4:%d", kubeadmconstants.EtcdListenClientPort),
				fmt.Sprintf("--listen-peer-urls=https://1.2.3.4:%d", kubeadmconstants.EtcdListenPeerPort),
				fmt.Sprintf("--initial-advertise-peer-urls=https://1.2.3.4:%d", kubeadmconstants.EtcdListenPeerPort),
				"--data-dir=/var/lib/etcd",
				"--cert-file=" + filepath.FromSlash(kubeadmconstants.EtcdServerCertName),
				"--key-file=" + filepath.FromSlash(kubeadmconstants.EtcdServerKeyName),
				"--trusted-ca-file=" + filepath.FromSlash(kubeadmconstants.EtcdCACertName),
				"--client-cert-auth=true",
				"--peer-cert-file=" + filepath.FromSlash(kubeadmconstants.EtcdPeerCertName),
				"--peer-key-file=" + filepath.FromSlash(kubeadmconstants.EtcdPeerKeyName),
				"--peer-trusted-ca-file=" + filepath.FromSlash(kubeadmconstants.EtcdCACertName),
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
				"--experimental-initial-corrupt-check=true",
				"--experimental-watch-progress-notify-interval=5s",
				fmt.Sprintf("--listen-client-urls=https://127.0.0.1:%d,https://1.2.3.4:%d", kubeadmconstants.EtcdListenClientPort, kubeadmconstants.EtcdListenClientPort),
				fmt.Sprintf("--listen-metrics-urls=http://127.0.0.1:%d", kubeadmconstants.EtcdMetricsPort),
				fmt.Sprintf("--advertise-client-urls=https://1.2.3.4:%d", kubeadmconstants.EtcdListenClientPort),
				fmt.Sprintf("--listen-peer-urls=https://1.2.3.4:%d", kubeadmconstants.EtcdListenPeerPort),
				fmt.Sprintf("--initial-advertise-peer-urls=https://1.2.3.4:%d", kubeadmconstants.EtcdListenPeerPort),
				"--data-dir=/var/lib/etcd",
				"--cert-file=" + filepath.FromSlash(kubeadmconstants.EtcdServerCertName),
				"--key-file=" + filepath.FromSlash(kubeadmconstants.EtcdServerKeyName),
				"--trusted-ca-file=" + filepath.FromSlash(kubeadmconstants.EtcdCACertName),
				"--client-cert-auth=true",
				"--peer-cert-file=" + filepath.FromSlash(kubeadmconstants.EtcdPeerCertName),
				"--peer-key-file=" + filepath.FromSlash(kubeadmconstants.EtcdPeerKeyName),
				"--peer-trusted-ca-file=" + filepath.FromSlash(kubeadmconstants.EtcdCACertName),
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
			extraArgs: []kubeadmapi.Arg{
				{Name: "listen-client-urls", Value: "https://10.0.1.10:2379"},
				{Name: "advertise-client-urls", Value: "https://10.0.1.10:2379"},
			},
			expected: []string{
				"etcd",
				"--name=bar",
				"--experimental-initial-corrupt-check=true",
				"--experimental-watch-progress-notify-interval=5s",
				"--listen-client-urls=https://10.0.1.10:2379",
				fmt.Sprintf("--listen-metrics-urls=http://127.0.0.1:%d", kubeadmconstants.EtcdMetricsPort),
				"--advertise-client-urls=https://10.0.1.10:2379",
				fmt.Sprintf("--listen-peer-urls=https://1.2.3.4:%d", kubeadmconstants.EtcdListenPeerPort),
				fmt.Sprintf("--initial-advertise-peer-urls=https://1.2.3.4:%d", kubeadmconstants.EtcdListenPeerPort),
				"--data-dir=/var/lib/etcd",
				"--cert-file=" + filepath.FromSlash(kubeadmconstants.EtcdServerCertName),
				"--key-file=" + filepath.FromSlash(kubeadmconstants.EtcdServerKeyName),
				"--trusted-ca-file=" + filepath.FromSlash(kubeadmconstants.EtcdCACertName),
				"--client-cert-auth=true",
				"--peer-cert-file=" + filepath.FromSlash(kubeadmconstants.EtcdPeerCertName),
				"--peer-key-file=" + filepath.FromSlash(kubeadmconstants.EtcdPeerKeyName),
				"--peer-trusted-ca-file=" + filepath.FromSlash(kubeadmconstants.EtcdCACertName),
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
				"--experimental-initial-corrupt-check=true",
				"--experimental-watch-progress-notify-interval=5s",
				fmt.Sprintf("--listen-client-urls=https://[::1]:%d,https://[2001:db8::3]:%d", kubeadmconstants.EtcdListenClientPort, kubeadmconstants.EtcdListenClientPort),
				fmt.Sprintf("--listen-metrics-urls=http://[::1]:%d", kubeadmconstants.EtcdMetricsPort),
				fmt.Sprintf("--advertise-client-urls=https://[2001:db8::3]:%d", kubeadmconstants.EtcdListenClientPort),
				fmt.Sprintf("--listen-peer-urls=https://[2001:db8::3]:%d", kubeadmconstants.EtcdListenPeerPort),
				fmt.Sprintf("--initial-advertise-peer-urls=https://[2001:db8::3]:%d", kubeadmconstants.EtcdListenPeerPort),
				"--data-dir=/var/lib/etcd",
				"--cert-file=" + filepath.FromSlash(kubeadmconstants.EtcdServerCertName),
				"--key-file=" + filepath.FromSlash(kubeadmconstants.EtcdServerKeyName),
				"--trusted-ca-file=" + filepath.FromSlash(kubeadmconstants.EtcdCACertName),
				"--client-cert-auth=true",
				"--peer-cert-file=" + filepath.FromSlash(kubeadmconstants.EtcdPeerCertName),
				"--peer-key-file=" + filepath.FromSlash(kubeadmconstants.EtcdPeerKeyName),
				"--peer-trusted-ca-file=" + filepath.FromSlash(kubeadmconstants.EtcdCACertName),
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
