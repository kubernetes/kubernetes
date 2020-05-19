/*
Copyright 2016 The Kubernetes Authors.

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

package controlplane

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/lithammer/dedent"

	"k8s.io/apimachinery/pkg/util/sets"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	staticpodutil "k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
)

const (
	testCertsDir = "/var/lib/certs"
)

var cpVersion = kubeadmconstants.MinimumControlPlaneVersion.WithPreRelease("beta.2").String()

func TestGetStaticPodSpecs(t *testing.T) {

	// Creates a Cluster Configuration
	cfg := &kubeadmapi.ClusterConfiguration{
		KubernetesVersion: "v1.9.0",
	}

	// Executes GetStaticPodSpecs
	specs := GetStaticPodSpecs(cfg, &kubeadmapi.APIEndpoint{})

	var tests = []struct {
		name          string
		staticPodName string
	}{
		{
			name:          "KubeAPIServer",
			staticPodName: kubeadmconstants.KubeAPIServer,
		},
		{
			name:          "KubeControllerManager",
			staticPodName: kubeadmconstants.KubeControllerManager,
		},
		{
			name:          "KubeScheduler",
			staticPodName: kubeadmconstants.KubeScheduler,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// assert the spec for the staticPodName exists
			if spec, ok := specs[tc.staticPodName]; ok {

				// Assert each specs refers to the right pod
				if spec.Spec.Containers[0].Name != tc.staticPodName {
					t.Errorf("getKubeConfigSpecs spec for %s contains pod %s, expects %s", tc.staticPodName, spec.Spec.Containers[0].Name, tc.staticPodName)
				}

			} else {
				t.Errorf("getStaticPodSpecs didn't create spec for %s ", tc.staticPodName)
			}
		})
	}
}

func TestCreateStaticPodFilesAndWrappers(t *testing.T) {

	var tests = []struct {
		name       string
		components []string
	}{
		{
			name:       "KubeAPIServer KubeAPIServer KubeScheduler",
			components: []string{kubeadmconstants.KubeAPIServer, kubeadmconstants.KubeControllerManager, kubeadmconstants.KubeScheduler},
		},
		{
			name:       "KubeAPIServer",
			components: []string{kubeadmconstants.KubeAPIServer},
		},
		{
			name:       "KubeControllerManager",
			components: []string{kubeadmconstants.KubeControllerManager},
		},
		{
			name:       "KubeScheduler",
			components: []string{kubeadmconstants.KubeScheduler},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Create temp folder for the test case
			tmpdir := testutil.SetupTempDir(t)
			defer os.RemoveAll(tmpdir)

			// Creates a Cluster Configuration
			cfg := &kubeadmapi.ClusterConfiguration{
				KubernetesVersion: "v1.9.0",
			}

			// Execute createStaticPodFunction
			manifestPath := filepath.Join(tmpdir, kubeadmconstants.ManifestsSubDirName)
			err := CreateStaticPodFiles(manifestPath, "", cfg, &kubeadmapi.APIEndpoint{}, test.components...)
			if err != nil {
				t.Errorf("Error executing createStaticPodFunction: %v", err)
				return
			}

			// Assert expected files are there
			testutil.AssertFilesCount(t, manifestPath, len(test.components))

			for _, fileName := range test.components {
				testutil.AssertFileExists(t, manifestPath, fileName+".yaml")
			}
		})
	}
}

func TestCreateStaticPodFilesKustomize(t *testing.T) {
	// Create temp folder for the test case
	tmpdir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpdir)

	// Creates a Cluster Configuration
	cfg := &kubeadmapi.ClusterConfiguration{
		KubernetesVersion: "v1.9.0",
	}

	kustomizePath := filepath.Join(tmpdir, "kustomize")
	err := os.MkdirAll(kustomizePath, 0777)
	if err != nil {
		t.Fatalf("Couldn't create %s", kustomizePath)
	}

	patchString := dedent.Dedent(`
    apiVersion: v1
    kind: Pod
    metadata:
        name: kube-apiserver
        namespace: kube-system
        annotations:
            kustomize: patch for kube-apiserver
    `)

	err = ioutil.WriteFile(filepath.Join(kustomizePath, "patch.yaml"), []byte(patchString), 0644)
	if err != nil {
		t.Fatalf("WriteFile returned unexpected error: %v", err)
	}

	// Execute createStaticPodFunction with kustomizations
	manifestPath := filepath.Join(tmpdir, kubeadmconstants.ManifestsSubDirName)
	err = CreateStaticPodFiles(manifestPath, kustomizePath, cfg, &kubeadmapi.APIEndpoint{}, kubeadmconstants.KubeAPIServer)
	if err != nil {
		t.Errorf("Error executing createStaticPodFunction: %v", err)
		return
	}

	pod, err := staticpodutil.ReadStaticPodFromDisk(filepath.Join(manifestPath, fmt.Sprintf("%s.yaml", kubeadmconstants.KubeAPIServer)))
	if err != nil {
		t.Errorf("Error executing ReadStaticPodFromDisk: %v", err)
		return
	}

	if _, ok := pod.ObjectMeta.Annotations["kustomize"]; !ok {
		t.Error("Kustomize did not apply patches corresponding to the resource")
	}
}

func TestGetAPIServerCommand(t *testing.T) {
	var tests = []struct {
		name     string
		cfg      *kubeadmapi.ClusterConfiguration
		endpoint *kubeadmapi.APIEndpoint
		expected []string
	}{
		{
			name: "testing defaults",
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking:      kubeadmapi.Networking{ServiceSubnet: "bar"},
				CertificatesDir: testCertsDir,
			},
			endpoint: &kubeadmapi.APIEndpoint{BindPort: 123, AdvertiseAddress: "1.2.3.4"},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--enable-admission-plugins=NodeRestriction",
				"--service-cluster-ip-range=bar",
				"--service-account-key-file=" + testCertsDir + "/sa.pub",
				"--client-ca-file=" + testCertsDir + "/ca.crt",
				"--tls-cert-file=" + testCertsDir + "/apiserver.crt",
				"--tls-private-key-file=" + testCertsDir + "/apiserver.key",
				"--kubelet-client-certificate=" + testCertsDir + "/apiserver-kubelet-client.crt",
				"--kubelet-client-key=" + testCertsDir + "/apiserver-kubelet-client.key",
				"--enable-bootstrap-token-auth=true",
				"--secure-port=123",
				"--allow-privileged=true",
				"--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname",
				"--proxy-client-cert-file=/var/lib/certs/front-proxy-client.crt",
				"--proxy-client-key-file=/var/lib/certs/front-proxy-client.key",
				"--requestheader-username-headers=X-Remote-User",
				"--requestheader-group-headers=X-Remote-Group",
				"--requestheader-extra-headers-prefix=X-Remote-Extra-",
				"--requestheader-client-ca-file=" + testCertsDir + "/front-proxy-ca.crt",
				"--requestheader-allowed-names=front-proxy-client",
				"--authorization-mode=Node,RBAC",
				"--advertise-address=1.2.3.4",
				fmt.Sprintf("--etcd-servers=https://127.0.0.1:%d", kubeadmconstants.EtcdListenClientPort),
				"--etcd-cafile=" + testCertsDir + "/etcd/ca.crt",
				"--etcd-certfile=" + testCertsDir + "/apiserver-etcd-client.crt",
				"--etcd-keyfile=" + testCertsDir + "/apiserver-etcd-client.key",
			},
		},
		{
			name: "ipv6 advertise address",
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking:      kubeadmapi.Networking{ServiceSubnet: "bar"},
				CertificatesDir: testCertsDir,
			},
			endpoint: &kubeadmapi.APIEndpoint{BindPort: 123, AdvertiseAddress: "2001:db8::1"},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--enable-admission-plugins=NodeRestriction",
				"--service-cluster-ip-range=bar",
				"--service-account-key-file=" + testCertsDir + "/sa.pub",
				"--client-ca-file=" + testCertsDir + "/ca.crt",
				"--tls-cert-file=" + testCertsDir + "/apiserver.crt",
				"--tls-private-key-file=" + testCertsDir + "/apiserver.key",
				"--kubelet-client-certificate=" + testCertsDir + "/apiserver-kubelet-client.crt",
				"--kubelet-client-key=" + testCertsDir + "/apiserver-kubelet-client.key",
				"--enable-bootstrap-token-auth=true",
				fmt.Sprintf("--secure-port=%d", 123),
				"--allow-privileged=true",
				"--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname",
				"--proxy-client-cert-file=/var/lib/certs/front-proxy-client.crt",
				"--proxy-client-key-file=/var/lib/certs/front-proxy-client.key",
				"--requestheader-username-headers=X-Remote-User",
				"--requestheader-group-headers=X-Remote-Group",
				"--requestheader-extra-headers-prefix=X-Remote-Extra-",
				"--requestheader-client-ca-file=" + testCertsDir + "/front-proxy-ca.crt",
				"--requestheader-allowed-names=front-proxy-client",
				"--authorization-mode=Node,RBAC",
				"--advertise-address=2001:db8::1",
				fmt.Sprintf("--etcd-servers=https://[::1]:%d", kubeadmconstants.EtcdListenClientPort),
				"--etcd-cafile=" + testCertsDir + "/etcd/ca.crt",
				"--etcd-certfile=" + testCertsDir + "/apiserver-etcd-client.crt",
				"--etcd-keyfile=" + testCertsDir + "/apiserver-etcd-client.key",
			},
		},
		{
			name: "an external etcd with custom ca, certs and keys",
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking: kubeadmapi.Networking{ServiceSubnet: "bar"},
				Etcd: kubeadmapi.Etcd{
					External: &kubeadmapi.ExternalEtcd{
						Endpoints: []string{"https://[2001:abcd:bcda::1]:2379", "https://[2001:abcd:bcda::2]:2379"},
						CAFile:    "fuz",
						CertFile:  "fiz",
						KeyFile:   "faz",
					},
				},
				CertificatesDir: testCertsDir,
			},
			endpoint: &kubeadmapi.APIEndpoint{BindPort: 123, AdvertiseAddress: "2001:db8::1"},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--enable-admission-plugins=NodeRestriction",
				"--service-cluster-ip-range=bar",
				"--service-account-key-file=" + testCertsDir + "/sa.pub",
				"--client-ca-file=" + testCertsDir + "/ca.crt",
				"--tls-cert-file=" + testCertsDir + "/apiserver.crt",
				"--tls-private-key-file=" + testCertsDir + "/apiserver.key",
				"--kubelet-client-certificate=" + testCertsDir + "/apiserver-kubelet-client.crt",
				"--kubelet-client-key=" + testCertsDir + "/apiserver-kubelet-client.key",
				fmt.Sprintf("--secure-port=%d", 123),
				"--allow-privileged=true",
				"--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname",
				"--enable-bootstrap-token-auth=true",
				"--proxy-client-cert-file=/var/lib/certs/front-proxy-client.crt",
				"--proxy-client-key-file=/var/lib/certs/front-proxy-client.key",
				"--requestheader-username-headers=X-Remote-User",
				"--requestheader-group-headers=X-Remote-Group",
				"--requestheader-extra-headers-prefix=X-Remote-Extra-",
				"--requestheader-client-ca-file=" + testCertsDir + "/front-proxy-ca.crt",
				"--requestheader-allowed-names=front-proxy-client",
				"--authorization-mode=Node,RBAC",
				"--advertise-address=2001:db8::1",
				"--etcd-servers=https://[2001:abcd:bcda::1]:2379,https://[2001:abcd:bcda::2]:2379",
				"--etcd-cafile=fuz",
				"--etcd-certfile=fiz",
				"--etcd-keyfile=faz",
			},
		},
		{
			name: "an insecure etcd",
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking: kubeadmapi.Networking{ServiceSubnet: "bar"},
				Etcd: kubeadmapi.Etcd{
					External: &kubeadmapi.ExternalEtcd{
						Endpoints: []string{"http://[::1]:2379", "http://[::1]:2380"},
					},
				},
				CertificatesDir: testCertsDir,
			},
			endpoint: &kubeadmapi.APIEndpoint{BindPort: 123, AdvertiseAddress: "2001:db8::1"},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--enable-admission-plugins=NodeRestriction",
				"--service-cluster-ip-range=bar",
				"--service-account-key-file=" + testCertsDir + "/sa.pub",
				"--client-ca-file=" + testCertsDir + "/ca.crt",
				"--tls-cert-file=" + testCertsDir + "/apiserver.crt",
				"--tls-private-key-file=" + testCertsDir + "/apiserver.key",
				"--kubelet-client-certificate=" + testCertsDir + "/apiserver-kubelet-client.crt",
				"--kubelet-client-key=" + testCertsDir + "/apiserver-kubelet-client.key",
				fmt.Sprintf("--secure-port=%d", 123),
				"--allow-privileged=true",
				"--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname",
				"--enable-bootstrap-token-auth=true",
				"--proxy-client-cert-file=/var/lib/certs/front-proxy-client.crt",
				"--proxy-client-key-file=/var/lib/certs/front-proxy-client.key",
				"--requestheader-username-headers=X-Remote-User",
				"--requestheader-group-headers=X-Remote-Group",
				"--requestheader-extra-headers-prefix=X-Remote-Extra-",
				"--requestheader-client-ca-file=" + testCertsDir + "/front-proxy-ca.crt",
				"--requestheader-allowed-names=front-proxy-client",
				"--authorization-mode=Node,RBAC",
				"--advertise-address=2001:db8::1",
				"--etcd-servers=http://[::1]:2379,http://[::1]:2380",
			},
		},
		{
			name: "test APIServer.ExtraArgs works as expected",
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking:      kubeadmapi.Networking{ServiceSubnet: "bar"},
				CertificatesDir: testCertsDir,
				APIServer: kubeadmapi.APIServer{
					ControlPlaneComponent: kubeadmapi.ControlPlaneComponent{
						ExtraArgs: map[string]string{
							"service-cluster-ip-range": "baz",
							"advertise-address":        "9.9.9.9",
							"audit-policy-file":        "/etc/config/audit.yaml",
							"audit-log-path":           "/var/log/kubernetes",
						},
					},
				},
			},
			endpoint: &kubeadmapi.APIEndpoint{BindPort: 123, AdvertiseAddress: "1.2.3.4"},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--enable-admission-plugins=NodeRestriction",
				"--service-cluster-ip-range=baz",
				"--service-account-key-file=" + testCertsDir + "/sa.pub",
				"--client-ca-file=" + testCertsDir + "/ca.crt",
				"--tls-cert-file=" + testCertsDir + "/apiserver.crt",
				"--tls-private-key-file=" + testCertsDir + "/apiserver.key",
				"--kubelet-client-certificate=" + testCertsDir + "/apiserver-kubelet-client.crt",
				"--kubelet-client-key=" + testCertsDir + "/apiserver-kubelet-client.key",
				"--enable-bootstrap-token-auth=true",
				"--secure-port=123",
				"--allow-privileged=true",
				"--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname",
				"--proxy-client-cert-file=/var/lib/certs/front-proxy-client.crt",
				"--proxy-client-key-file=/var/lib/certs/front-proxy-client.key",
				"--requestheader-username-headers=X-Remote-User",
				"--requestheader-group-headers=X-Remote-Group",
				"--requestheader-extra-headers-prefix=X-Remote-Extra-",
				"--requestheader-client-ca-file=" + testCertsDir + "/front-proxy-ca.crt",
				"--requestheader-allowed-names=front-proxy-client",
				"--authorization-mode=Node,RBAC",
				"--advertise-address=9.9.9.9",
				fmt.Sprintf("--etcd-servers=https://127.0.0.1:%d", kubeadmconstants.EtcdListenClientPort),
				"--etcd-cafile=" + testCertsDir + "/etcd/ca.crt",
				"--etcd-certfile=" + testCertsDir + "/apiserver-etcd-client.crt",
				"--etcd-keyfile=" + testCertsDir + "/apiserver-etcd-client.key",
				"--audit-policy-file=/etc/config/audit.yaml",
				"--audit-log-path=/var/log/kubernetes",
			},
		},
		{
			name: "authorization-mode extra-args ABAC",
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking:      kubeadmapi.Networking{ServiceSubnet: "bar"},
				CertificatesDir: testCertsDir,
				APIServer: kubeadmapi.APIServer{
					ControlPlaneComponent: kubeadmapi.ControlPlaneComponent{
						ExtraArgs: map[string]string{
							"authorization-mode": kubeadmconstants.ModeABAC,
						},
					},
				},
			},
			endpoint: &kubeadmapi.APIEndpoint{BindPort: 123, AdvertiseAddress: "1.2.3.4"},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--enable-admission-plugins=NodeRestriction",
				"--service-cluster-ip-range=bar",
				"--service-account-key-file=" + testCertsDir + "/sa.pub",
				"--client-ca-file=" + testCertsDir + "/ca.crt",
				"--tls-cert-file=" + testCertsDir + "/apiserver.crt",
				"--tls-private-key-file=" + testCertsDir + "/apiserver.key",
				"--kubelet-client-certificate=" + testCertsDir + "/apiserver-kubelet-client.crt",
				"--kubelet-client-key=" + testCertsDir + "/apiserver-kubelet-client.key",
				"--enable-bootstrap-token-auth=true",
				"--secure-port=123",
				"--allow-privileged=true",
				"--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname",
				"--proxy-client-cert-file=/var/lib/certs/front-proxy-client.crt",
				"--proxy-client-key-file=/var/lib/certs/front-proxy-client.key",
				"--requestheader-username-headers=X-Remote-User",
				"--requestheader-group-headers=X-Remote-Group",
				"--requestheader-extra-headers-prefix=X-Remote-Extra-",
				"--requestheader-client-ca-file=" + testCertsDir + "/front-proxy-ca.crt",
				"--requestheader-allowed-names=front-proxy-client",
				"--authorization-mode=ABAC",
				"--advertise-address=1.2.3.4",
				fmt.Sprintf("--etcd-servers=https://127.0.0.1:%d", kubeadmconstants.EtcdListenClientPort),
				"--etcd-cafile=" + testCertsDir + "/etcd/ca.crt",
				"--etcd-certfile=" + testCertsDir + "/apiserver-etcd-client.crt",
				"--etcd-keyfile=" + testCertsDir + "/apiserver-etcd-client.key",
			},
		},
		{
			name: "insecure-port extra-args",
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking:      kubeadmapi.Networking{ServiceSubnet: "bar"},
				CertificatesDir: testCertsDir,
				APIServer: kubeadmapi.APIServer{
					ControlPlaneComponent: kubeadmapi.ControlPlaneComponent{
						ExtraArgs: map[string]string{
							"insecure-port": "1234",
						},
					},
				},
			},
			endpoint: &kubeadmapi.APIEndpoint{BindPort: 123, AdvertiseAddress: "1.2.3.4"},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=1234",
				"--enable-admission-plugins=NodeRestriction",
				"--service-cluster-ip-range=bar",
				"--service-account-key-file=" + testCertsDir + "/sa.pub",
				"--client-ca-file=" + testCertsDir + "/ca.crt",
				"--tls-cert-file=" + testCertsDir + "/apiserver.crt",
				"--tls-private-key-file=" + testCertsDir + "/apiserver.key",
				"--kubelet-client-certificate=" + testCertsDir + "/apiserver-kubelet-client.crt",
				"--kubelet-client-key=" + testCertsDir + "/apiserver-kubelet-client.key",
				"--enable-bootstrap-token-auth=true",
				"--secure-port=123",
				"--allow-privileged=true",
				"--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname",
				"--proxy-client-cert-file=/var/lib/certs/front-proxy-client.crt",
				"--proxy-client-key-file=/var/lib/certs/front-proxy-client.key",
				"--requestheader-username-headers=X-Remote-User",
				"--requestheader-group-headers=X-Remote-Group",
				"--requestheader-extra-headers-prefix=X-Remote-Extra-",
				"--requestheader-client-ca-file=" + testCertsDir + "/front-proxy-ca.crt",
				"--requestheader-allowed-names=front-proxy-client",
				"--authorization-mode=Node,RBAC",
				"--advertise-address=1.2.3.4",
				fmt.Sprintf("--etcd-servers=https://127.0.0.1:%d", kubeadmconstants.EtcdListenClientPort),
				"--etcd-cafile=" + testCertsDir + "/etcd/ca.crt",
				"--etcd-certfile=" + testCertsDir + "/apiserver-etcd-client.crt",
				"--etcd-keyfile=" + testCertsDir + "/apiserver-etcd-client.key",
			},
		},
		{
			name: "authorization-mode extra-args Webhook",
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking:      kubeadmapi.Networking{ServiceSubnet: "bar"},
				CertificatesDir: testCertsDir,
				APIServer: kubeadmapi.APIServer{
					ControlPlaneComponent: kubeadmapi.ControlPlaneComponent{
						ExtraArgs: map[string]string{
							"authorization-mode": strings.Join([]string{
								kubeadmconstants.ModeNode,
								kubeadmconstants.ModeRBAC,
								kubeadmconstants.ModeWebhook,
							}, ","),
						},
					},
				},
			},
			endpoint: &kubeadmapi.APIEndpoint{BindPort: 123, AdvertiseAddress: "1.2.3.4"},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--enable-admission-plugins=NodeRestriction",
				"--service-cluster-ip-range=bar",
				"--service-account-key-file=" + testCertsDir + "/sa.pub",
				"--client-ca-file=" + testCertsDir + "/ca.crt",
				"--tls-cert-file=" + testCertsDir + "/apiserver.crt",
				"--tls-private-key-file=" + testCertsDir + "/apiserver.key",
				"--kubelet-client-certificate=" + testCertsDir + "/apiserver-kubelet-client.crt",
				"--kubelet-client-key=" + testCertsDir + "/apiserver-kubelet-client.key",
				"--enable-bootstrap-token-auth=true",
				"--secure-port=123",
				"--allow-privileged=true",
				"--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname",
				"--proxy-client-cert-file=/var/lib/certs/front-proxy-client.crt",
				"--proxy-client-key-file=/var/lib/certs/front-proxy-client.key",
				"--requestheader-username-headers=X-Remote-User",
				"--requestheader-group-headers=X-Remote-Group",
				"--requestheader-extra-headers-prefix=X-Remote-Extra-",
				"--requestheader-client-ca-file=" + testCertsDir + "/front-proxy-ca.crt",
				"--requestheader-allowed-names=front-proxy-client",
				"--authorization-mode=Node,RBAC,Webhook",
				"--advertise-address=1.2.3.4",
				fmt.Sprintf("--etcd-servers=https://127.0.0.1:%d", kubeadmconstants.EtcdListenClientPort),
				"--etcd-cafile=" + testCertsDir + "/etcd/ca.crt",
				"--etcd-certfile=" + testCertsDir + "/apiserver-etcd-client.crt",
				"--etcd-keyfile=" + testCertsDir + "/apiserver-etcd-client.key",
			},
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			actual := getAPIServerCommand(rt.cfg, rt.endpoint)
			sort.Strings(actual)
			sort.Strings(rt.expected)
			if !reflect.DeepEqual(actual, rt.expected) {
				errorDiffArguments(t, rt.name, actual, rt.expected)
			}
		})
	}
}

func errorDiffArguments(t *testing.T, name string, actual, expected []string) {
	expectedShort := removeCommon(expected, actual)
	actualShort := removeCommon(actual, expected)
	t.Errorf(
		"[%s] failed getAPIServerCommand:\nexpected:\n%v\nsaw:\n%v"+
			"\nexpectedShort:\n%v\nsawShort:\n%v\n",
		name, expected, actual,
		expectedShort, actualShort)
}

// removeCommon removes common items from left list
// makes compairing two cmdline (with lots of arguments) easier
func removeCommon(left, right []string) []string {
	origSet := sets.NewString(left...)
	origSet.Delete(right...)
	return origSet.List()
}

func TestGetControllerManagerCommand(t *testing.T) {
	var tests = []struct {
		name     string
		cfg      *kubeadmapi.ClusterConfiguration
		expected []string
	}{
		{
			name: "custom cluster name for " + cpVersion,
			cfg: &kubeadmapi.ClusterConfiguration{
				KubernetesVersion: cpVersion,
				CertificatesDir:   testCertsDir,
				ClusterName:       "some-other-cluster-name",
			},
			expected: []string{
				"kube-controller-manager",
				"--bind-address=127.0.0.1",
				"--leader-elect=true",
				"--kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--root-ca-file=" + testCertsDir + "/ca.crt",
				"--service-account-private-key-file=" + testCertsDir + "/sa.key",
				"--cluster-signing-cert-file=" + testCertsDir + "/ca.crt",
				"--cluster-signing-key-file=" + testCertsDir + "/ca.key",
				"--use-service-account-credentials=true",
				"--controllers=*,bootstrapsigner,tokencleaner",
				"--authentication-kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--authorization-kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--client-ca-file=" + testCertsDir + "/ca.crt",
				"--requestheader-client-ca-file=" + testCertsDir + "/front-proxy-ca.crt",
				"--cluster-name=some-other-cluster-name",
			},
		},
		{
			name: "custom certs dir for " + cpVersion,
			cfg: &kubeadmapi.ClusterConfiguration{
				CertificatesDir:   testCertsDir,
				KubernetesVersion: cpVersion,
			},
			expected: []string{
				"kube-controller-manager",
				"--bind-address=127.0.0.1",
				"--leader-elect=true",
				"--kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--root-ca-file=" + testCertsDir + "/ca.crt",
				"--service-account-private-key-file=" + testCertsDir + "/sa.key",
				"--cluster-signing-cert-file=" + testCertsDir + "/ca.crt",
				"--cluster-signing-key-file=" + testCertsDir + "/ca.key",
				"--use-service-account-credentials=true",
				"--controllers=*,bootstrapsigner,tokencleaner",
				"--authentication-kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--authorization-kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--client-ca-file=" + testCertsDir + "/ca.crt",
				"--requestheader-client-ca-file=" + testCertsDir + "/front-proxy-ca.crt",
			},
		},
		{
			name: "custom cluster-cidr for " + cpVersion,
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking:        kubeadmapi.Networking{PodSubnet: "10.0.1.15/16"},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: cpVersion,
			},
			expected: []string{
				"kube-controller-manager",
				"--bind-address=127.0.0.1",
				"--leader-elect=true",
				"--kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--root-ca-file=" + testCertsDir + "/ca.crt",
				"--service-account-private-key-file=" + testCertsDir + "/sa.key",
				"--cluster-signing-cert-file=" + testCertsDir + "/ca.crt",
				"--cluster-signing-key-file=" + testCertsDir + "/ca.key",
				"--use-service-account-credentials=true",
				"--controllers=*,bootstrapsigner,tokencleaner",
				"--authentication-kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--authorization-kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--client-ca-file=" + testCertsDir + "/ca.crt",
				"--requestheader-client-ca-file=" + testCertsDir + "/front-proxy-ca.crt",
				"--allocate-node-cidrs=true",
				"--cluster-cidr=10.0.1.15/16",
				"--node-cidr-mask-size=24",
			},
		},
		{
			name: "custom service-cluster-ip-range for " + cpVersion,
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking: kubeadmapi.Networking{
					PodSubnet:     "10.0.1.15/16",
					ServiceSubnet: "172.20.0.0/24"},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: cpVersion,
			},
			expected: []string{
				"kube-controller-manager",
				"--bind-address=127.0.0.1",
				"--leader-elect=true",
				"--kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--root-ca-file=" + testCertsDir + "/ca.crt",
				"--service-account-private-key-file=" + testCertsDir + "/sa.key",
				"--cluster-signing-cert-file=" + testCertsDir + "/ca.crt",
				"--cluster-signing-key-file=" + testCertsDir + "/ca.key",
				"--use-service-account-credentials=true",
				"--controllers=*,bootstrapsigner,tokencleaner",
				"--authentication-kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--authorization-kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--client-ca-file=" + testCertsDir + "/ca.crt",
				"--requestheader-client-ca-file=" + testCertsDir + "/front-proxy-ca.crt",
				"--allocate-node-cidrs=true",
				"--cluster-cidr=10.0.1.15/16",
				"--node-cidr-mask-size=24",
				"--service-cluster-ip-range=172.20.0.0/24",
			},
		},
		{
			name: "custom extra-args for " + cpVersion,
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking: kubeadmapi.Networking{PodSubnet: "10.0.1.15/16"},
				ControllerManager: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: map[string]string{"node-cidr-mask-size": "20"},
				},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: cpVersion,
			},
			expected: []string{
				"kube-controller-manager",
				"--bind-address=127.0.0.1",
				"--leader-elect=true",
				"--kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--root-ca-file=" + testCertsDir + "/ca.crt",
				"--service-account-private-key-file=" + testCertsDir + "/sa.key",
				"--cluster-signing-cert-file=" + testCertsDir + "/ca.crt",
				"--cluster-signing-key-file=" + testCertsDir + "/ca.key",
				"--use-service-account-credentials=true",
				"--controllers=*,bootstrapsigner,tokencleaner",
				"--authentication-kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--authorization-kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--client-ca-file=" + testCertsDir + "/ca.crt",
				"--requestheader-client-ca-file=" + testCertsDir + "/front-proxy-ca.crt",
				"--allocate-node-cidrs=true",
				"--cluster-cidr=10.0.1.15/16",
				"--node-cidr-mask-size=20",
			},
		},
		{
			name: "custom IPv6 networking for " + cpVersion,
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking: kubeadmapi.Networking{
					PodSubnet:     "2001:db8::/64",
					ServiceSubnet: "fd03::/112",
				},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: cpVersion,
			},
			expected: []string{
				"kube-controller-manager",
				"--bind-address=127.0.0.1",
				"--leader-elect=true",
				"--kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--root-ca-file=" + testCertsDir + "/ca.crt",
				"--service-account-private-key-file=" + testCertsDir + "/sa.key",
				"--cluster-signing-cert-file=" + testCertsDir + "/ca.crt",
				"--cluster-signing-key-file=" + testCertsDir + "/ca.key",
				"--use-service-account-credentials=true",
				"--controllers=*,bootstrapsigner,tokencleaner",
				"--authentication-kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--authorization-kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--client-ca-file=" + testCertsDir + "/ca.crt",
				"--requestheader-client-ca-file=" + testCertsDir + "/front-proxy-ca.crt",
				"--allocate-node-cidrs=true",
				"--cluster-cidr=2001:db8::/64",
				"--node-cidr-mask-size=80",
				"--service-cluster-ip-range=fd03::/112",
			},
		},
		{
			name: "dual-stack networking for " + cpVersion,
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking: kubeadmapi.Networking{
					PodSubnet:     "2001:db8::/64,10.1.0.0/16",
					ServiceSubnet: "fd03::/112,192.168.0.0/16",
				},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: cpVersion,
				FeatureGates:      map[string]bool{features.IPv6DualStack: true},
			},
			expected: []string{
				"kube-controller-manager",
				"--bind-address=127.0.0.1",
				"--leader-elect=true",
				"--kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--root-ca-file=" + testCertsDir + "/ca.crt",
				"--service-account-private-key-file=" + testCertsDir + "/sa.key",
				"--cluster-signing-cert-file=" + testCertsDir + "/ca.crt",
				"--cluster-signing-key-file=" + testCertsDir + "/ca.key",
				"--use-service-account-credentials=true",
				"--controllers=*,bootstrapsigner,tokencleaner",
				"--authentication-kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--authorization-kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--client-ca-file=" + testCertsDir + "/ca.crt",
				"--requestheader-client-ca-file=" + testCertsDir + "/front-proxy-ca.crt",
				"--feature-gates=IPv6DualStack=true",
				"--allocate-node-cidrs=true",
				"--cluster-cidr=2001:db8::/64,10.1.0.0/16",
				"--node-cidr-mask-size-ipv4=24",
				"--node-cidr-mask-size-ipv6=80",
				"--service-cluster-ip-range=fd03::/112,192.168.0.0/16",
			},
		},
		{
			name: "dual-stack networking custom extra-args for " + cpVersion,
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking: kubeadmapi.Networking{PodSubnet: "10.0.1.15/16,2001:db8::/64"},
				ControllerManager: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: map[string]string{"node-cidr-mask-size-ipv4": "20", "node-cidr-mask-size-ipv6": "96"},
				},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: cpVersion,
				FeatureGates:      map[string]bool{features.IPv6DualStack: true},
			},
			expected: []string{
				"kube-controller-manager",
				"--bind-address=127.0.0.1",
				"--leader-elect=true",
				"--kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--root-ca-file=" + testCertsDir + "/ca.crt",
				"--service-account-private-key-file=" + testCertsDir + "/sa.key",
				"--cluster-signing-cert-file=" + testCertsDir + "/ca.crt",
				"--cluster-signing-key-file=" + testCertsDir + "/ca.key",
				"--use-service-account-credentials=true",
				"--controllers=*,bootstrapsigner,tokencleaner",
				"--authentication-kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--authorization-kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--client-ca-file=" + testCertsDir + "/ca.crt",
				"--requestheader-client-ca-file=" + testCertsDir + "/front-proxy-ca.crt",
				"--feature-gates=IPv6DualStack=true",
				"--allocate-node-cidrs=true",
				"--cluster-cidr=10.0.1.15/16,2001:db8::/64",
				"--node-cidr-mask-size-ipv4=20",
				"--node-cidr-mask-size-ipv6=96",
			},
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			actual := getControllerManagerCommand(rt.cfg)
			sort.Strings(actual)
			sort.Strings(rt.expected)
			if !reflect.DeepEqual(actual, rt.expected) {
				errorDiffArguments(t, rt.name, actual, rt.expected)
			}
		})
	}
}

func TestCalcNodeCidrSize(t *testing.T) {
	tests := []struct {
		name           string
		podSubnet      string
		expectedPrefix string
		expectedIPv6   bool
	}{
		{
			name:           "Malformed pod subnet",
			podSubnet:      "10.10.10/160",
			expectedPrefix: "24",
			expectedIPv6:   false,
		},
		{
			name:           "V4: Always uses 24",
			podSubnet:      "10.10.10.10/16",
			expectedPrefix: "24",
			expectedIPv6:   false,
		},
		{
			name:           "V6: Use pod subnet size, when not enough space",
			podSubnet:      "2001:db8::/128",
			expectedPrefix: "128",
			expectedIPv6:   true,
		},
		{
			name:           "V6: Use pod subnet size, when not enough space",
			podSubnet:      "2001:db8::/113",
			expectedPrefix: "113",
			expectedIPv6:   true,
		},
		{
			name:           "V6: Special case with 256 nodes",
			podSubnet:      "2001:db8::/112",
			expectedPrefix: "120",
			expectedIPv6:   true,
		},
		{
			name:           "V6: Using /120 for node CIDR",
			podSubnet:      "2001:db8::/104",
			expectedPrefix: "120",
			expectedIPv6:   true,
		},
		{
			name:           "V6: Using /112 for node CIDR",
			podSubnet:      "2001:db8::/103",
			expectedPrefix: "112",
			expectedIPv6:   true,
		},
		{
			name:           "V6: Using /112 for node CIDR",
			podSubnet:      "2001:db8::/96",
			expectedPrefix: "112",
			expectedIPv6:   true,
		},
		{
			name:           "V6: Using /104 for node CIDR",
			podSubnet:      "2001:db8::/95",
			expectedPrefix: "104",
			expectedIPv6:   true,
		},
		{
			name:           "V6: For /64 pod net, use /80",
			podSubnet:      "2001:db8::/64",
			expectedPrefix: "80",
			expectedIPv6:   true,
		},
		{
			name:           "V6: For /48 pod net, use /64",
			podSubnet:      "2001:db8::/48",
			expectedPrefix: "64",
			expectedIPv6:   true,
		},
		{
			name:           "V6: For /32 pod net, use /48",
			podSubnet:      "2001:db8::/32",
			expectedPrefix: "48",
			expectedIPv6:   true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actualPrefix, actualIPv6 := calcNodeCidrSize(test.podSubnet)
			if actualPrefix != test.expectedPrefix {
				t.Errorf("Case [%s]\nCalc of node CIDR size for pod subnet %q failed: Expected %q, saw %q",
					test.name, test.podSubnet, test.expectedPrefix, actualPrefix)
			}
			if actualIPv6 != test.expectedIPv6 {
				t.Errorf("Case [%s]\nCalc of node CIDR size for pod subnet %q failed: Expected isIPv6=%v, saw isIPv6=%v",
					test.name, test.podSubnet, test.expectedIPv6, actualIPv6)
			}
		})
	}

}
func TestGetControllerManagerCommandExternalCA(t *testing.T) {
	tests := []struct {
		name            string
		cfg             *kubeadmapi.InitConfiguration
		caKeyPresent    bool
		expectedArgFunc func(dir string) []string
	}{
		{
			name: "caKeyPresent-false for " + cpVersion,
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{AdvertiseAddress: "1.2.3.4"},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					KubernetesVersion: cpVersion,
					Networking:        kubeadmapi.Networking{ServiceSubnet: "10.96.0.0/12", DNSDomain: "cluster.local"},
				},
			},
			caKeyPresent: false,
			expectedArgFunc: func(tmpdir string) []string {
				return []string{
					"kube-controller-manager",
					"--bind-address=127.0.0.1",
					"--leader-elect=true",
					"--kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
					"--root-ca-file=" + tmpdir + "/ca.crt",
					"--service-account-private-key-file=" + tmpdir + "/sa.key",
					"--cluster-signing-cert-file=",
					"--cluster-signing-key-file=",
					"--use-service-account-credentials=true",
					"--controllers=*,bootstrapsigner,tokencleaner",
					"--authentication-kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
					"--authorization-kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
					"--client-ca-file=" + tmpdir + "/ca.crt",
					"--requestheader-client-ca-file=" + tmpdir + "/front-proxy-ca.crt",
				}
			},
		},
		{
			name: "caKeyPresent true for " + cpVersion,
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{AdvertiseAddress: "1.2.3.4"},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					KubernetesVersion: cpVersion,
					Networking:        kubeadmapi.Networking{ServiceSubnet: "10.96.0.0/12", DNSDomain: "cluster.local"},
				},
			},
			caKeyPresent: true,
			expectedArgFunc: func(tmpdir string) []string {
				return []string{
					"kube-controller-manager",
					"--bind-address=127.0.0.1",
					"--leader-elect=true",
					"--kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
					"--root-ca-file=" + tmpdir + "/ca.crt",
					"--service-account-private-key-file=" + tmpdir + "/sa.key",
					"--cluster-signing-cert-file=" + tmpdir + "/ca.crt",
					"--cluster-signing-key-file=" + tmpdir + "/ca.key",
					"--use-service-account-credentials=true",
					"--controllers=*,bootstrapsigner,tokencleaner",
					"--authentication-kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
					"--authorization-kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
					"--client-ca-file=" + tmpdir + "/ca.crt",
					"--requestheader-client-ca-file=" + tmpdir + "/front-proxy-ca.crt",
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Create temp folder for the test case
			tmpdir := testutil.SetupTempDir(t)
			defer os.RemoveAll(tmpdir)
			test.cfg.CertificatesDir = tmpdir

			if err := certs.CreatePKIAssets(test.cfg); err != nil {
				t.Errorf("failed creating pki assets: %v", err)
			}

			// delete ca.key and front-proxy-ca.key if test.caKeyPresent is false
			if !test.caKeyPresent {
				if err := os.Remove(filepath.Join(test.cfg.CertificatesDir, kubeadmconstants.CAKeyName)); err != nil {
					t.Errorf("failed removing %s: %v", kubeadmconstants.CAKeyName, err)
				}
				if err := os.Remove(filepath.Join(test.cfg.CertificatesDir, kubeadmconstants.FrontProxyCAKeyName)); err != nil {
					t.Errorf("failed removing %s: %v", kubeadmconstants.FrontProxyCAKeyName, err)
				}
			}

			actual := getControllerManagerCommand(&test.cfg.ClusterConfiguration)
			expected := test.expectedArgFunc(tmpdir)
			sort.Strings(actual)
			sort.Strings(expected)
			if !reflect.DeepEqual(actual, expected) {
				errorDiffArguments(t, test.name, actual, expected)
			}
		})
	}
}

func TestGetSchedulerCommand(t *testing.T) {
	var tests = []struct {
		name     string
		cfg      *kubeadmapi.ClusterConfiguration
		expected []string
	}{
		{
			name: "scheduler defaults",
			cfg:  &kubeadmapi.ClusterConfiguration{},
			expected: []string{
				"kube-scheduler",
				"--bind-address=127.0.0.1",
				"--leader-elect=true",
				"--kubeconfig=" + kubeadmconstants.KubernetesDir + "/scheduler.conf",
				"--authentication-kubeconfig=" + kubeadmconstants.KubernetesDir + "/scheduler.conf",
				"--authorization-kubeconfig=" + kubeadmconstants.KubernetesDir + "/scheduler.conf",
			},
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			actual := getSchedulerCommand(rt.cfg)
			sort.Strings(actual)
			sort.Strings(rt.expected)
			if !reflect.DeepEqual(actual, rt.expected) {
				errorDiffArguments(t, rt.name, actual, rt.expected)
			}
		})
	}
}

func TestGetAuthzModes(t *testing.T) {
	var tests = []struct {
		name     string
		authMode []string
		expected string
	}{
		{
			name:     "default if empty",
			authMode: []string{},
			expected: "Node,RBAC",
		},
		{
			name:     "default non empty",
			authMode: []string{kubeadmconstants.ModeNode, kubeadmconstants.ModeRBAC},
			expected: "Node,RBAC",
		},
		{
			name:     "single unspecified returning default",
			authMode: []string{"FooAuthzMode"},
			expected: "Node,RBAC",
		},
		{
			name:     "multiple ignored",
			authMode: []string{kubeadmconstants.ModeNode, "foo", kubeadmconstants.ModeRBAC, "bar"},
			expected: "Node,RBAC",
		},
		{
			name:     "single mode",
			authMode: []string{kubeadmconstants.ModeAlwaysDeny},
			expected: "AlwaysDeny",
		},
		{
			name:     "multiple special order",
			authMode: []string{kubeadmconstants.ModeNode, kubeadmconstants.ModeWebhook, kubeadmconstants.ModeRBAC, kubeadmconstants.ModeABAC},
			expected: "Node,Webhook,RBAC,ABAC",
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			actual := getAuthzModes(strings.Join(rt.authMode, ","))
			if actual != rt.expected {
				t.Errorf("failed getAuthzModes:\nexpected:\n%v\nsaw:\n%v", rt.expected, actual)
			}
		})
	}
}

func TestIsValidAuthzMode(t *testing.T) {
	var tests = []struct {
		mode  string
		valid bool
	}{
		{
			mode:  "Node",
			valid: true,
		},
		{
			mode:  "RBAC",
			valid: true,
		},
		{
			mode:  "ABAC",
			valid: true,
		},
		{
			mode:  "AlwaysAllow",
			valid: true,
		},
		{
			mode:  "Webhook",
			valid: true,
		},
		{
			mode:  "AlwaysDeny",
			valid: true,
		},
		{
			mode:  "Foo",
			valid: false,
		},
	}

	for _, rt := range tests {
		t.Run(rt.mode, func(t *testing.T) {
			isValid := isValidAuthzMode(rt.mode)
			if isValid != rt.valid {
				t.Errorf("failed isValidAuthzMode:\nexpected:\n%v\nsaw:\n%v", rt.valid, isValid)
			}
		})
	}
}

func TestCompareAuthzModes(t *testing.T) {
	var tests = []struct {
		name   string
		modesA []string
		modesB []string
		equal  bool
	}{
		{
			name:   "modes match",
			modesA: []string{"a", "b", "c"},
			modesB: []string{"a", "b", "c"},
			equal:  true,
		},
		{
			name:   "modes order does not match",
			modesA: []string{"a", "c", "b"},
			modesB: []string{"a", "b", "c"},
		},
		{
			name:   "modes do not match; A has less modes",
			modesA: []string{"a", "b"},
			modesB: []string{"a", "b", "c"},
		},
		{
			name:   "modes do not match; B has less modes",
			modesA: []string{"a", "b", "c"},
			modesB: []string{"a", "b"},
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			equal := compareAuthzModes(rt.modesA, rt.modesB)
			if equal != rt.equal {
				t.Errorf("failed compareAuthzModes:\nexpected:\n%v\nsaw:\n%v", rt.equal, equal)
			}
		})
	}
}
