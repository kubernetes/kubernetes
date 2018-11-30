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
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/version"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	authzmodes "k8s.io/kubernetes/pkg/kubeapiserver/authorizer/modes"

	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
)

const (
	testCertsDir = "/var/lib/certs"
	etcdDataDir  = "/var/lib/etcd"
)

func TestGetStaticPodSpecs(t *testing.T) {

	// Creates a Master Configuration
	cfg := &kubeadmapi.InitConfiguration{
		ClusterConfiguration: kubeadmapi.ClusterConfiguration{
			KubernetesVersion: "v1.9.0",
		},
	}

	// Executes GetStaticPodSpecs

	// TODO: Move the "pkg/util/version".Version object into the internal API instead of always parsing the string
	k8sVersion, _ := version.ParseSemantic(cfg.KubernetesVersion)

	specs := GetStaticPodSpecs(cfg, k8sVersion)

	var assertions = []struct {
		staticPodName string
	}{
		{
			staticPodName: kubeadmconstants.KubeAPIServer,
		},
		{
			staticPodName: kubeadmconstants.KubeControllerManager,
		},
		{
			staticPodName: kubeadmconstants.KubeScheduler,
		},
	}

	for _, assertion := range assertions {

		// assert the spec for the staticPodName exists
		if spec, ok := specs[assertion.staticPodName]; ok {

			// Assert each specs refers to the right pod
			if spec.Spec.Containers[0].Name != assertion.staticPodName {
				t.Errorf("getKubeConfigSpecs spec for %s contains pod %s, expects %s", assertion.staticPodName, spec.Spec.Containers[0].Name, assertion.staticPodName)
			}

		} else {
			t.Errorf("getStaticPodSpecs didn't create spec for %s ", assertion.staticPodName)
		}
	}
}

func TestCreateStaticPodFilesAndWrappers(t *testing.T) {

	var tests = []struct {
		components []string
	}{
		{
			components: []string{kubeadmconstants.KubeAPIServer, kubeadmconstants.KubeControllerManager, kubeadmconstants.KubeScheduler},
		},
		{
			components: []string{kubeadmconstants.KubeAPIServer},
		},
		{
			components: []string{kubeadmconstants.KubeControllerManager},
		},
		{
			components: []string{kubeadmconstants.KubeScheduler},
		},
	}

	for _, test := range tests {

		// Create temp folder for the test case
		tmpdir := testutil.SetupTempDir(t)
		defer os.RemoveAll(tmpdir)

		// Creates a Master Configuration
		cfg := &kubeadmapi.InitConfiguration{
			ClusterConfiguration: kubeadmapi.ClusterConfiguration{
				KubernetesVersion: "v1.9.0",
			},
		}

		// Execute createStaticPodFunction
		manifestPath := filepath.Join(tmpdir, kubeadmconstants.ManifestsSubDirName)
		err := CreateStaticPodFiles(manifestPath, cfg, test.components...)
		if err != nil {
			t.Errorf("Error executing createStaticPodFunction: %v", err)
			continue
		}

		// Assert expected files are there
		testutil.AssertFilesCount(t, manifestPath, len(test.components))

		for _, fileName := range test.components {
			testutil.AssertFileExists(t, manifestPath, fileName+".yaml")
		}
	}
}

func TestGetAPIServerCommand(t *testing.T) {
	var tests = []struct {
		name     string
		cfg      *kubeadmapi.InitConfiguration
		expected []string
	}{
		{
			name: "testing defaults",
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{BindPort: 123, AdvertiseAddress: "1.2.3.4"},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					Networking:      kubeadmapi.Networking{ServiceSubnet: "bar"},
					CertificatesDir: testCertsDir,
				},
			},
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
			name: "ignores the audit policy if the feature gate is not enabled",
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{BindPort: 123, AdvertiseAddress: "4.3.2.1"},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					Networking:      kubeadmapi.Networking{ServiceSubnet: "bar"},
					CertificatesDir: testCertsDir,
				},
			},
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
				"--advertise-address=4.3.2.1",
				fmt.Sprintf("--etcd-servers=https://127.0.0.1:%d", kubeadmconstants.EtcdListenClientPort),
				"--etcd-cafile=" + testCertsDir + "/etcd/ca.crt",
				"--etcd-certfile=" + testCertsDir + "/apiserver-etcd-client.crt",
				"--etcd-keyfile=" + testCertsDir + "/apiserver-etcd-client.key",
			},
		},
		{
			name: "ipv6 advertise address",
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{BindPort: 123, AdvertiseAddress: "2001:db8::1"},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					Networking:      kubeadmapi.Networking{ServiceSubnet: "bar"},
					CertificatesDir: testCertsDir,
				},
			},
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
				fmt.Sprintf("--etcd-servers=https://127.0.0.1:%d", kubeadmconstants.EtcdListenClientPort),
				"--etcd-cafile=" + testCertsDir + "/etcd/ca.crt",
				"--etcd-certfile=" + testCertsDir + "/apiserver-etcd-client.crt",
				"--etcd-keyfile=" + testCertsDir + "/apiserver-etcd-client.key",
			},
		},
		{
			name: "an external etcd with custom ca, certs and keys",
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{BindPort: 123, AdvertiseAddress: "2001:db8::1"},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					Networking: kubeadmapi.Networking{ServiceSubnet: "bar"},
					Etcd: kubeadmapi.Etcd{
						External: &kubeadmapi.ExternalEtcd{
							Endpoints: []string{"https://8.6.4.1:2379", "https://8.6.4.2:2379"},
							CAFile:    "fuz",
							CertFile:  "fiz",
							KeyFile:   "faz",
						},
					},
					CertificatesDir: testCertsDir,
				},
			},
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
				"--etcd-servers=https://8.6.4.1:2379,https://8.6.4.2:2379",
				"--etcd-cafile=fuz",
				"--etcd-certfile=fiz",
				"--etcd-keyfile=faz",
			},
		},
		{
			name: "an insecure etcd",
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{BindPort: 123, AdvertiseAddress: "2001:db8::1"},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					Networking: kubeadmapi.Networking{ServiceSubnet: "bar"},
					Etcd: kubeadmapi.Etcd{
						External: &kubeadmapi.ExternalEtcd{
							Endpoints: []string{"http://127.0.0.1:2379", "http://127.0.0.1:2380"},
						},
					},
					CertificatesDir: testCertsDir,
				},
			},
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
				"--etcd-servers=http://127.0.0.1:2379,http://127.0.0.1:2380",
			},
		},
		{
			name: "test APIServer.ExtraArgs works as expected",
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{BindPort: 123, AdvertiseAddress: "1.2.3.4"},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
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
			},
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
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{BindPort: 123, AdvertiseAddress: "1.2.3.4"},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					Networking:      kubeadmapi.Networking{ServiceSubnet: "bar"},
					CertificatesDir: testCertsDir,
					APIServer: kubeadmapi.APIServer{
						ControlPlaneComponent: kubeadmapi.ControlPlaneComponent{
							ExtraArgs: map[string]string{
								"authorization-mode": authzmodes.ModeABAC,
							},
						},
					},
				},
			},
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
				"--authorization-mode=Node,RBAC,ABAC",
				"--advertise-address=1.2.3.4",
				fmt.Sprintf("--etcd-servers=https://127.0.0.1:%d", kubeadmconstants.EtcdListenClientPort),
				"--etcd-cafile=" + testCertsDir + "/etcd/ca.crt",
				"--etcd-certfile=" + testCertsDir + "/apiserver-etcd-client.crt",
				"--etcd-keyfile=" + testCertsDir + "/apiserver-etcd-client.key",
			},
		},
		{
			name: "insecure-port extra-args",
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{BindPort: 123, AdvertiseAddress: "1.2.3.4"},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
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
			},
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
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{BindPort: 123, AdvertiseAddress: "1.2.3.4"},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					Networking:      kubeadmapi.Networking{ServiceSubnet: "bar"},
					CertificatesDir: testCertsDir,
					APIServer: kubeadmapi.APIServer{
						ControlPlaneComponent: kubeadmapi.ControlPlaneComponent{
							ExtraArgs: map[string]string{
								"authorization-mode": authzmodes.ModeWebhook,
							},
						},
					},
				},
			},
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
			actual := getAPIServerCommand(rt.cfg)
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
			name: "custom certs dir for v1.12.0-beta.2",
			cfg: &kubeadmapi.ClusterConfiguration{
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.12.0-beta.2",
			},
			expected: []string{
				"kube-controller-manager",
				"--address=127.0.0.1",
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
			name: "custom cloudprovider for v1.12.0-beta.2",
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking:        kubeadmapi.Networking{PodSubnet: "10.0.1.15/16"},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.12.0-beta.2",
			},
			expected: []string{
				"kube-controller-manager",
				"--address=127.0.0.1",
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
			name: "custom extra-args for v1.12.0-beta.2",
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking: kubeadmapi.Networking{PodSubnet: "10.0.1.15/16"},
				ControllerManager: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: map[string]string{"node-cidr-mask-size": "20"},
				},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.12.0-beta.2",
			},
			expected: []string{
				"kube-controller-manager",
				"--address=127.0.0.1",
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
			name: "custom IPv6 networking for v1.12.0-beta.2",
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking:        kubeadmapi.Networking{PodSubnet: "2001:db8::/64"},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.12.0-beta.2",
			},
			expected: []string{
				"kube-controller-manager",
				"--address=127.0.0.1",
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
			},
		},
		{
			name: "custom certs dir for v1.11.3",
			cfg: &kubeadmapi.ClusterConfiguration{
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.11.3",
			},
			expected: []string{
				"kube-controller-manager",
				"--address=127.0.0.1",
				"--leader-elect=true",
				"--kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--root-ca-file=" + testCertsDir + "/ca.crt",
				"--service-account-private-key-file=" + testCertsDir + "/sa.key",
				"--cluster-signing-cert-file=" + testCertsDir + "/ca.crt",
				"--cluster-signing-key-file=" + testCertsDir + "/ca.key",
				"--use-service-account-credentials=true",
				"--controllers=*,bootstrapsigner,tokencleaner",
			},
		},
		{
			name: "custom cloudprovider for v1.11.3",
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking:        kubeadmapi.Networking{PodSubnet: "10.0.1.15/16"},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.11.3",
			},
			expected: []string{
				"kube-controller-manager",
				"--address=127.0.0.1",
				"--leader-elect=true",
				"--kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--root-ca-file=" + testCertsDir + "/ca.crt",
				"--service-account-private-key-file=" + testCertsDir + "/sa.key",
				"--cluster-signing-cert-file=" + testCertsDir + "/ca.crt",
				"--cluster-signing-key-file=" + testCertsDir + "/ca.key",
				"--use-service-account-credentials=true",
				"--controllers=*,bootstrapsigner,tokencleaner",
				"--allocate-node-cidrs=true",
				"--cluster-cidr=10.0.1.15/16",
				"--node-cidr-mask-size=24",
			},
		},
		{
			name: "custom extra-args for v1.11.3",
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking: kubeadmapi.Networking{PodSubnet: "10.0.1.15/16"},
				ControllerManager: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: map[string]string{"node-cidr-mask-size": "20"},
				},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.11.3",
			},
			expected: []string{
				"kube-controller-manager",
				"--address=127.0.0.1",
				"--leader-elect=true",
				"--kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--root-ca-file=" + testCertsDir + "/ca.crt",
				"--service-account-private-key-file=" + testCertsDir + "/sa.key",
				"--cluster-signing-cert-file=" + testCertsDir + "/ca.crt",
				"--cluster-signing-key-file=" + testCertsDir + "/ca.key",
				"--use-service-account-credentials=true",
				"--controllers=*,bootstrapsigner,tokencleaner",
				"--allocate-node-cidrs=true",
				"--cluster-cidr=10.0.1.15/16",
				"--node-cidr-mask-size=20",
			},
		},
		{
			name: "custom IPv6 networking for v1.11.3",
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking:        kubeadmapi.Networking{PodSubnet: "2001:db8::/64"},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.11.3",
			},
			expected: []string{
				"kube-controller-manager",
				"--address=127.0.0.1",
				"--leader-elect=true",
				"--kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
				"--root-ca-file=" + testCertsDir + "/ca.crt",
				"--service-account-private-key-file=" + testCertsDir + "/sa.key",
				"--cluster-signing-cert-file=" + testCertsDir + "/ca.crt",
				"--cluster-signing-key-file=" + testCertsDir + "/ca.key",
				"--use-service-account-credentials=true",
				"--controllers=*,bootstrapsigner,tokencleaner",
				"--allocate-node-cidrs=true",
				"--cluster-cidr=2001:db8::/64",
				"--node-cidr-mask-size=80",
			},
		},
	}

	for _, rt := range tests {
		// TODO: Make getControllerManagerCommand accept a ClusterConfiguration object instead of InitConfiguration
		initcfg := &kubeadmapi.InitConfiguration{
			ClusterConfiguration: *rt.cfg,
		}
		actual := getControllerManagerCommand(initcfg, version.MustParseSemantic(rt.cfg.KubernetesVersion))
		sort.Strings(actual)
		sort.Strings(rt.expected)
		if !reflect.DeepEqual(actual, rt.expected) {
			errorDiffArguments(t, rt.name, actual, rt.expected)
		}
	}
}

func TestCalcNodeCidrSize(t *testing.T) {
	tests := []struct {
		name           string
		podSubnet      string
		expectedPrefix string
	}{
		{
			name:           "Malformed pod subnet",
			podSubnet:      "10.10.10/160",
			expectedPrefix: "24",
		},
		{
			name:           "V4: Always uses 24",
			podSubnet:      "10.10.10.10/16",
			expectedPrefix: "24",
		},
		{
			name:           "V6: Use pod subnet size, when not enough space",
			podSubnet:      "2001:db8::/128",
			expectedPrefix: "128",
		},
		{
			name:           "V6: Use pod subnet size, when not enough space",
			podSubnet:      "2001:db8::/113",
			expectedPrefix: "113",
		},
		{
			name:           "V6: Special case with 256 nodes",
			podSubnet:      "2001:db8::/112",
			expectedPrefix: "120",
		},
		{
			name:           "V6: Using /120 for node CIDR",
			podSubnet:      "2001:db8::/104",
			expectedPrefix: "120",
		},
		{
			name:           "V6: Using /112 for node CIDR",
			podSubnet:      "2001:db8::/103",
			expectedPrefix: "112",
		},
		{
			name:           "V6: Using /112 for node CIDR",
			podSubnet:      "2001:db8::/96",
			expectedPrefix: "112",
		},
		{
			name:           "V6: Using /104 for node CIDR",
			podSubnet:      "2001:db8::/95",
			expectedPrefix: "104",
		},
		{
			name:           "V6: For /64 pod net, use /80",
			podSubnet:      "2001:db8::/64",
			expectedPrefix: "80",
		},
		{
			name:           "V6: For /48 pod net, use /64",
			podSubnet:      "2001:db8::/48",
			expectedPrefix: "64",
		},
		{
			name:           "V6: For /32 pod net, use /48",
			podSubnet:      "2001:db8::/32",
			expectedPrefix: "48",
		},
	}
	for _, test := range tests {
		actualPrefix := calcNodeCidrSize(test.podSubnet)
		if actualPrefix != test.expectedPrefix {
			t.Errorf("Case [%s]\nCalc of node CIDR size for pod subnet %q failed: Expected %q, saw %q",
				test.name, test.podSubnet, test.expectedPrefix, actualPrefix)
		}
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
			name: "caKeyPresent-false for v1.12.0-beta.2",
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{AdvertiseAddress: "1.2.3.4"},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					KubernetesVersion: "v1.12.0-beta.2",
					Networking:        kubeadmapi.Networking{ServiceSubnet: "10.96.0.0/12", DNSDomain: "cluster.local"},
				},
			},
			caKeyPresent: false,
			expectedArgFunc: func(tmpdir string) []string {
				return []string{
					"kube-controller-manager",
					"--address=127.0.0.1",
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
			name: "caKeyPresent true for v1.12.0-beta.2",
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{AdvertiseAddress: "1.2.3.4"},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					KubernetesVersion: "v1.12.0-beta.2",
					Networking:        kubeadmapi.Networking{ServiceSubnet: "10.96.0.0/12", DNSDomain: "cluster.local"},
				},
			},
			caKeyPresent: true,
			expectedArgFunc: func(tmpdir string) []string {
				return []string{
					"kube-controller-manager",
					"--address=127.0.0.1",
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
		{
			name: "caKeyPresent-false for v1.11.3",
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{AdvertiseAddress: "1.2.3.4"},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					KubernetesVersion: "v1.11.3",
					Networking:        kubeadmapi.Networking{ServiceSubnet: "10.96.0.0/12", DNSDomain: "cluster.local"},
				},
			},
			caKeyPresent: false,
			expectedArgFunc: func(tmpdir string) []string {
				return []string{
					"kube-controller-manager",
					"--address=127.0.0.1",
					"--leader-elect=true",
					"--kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
					"--root-ca-file=" + tmpdir + "/ca.crt",
					"--service-account-private-key-file=" + tmpdir + "/sa.key",
					"--cluster-signing-cert-file=",
					"--cluster-signing-key-file=",
					"--use-service-account-credentials=true",
					"--controllers=*,bootstrapsigner,tokencleaner",
				}
			},
		},
		{
			name: "caKeyPresent true for v1.11.3",
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{AdvertiseAddress: "1.2.3.4"},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					KubernetesVersion: "v1.11.3",
					Networking:        kubeadmapi.Networking{ServiceSubnet: "10.96.0.0/12", DNSDomain: "cluster.local"},
				},
			},
			caKeyPresent: true,
			expectedArgFunc: func(tmpdir string) []string {
				return []string{
					"kube-controller-manager",
					"--address=127.0.0.1",
					"--leader-elect=true",
					"--kubeconfig=" + kubeadmconstants.KubernetesDir + "/controller-manager.conf",
					"--root-ca-file=" + tmpdir + "/ca.crt",
					"--service-account-private-key-file=" + tmpdir + "/sa.key",
					"--cluster-signing-cert-file=" + tmpdir + "/ca.crt",
					"--cluster-signing-key-file=" + tmpdir + "/ca.key",
					"--use-service-account-credentials=true",
					"--controllers=*,bootstrapsigner,tokencleaner",
				}
			},
		},
	}

	for _, test := range tests {
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

		actual := getControllerManagerCommand(test.cfg, version.MustParseSemantic(test.cfg.KubernetesVersion))
		expected := test.expectedArgFunc(tmpdir)
		sort.Strings(actual)
		sort.Strings(expected)
		if !reflect.DeepEqual(actual, expected) {
			errorDiffArguments(t, test.name, actual, expected)
		}
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
				"--address=127.0.0.1",
				"--leader-elect=true",
				"--kubeconfig=" + kubeadmconstants.KubernetesDir + "/scheduler.conf",
			},
		},
	}

	for _, rt := range tests {
		// TODO: Make getSchedulerCommand accept a ClusterConfiguration object instead of InitConfiguration
		initcfg := &kubeadmapi.InitConfiguration{
			ClusterConfiguration: *rt.cfg,
		}
		actual := getSchedulerCommand(initcfg)
		sort.Strings(actual)
		sort.Strings(rt.expected)
		if !reflect.DeepEqual(actual, rt.expected) {
			errorDiffArguments(t, rt.name, actual, rt.expected)
		}
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
			name:     "add missing Node",
			authMode: []string{authzmodes.ModeRBAC},
			expected: "Node,RBAC",
		},
		{
			name:     "add missing RBAC",
			authMode: []string{authzmodes.ModeNode},
			expected: "Node,RBAC",
		},
		{
			name:     "add defaults to ABAC",
			authMode: []string{authzmodes.ModeABAC},
			expected: "Node,RBAC,ABAC",
		},
		{
			name:     "add defaults to RBAC+Webhook",
			authMode: []string{authzmodes.ModeRBAC, authzmodes.ModeWebhook},
			expected: "Node,RBAC,Webhook",
		},
		{
			name:     "add default to Webhook",
			authMode: []string{authzmodes.ModeWebhook},
			expected: "Node,RBAC,Webhook",
		},
		{
			name:     "AlwaysAllow ignored",
			authMode: []string{authzmodes.ModeAlwaysAllow},
			expected: "Node,RBAC",
		},
		{
			name:     "AlwaysDeny ignored",
			authMode: []string{authzmodes.ModeAlwaysDeny},
			expected: "Node,RBAC",
		},
		{
			name:     "Unspecified ignored",
			authMode: []string{"FooAuthzMode"},
			expected: "Node,RBAC",
		},
		{
			name:     "Multiple ignored",
			authMode: []string{authzmodes.ModeAlwaysAllow, authzmodes.ModeAlwaysDeny, "foo"},
			expected: "Node,RBAC",
		},
		{
			name:     "all",
			authMode: []string{authzmodes.ModeNode, authzmodes.ModeRBAC, authzmodes.ModeWebhook, authzmodes.ModeABAC},
			expected: "Node,RBAC,ABAC,Webhook",
		},
	}

	for _, rt := range tests {

		t.Run(rt.name, func(t *testing.T) {
			actual := getAuthzModes(strings.Join(rt.authMode, ","))
			if actual != rt.expected {
				t.Errorf("failed getAuthzParameters:\nexpected:\n%v\nsaw:\n%v", rt.expected, actual)
			}
		})
	}
}
