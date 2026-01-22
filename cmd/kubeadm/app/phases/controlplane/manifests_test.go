//go:build !windows

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

	"github.com/lithammer/dedent"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	pkiutiltesting "k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil/testing"
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
		Scheduler: kubeadmapi.ControlPlaneComponent{ExtraEnvs: []kubeadmapi.EnvVar{
			{
				EnvVar: v1.EnvVar{Name: "Foo", Value: "Bar"},
			},
		}},
	}

	// Executes GetStaticPodSpecs
	specs := GetStaticPodSpecs(cfg, &kubeadmapi.APIEndpoint{
		BindPort: kubeadmconstants.KubeAPIServerPort,
	}, []kubeadmapi.EnvVar{})

	var tests = []struct {
		name                 string
		staticPodName        string
		expectLivenessProbe  bool
		expectReadinessProbe bool
		expectStartupProbe   bool
		probePort            string
		env                  []v1.EnvVar
		containerPorts       []v1.ContainerPort
	}{
		{
			name:                 "KubeAPIServer",
			staticPodName:        kubeadmconstants.KubeAPIServer,
			expectLivenessProbe:  true,
			expectReadinessProbe: true,
			expectStartupProbe:   true,
			probePort:            kubeadmconstants.ProbePort,
			containerPorts: []v1.ContainerPort{
				{
					Name:          kubeadmconstants.ProbePort,
					ContainerPort: kubeadmconstants.KubeAPIServerPort,
					Protocol:      v1.ProtocolTCP,
				},
			},
		},
		{
			name:                 "KubeControllerManager",
			staticPodName:        kubeadmconstants.KubeControllerManager,
			expectLivenessProbe:  true,
			expectReadinessProbe: false,
			expectStartupProbe:   true,
			probePort:            kubeadmconstants.ProbePort,
			containerPorts: []v1.ContainerPort{
				{
					Name:          kubeadmconstants.ProbePort,
					ContainerPort: kubeadmconstants.KubeControllerManagerPort,
					Protocol:      v1.ProtocolTCP,
				},
			},
		},
		{
			name:                 "KubeScheduler",
			staticPodName:        kubeadmconstants.KubeScheduler,
			expectLivenessProbe:  true,
			expectReadinessProbe: true,
			expectStartupProbe:   true,
			probePort:            kubeadmconstants.ProbePort,
			env:                  []v1.EnvVar{{Name: "Foo", Value: "Bar"}},
			containerPorts: []v1.ContainerPort{
				{
					Name:          kubeadmconstants.ProbePort,
					ContainerPort: kubeadmconstants.KubeSchedulerPort,
					Protocol:      v1.ProtocolTCP,
				},
			},
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
				if tc.env != nil {
					if !reflect.DeepEqual(spec.Spec.Containers[0].Env, tc.env) {
						t.Errorf("expected env: %v, got: %v", tc.env, spec.Spec.Containers[0].Env)
					}
				}

				if tc.expectLivenessProbe != (spec.Spec.Containers[0].LivenessProbe != nil) {
					t.Errorf("expected livenessProbe: %v, got: %v", tc.expectLivenessProbe, (spec.Spec.Containers[0].LivenessProbe != nil))
				}
				if tc.expectReadinessProbe != (spec.Spec.Containers[0].ReadinessProbe != nil) {
					t.Errorf("expected readinessProbe: %v, got: %v", tc.expectReadinessProbe, (spec.Spec.Containers[0].ReadinessProbe != nil))
				}
				if tc.expectStartupProbe != (spec.Spec.Containers[0].StartupProbe != nil) {
					t.Errorf("expected startupProbe: %v, got: %v", tc.expectStartupProbe, (spec.Spec.Containers[0].StartupProbe != nil))
				}

				if len(tc.probePort) > 0 {
					if spec.Spec.Containers[0].LivenessProbe != nil && !reflect.DeepEqual(intstr.FromString(tc.probePort), spec.Spec.Containers[0].LivenessProbe.HTTPGet.Port) {
						t.Errorf("expected livenessProbe port: %v, got: %v", intstr.FromString(tc.probePort), spec.Spec.Containers[0].LivenessProbe.HTTPGet.Port)
					}
					if spec.Spec.Containers[0].ReadinessProbe != nil && len(tc.probePort) > 0 && !reflect.DeepEqual(intstr.FromString(tc.probePort), spec.Spec.Containers[0].ReadinessProbe.HTTPGet.Port) {
						t.Errorf("expected readinessProbe port: %v, got: %v", intstr.FromString(tc.probePort), spec.Spec.Containers[0].ReadinessProbe.HTTPGet.Port)
					}
					if spec.Spec.Containers[0].StartupProbe != nil && !reflect.DeepEqual(intstr.FromString(tc.probePort), spec.Spec.Containers[0].StartupProbe.HTTPGet.Port) {
						t.Errorf("expected startupProbe port: %v, got: %v", intstr.FromString(tc.probePort), spec.Spec.Containers[0].StartupProbe.HTTPGet.Port)
					}
				}

				if len(tc.containerPorts) > 0 {
					if !reflect.DeepEqual(spec.Spec.Containers[0].Ports, tc.containerPorts) {
						t.Errorf("expected ports: %v, got: %v", tc.containerPorts, spec.Spec.Containers[0].Ports)
					}
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
			name:       "KubeAPIServer KubeControllerManager KubeScheduler",
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
			tmpdir := t.TempDir()

			// Creates a Cluster Configuration
			cfg := &kubeadmapi.ClusterConfiguration{
				KubernetesVersion: "v1.9.0",
			}

			// Execute createStaticPodFunction
			manifestPath := filepath.Join(tmpdir, kubeadmconstants.ManifestsSubDirName)
			err := CreateStaticPodFiles(manifestPath, "", cfg, &kubeadmapi.APIEndpoint{}, false /* isDryRun */, test.components...)
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

func TestCreateStaticPodFilesWithPatches(t *testing.T) {
	// Create temp folder for the test case
	tmpdir := t.TempDir()

	// Creates a Cluster Configuration
	cfg := &kubeadmapi.ClusterConfiguration{
		KubernetesVersion: "v1.9.0",
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

	err = os.WriteFile(filepath.Join(patchesPath, kubeadmconstants.KubeAPIServer+".yaml"), []byte(patchString), 0644)
	if err != nil {
		t.Fatalf("WriteFile returned unexpected error: %v", err)
	}

	// Execute createStaticPodFunction with patches
	manifestPath := filepath.Join(tmpdir, kubeadmconstants.ManifestsSubDirName)
	err = CreateStaticPodFiles(manifestPath, patchesPath, cfg, &kubeadmapi.APIEndpoint{}, false /* isDryRun */, kubeadmconstants.KubeAPIServer)
	if err != nil {
		t.Errorf("Error executing createStaticPodFunction: %v", err)
		return
	}

	pod, err := staticpodutil.ReadStaticPodFromDisk(filepath.Join(manifestPath, fmt.Sprintf("%s.yaml", kubeadmconstants.KubeAPIServer)))
	if err != nil {
		t.Errorf("Error executing ReadStaticPodFromDisk: %v", err)
		return
	}

	if _, ok := pod.ObjectMeta.Annotations["patched"]; !ok {
		t.Errorf("Patches were not applied to %s", kubeadmconstants.KubeAPIServer)
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
				Networking:      kubeadmapi.Networking{ServiceSubnet: "bar", DNSDomain: "cluster.local"},
				CertificatesDir: testCertsDir,
			},
			endpoint: &kubeadmapi.APIEndpoint{BindPort: 123, AdvertiseAddress: "1.2.3.4"},
			expected: []string{
				"kube-apiserver",
				"--enable-admission-plugins=NodeRestriction",
				"--service-cluster-ip-range=bar",
				"--service-account-key-file=" + filepath.Join(testCertsDir, "sa.pub"),
				"--service-account-signing-key-file=" + filepath.Join(testCertsDir, "sa.key"),
				"--service-account-issuer=https://kubernetes.default.svc.cluster.local",
				"--client-ca-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--tls-cert-file=" + filepath.Join(testCertsDir, "apiserver.crt"),
				"--tls-private-key-file=" + filepath.Join(testCertsDir, "apiserver.key"),
				"--kubelet-client-certificate=" + filepath.Join(testCertsDir, "apiserver-kubelet-client.crt"),
				"--kubelet-client-key=" + filepath.Join(testCertsDir, "apiserver-kubelet-client.key"),
				"--enable-bootstrap-token-auth=true",
				"--secure-port=123",
				"--allow-privileged=true",
				"--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname",
				"--proxy-client-cert-file=" + filepath.FromSlash("/var/lib/certs/front-proxy-client.crt"),
				"--proxy-client-key-file=" + filepath.FromSlash("/var/lib/certs/front-proxy-client.key"),
				"--requestheader-username-headers=X-Remote-User",
				"--requestheader-group-headers=X-Remote-Group",
				"--requestheader-extra-headers-prefix=X-Remote-Extra-",
				"--requestheader-client-ca-file=" + filepath.Join(testCertsDir, "front-proxy-ca.crt"),
				"--requestheader-allowed-names=front-proxy-client",
				"--authorization-mode=Node,RBAC",
				"--advertise-address=1.2.3.4",
				fmt.Sprintf("--etcd-servers=https://127.0.0.1:%d", kubeadmconstants.EtcdListenClientPort),
				"--etcd-cafile=" + filepath.Join(testCertsDir, "etcd/ca.crt"),
				"--etcd-certfile=" + filepath.Join(testCertsDir, "apiserver-etcd-client.crt"),
				"--etcd-keyfile=" + filepath.Join(testCertsDir, "apiserver-etcd-client.key"),
			},
		},
		{
			name: "ipv6 advertise address",
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking:      kubeadmapi.Networking{ServiceSubnet: "bar", DNSDomain: "cluster.local"},
				CertificatesDir: testCertsDir,
			},
			endpoint: &kubeadmapi.APIEndpoint{BindPort: 123, AdvertiseAddress: "2001:db8::1"},
			expected: []string{
				"kube-apiserver",
				"--enable-admission-plugins=NodeRestriction",
				"--service-cluster-ip-range=bar",
				"--service-account-key-file=" + filepath.Join(testCertsDir, "sa.pub"),
				"--service-account-signing-key-file=" + filepath.Join(testCertsDir, "sa.key"),
				"--service-account-issuer=https://kubernetes.default.svc.cluster.local",
				"--client-ca-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--tls-cert-file=" + filepath.Join(testCertsDir, "apiserver.crt"),
				"--tls-private-key-file=" + filepath.Join(testCertsDir, "apiserver.key"),
				"--kubelet-client-certificate=" + filepath.Join(testCertsDir, "apiserver-kubelet-client.crt"),
				"--kubelet-client-key=" + filepath.Join(testCertsDir, "apiserver-kubelet-client.key"),
				"--enable-bootstrap-token-auth=true",
				fmt.Sprintf("--secure-port=%d", 123),
				"--allow-privileged=true",
				"--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname",
				"--proxy-client-cert-file=" + filepath.FromSlash("/var/lib/certs/front-proxy-client.crt"),
				"--proxy-client-key-file=" + filepath.FromSlash("/var/lib/certs/front-proxy-client.key"),
				"--requestheader-username-headers=X-Remote-User",
				"--requestheader-group-headers=X-Remote-Group",
				"--requestheader-extra-headers-prefix=X-Remote-Extra-",
				"--requestheader-client-ca-file=" + filepath.Join(testCertsDir, "front-proxy-ca.crt"),
				"--requestheader-allowed-names=front-proxy-client",
				"--authorization-mode=Node,RBAC",
				"--advertise-address=2001:db8::1",
				fmt.Sprintf("--etcd-servers=https://[::1]:%d", kubeadmconstants.EtcdListenClientPort),
				"--etcd-cafile=" + filepath.Join(testCertsDir, "etcd/ca.crt"),
				"--etcd-certfile=" + filepath.Join(testCertsDir, "apiserver-etcd-client.crt"),
				"--etcd-keyfile=" + filepath.Join(testCertsDir, "apiserver-etcd-client.key"),
			},
		},
		{
			name: "an external etcd with custom ca, certs and keys",
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking: kubeadmapi.Networking{ServiceSubnet: "bar", DNSDomain: "cluster.local"},
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
				"--enable-admission-plugins=NodeRestriction",
				"--service-cluster-ip-range=bar",
				"--service-account-key-file=" + filepath.Join(testCertsDir, "sa.pub"),
				"--service-account-signing-key-file=" + filepath.Join(testCertsDir, "sa.key"),
				"--service-account-issuer=https://kubernetes.default.svc.cluster.local",
				"--client-ca-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--tls-cert-file=" + filepath.Join(testCertsDir, "apiserver.crt"),
				"--tls-private-key-file=" + filepath.Join(testCertsDir, "apiserver.key"),
				"--kubelet-client-certificate=" + filepath.Join(testCertsDir, "apiserver-kubelet-client.crt"),
				"--kubelet-client-key=" + filepath.Join(testCertsDir, "apiserver-kubelet-client.key"),
				fmt.Sprintf("--secure-port=%d", 123),
				"--allow-privileged=true",
				"--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname",
				"--enable-bootstrap-token-auth=true",
				"--proxy-client-cert-file=" + filepath.FromSlash("/var/lib/certs/front-proxy-client.crt"),
				"--proxy-client-key-file=" + filepath.FromSlash("/var/lib/certs/front-proxy-client.key"),
				"--requestheader-username-headers=X-Remote-User",
				"--requestheader-group-headers=X-Remote-Group",
				"--requestheader-extra-headers-prefix=X-Remote-Extra-",
				"--requestheader-client-ca-file=" + filepath.Join(testCertsDir, "front-proxy-ca.crt"),
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
				Networking: kubeadmapi.Networking{ServiceSubnet: "bar", DNSDomain: "cluster.local"},
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
				"--enable-admission-plugins=NodeRestriction",
				"--service-cluster-ip-range=bar",
				"--service-account-key-file=" + filepath.Join(testCertsDir, "sa.pub"),
				"--service-account-signing-key-file=" + filepath.Join(testCertsDir, "sa.key"),
				"--service-account-issuer=https://kubernetes.default.svc.cluster.local",
				"--client-ca-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--tls-cert-file=" + filepath.Join(testCertsDir, "apiserver.crt"),
				"--tls-private-key-file=" + filepath.Join(testCertsDir, "apiserver.key"),
				"--kubelet-client-certificate=" + filepath.Join(testCertsDir, "apiserver-kubelet-client.crt"),
				"--kubelet-client-key=" + filepath.Join(testCertsDir, "apiserver-kubelet-client.key"),
				fmt.Sprintf("--secure-port=%d", 123),
				"--allow-privileged=true",
				"--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname",
				"--enable-bootstrap-token-auth=true",
				"--proxy-client-cert-file=" + filepath.FromSlash("/var/lib/certs/front-proxy-client.crt"),
				"--proxy-client-key-file=" + filepath.FromSlash("/var/lib/certs/front-proxy-client.key"),
				"--requestheader-username-headers=X-Remote-User",
				"--requestheader-group-headers=X-Remote-Group",
				"--requestheader-extra-headers-prefix=X-Remote-Extra-",
				"--requestheader-client-ca-file=" + filepath.Join(testCertsDir, "front-proxy-ca.crt"),
				"--requestheader-allowed-names=front-proxy-client",
				"--authorization-mode=Node,RBAC",
				"--advertise-address=2001:db8::1",
				"--etcd-servers=http://[::1]:2379,http://[::1]:2380",
			},
		},
		{
			name: "test APIServer.ExtraArgs works as expected",
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking:      kubeadmapi.Networking{ServiceSubnet: "bar", DNSDomain: "cluster.local"},
				CertificatesDir: testCertsDir,
				APIServer: kubeadmapi.APIServer{
					ControlPlaneComponent: kubeadmapi.ControlPlaneComponent{
						ExtraArgs: []kubeadmapi.Arg{
							{Name: "service-cluster-ip-range", Value: "baz"},
							{Name: "advertise-address", Value: "9.9.9.9"},
							{Name: "audit-policy-file", Value: "/etc/config/audit.yaml"},
							{Name: "audit-log-path", Value: "/var/log/kubernetes"},
						},
					},
				},
			},
			endpoint: &kubeadmapi.APIEndpoint{BindPort: 123, AdvertiseAddress: "1.2.3.4"},
			expected: []string{
				"kube-apiserver",
				"--enable-admission-plugins=NodeRestriction",
				"--service-cluster-ip-range=baz",
				"--service-account-key-file=" + filepath.Join(testCertsDir, "sa.pub"),
				"--service-account-signing-key-file=" + filepath.Join(testCertsDir, "sa.key"),
				"--service-account-issuer=https://kubernetes.default.svc.cluster.local",
				"--client-ca-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--tls-cert-file=" + filepath.Join(testCertsDir, "apiserver.crt"),
				"--tls-private-key-file=" + filepath.Join(testCertsDir, "apiserver.key"),
				"--kubelet-client-certificate=" + filepath.Join(testCertsDir, "apiserver-kubelet-client.crt"),
				"--kubelet-client-key=" + filepath.Join(testCertsDir, "apiserver-kubelet-client.key"),
				"--enable-bootstrap-token-auth=true",
				"--secure-port=123",
				"--allow-privileged=true",
				"--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname",
				"--proxy-client-cert-file=" + filepath.FromSlash("/var/lib/certs/front-proxy-client.crt"),
				"--proxy-client-key-file=" + filepath.FromSlash("/var/lib/certs/front-proxy-client.key"),
				"--requestheader-username-headers=X-Remote-User",
				"--requestheader-group-headers=X-Remote-Group",
				"--requestheader-extra-headers-prefix=X-Remote-Extra-",
				"--requestheader-client-ca-file=" + filepath.Join(testCertsDir, "front-proxy-ca.crt"),
				"--requestheader-allowed-names=front-proxy-client",
				"--authorization-mode=Node,RBAC",
				"--advertise-address=9.9.9.9",
				fmt.Sprintf("--etcd-servers=https://127.0.0.1:%d", kubeadmconstants.EtcdListenClientPort),
				"--etcd-cafile=" + filepath.Join(testCertsDir, "etcd/ca.crt"),
				"--etcd-certfile=" + filepath.Join(testCertsDir, "apiserver-etcd-client.crt"),
				"--etcd-keyfile=" + filepath.Join(testCertsDir, "apiserver-etcd-client.key"),
				"--audit-policy-file=/etc/config/audit.yaml",
				"--audit-log-path=/var/log/kubernetes",
			},
		},
		{
			name: "authorization-mode extra-args ABAC",
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking:      kubeadmapi.Networking{ServiceSubnet: "bar", DNSDomain: "cluster.local"},
				CertificatesDir: testCertsDir,
				APIServer: kubeadmapi.APIServer{
					ControlPlaneComponent: kubeadmapi.ControlPlaneComponent{
						ExtraArgs: []kubeadmapi.Arg{
							{Name: "authorization-mode", Value: kubeadmconstants.ModeABAC},
						},
					},
				},
			},
			endpoint: &kubeadmapi.APIEndpoint{BindPort: 123, AdvertiseAddress: "1.2.3.4"},
			expected: []string{
				"kube-apiserver",
				"--enable-admission-plugins=NodeRestriction",
				"--service-cluster-ip-range=bar",
				"--service-account-key-file=" + filepath.Join(testCertsDir, "sa.pub"),
				"--service-account-signing-key-file=" + filepath.Join(testCertsDir, "sa.key"),
				"--service-account-issuer=https://kubernetes.default.svc.cluster.local",
				"--client-ca-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--tls-cert-file=" + filepath.Join(testCertsDir, "apiserver.crt"),
				"--tls-private-key-file=" + filepath.Join(testCertsDir, "apiserver.key"),
				"--kubelet-client-certificate=" + filepath.Join(testCertsDir, "apiserver-kubelet-client.crt"),
				"--kubelet-client-key=" + filepath.Join(testCertsDir, "apiserver-kubelet-client.key"),
				"--enable-bootstrap-token-auth=true",
				"--secure-port=123",
				"--allow-privileged=true",
				"--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname",
				"--proxy-client-cert-file=" + filepath.FromSlash("/var/lib/certs/front-proxy-client.crt"),
				"--proxy-client-key-file=" + filepath.FromSlash("/var/lib/certs/front-proxy-client.key"),
				"--requestheader-username-headers=X-Remote-User",
				"--requestheader-group-headers=X-Remote-Group",
				"--requestheader-extra-headers-prefix=X-Remote-Extra-",
				"--requestheader-client-ca-file=" + filepath.Join(testCertsDir, "front-proxy-ca.crt"),
				"--requestheader-allowed-names=front-proxy-client",
				"--authorization-mode=ABAC",
				"--advertise-address=1.2.3.4",
				fmt.Sprintf("--etcd-servers=https://127.0.0.1:%d", kubeadmconstants.EtcdListenClientPort),
				"--etcd-cafile=" + filepath.Join(testCertsDir, "etcd/ca.crt"),
				"--etcd-certfile=" + filepath.Join(testCertsDir, "apiserver-etcd-client.crt"),
				"--etcd-keyfile=" + filepath.Join(testCertsDir, "apiserver-etcd-client.key"),
			},
		},
		{
			name: "authorization-mode extra-args Webhook",
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking:      kubeadmapi.Networking{ServiceSubnet: "bar", DNSDomain: "cluster.local"},
				CertificatesDir: testCertsDir,
				APIServer: kubeadmapi.APIServer{
					ControlPlaneComponent: kubeadmapi.ControlPlaneComponent{
						ExtraArgs: []kubeadmapi.Arg{
							{Name: "authorization-mode", Value: strings.Join([]string{
								kubeadmconstants.ModeNode,
								kubeadmconstants.ModeRBAC,
								kubeadmconstants.ModeWebhook,
							}, ",")},
						},
					},
				},
			},
			endpoint: &kubeadmapi.APIEndpoint{BindPort: 123, AdvertiseAddress: "1.2.3.4"},
			expected: []string{
				"kube-apiserver",
				"--enable-admission-plugins=NodeRestriction",
				"--service-cluster-ip-range=bar",
				"--service-account-key-file=" + filepath.Join(testCertsDir, "sa.pub"),
				"--service-account-signing-key-file=" + filepath.Join(testCertsDir, "sa.key"),
				"--service-account-issuer=https://kubernetes.default.svc.cluster.local",
				"--client-ca-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--tls-cert-file=" + filepath.Join(testCertsDir, "apiserver.crt"),
				"--tls-private-key-file=" + filepath.Join(testCertsDir, "apiserver.key"),
				"--kubelet-client-certificate=" + filepath.Join(testCertsDir, "apiserver-kubelet-client.crt"),
				"--kubelet-client-key=" + filepath.Join(testCertsDir, "apiserver-kubelet-client.key"),
				"--enable-bootstrap-token-auth=true",
				"--secure-port=123",
				"--allow-privileged=true",
				"--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname",
				"--proxy-client-cert-file=" + filepath.FromSlash("/var/lib/certs/front-proxy-client.crt"),
				"--proxy-client-key-file=" + filepath.FromSlash("/var/lib/certs/front-proxy-client.key"),
				"--requestheader-username-headers=X-Remote-User",
				"--requestheader-group-headers=X-Remote-Group",
				"--requestheader-extra-headers-prefix=X-Remote-Extra-",
				"--requestheader-client-ca-file=" + filepath.Join(testCertsDir, "front-proxy-ca.crt"),
				"--requestheader-allowed-names=front-proxy-client",
				"--authorization-mode=Node,RBAC,Webhook",
				"--advertise-address=1.2.3.4",
				fmt.Sprintf("--etcd-servers=https://127.0.0.1:%d", kubeadmconstants.EtcdListenClientPort),
				"--etcd-cafile=" + filepath.Join(testCertsDir, "etcd/ca.crt"),
				"--etcd-certfile=" + filepath.Join(testCertsDir, "apiserver-etcd-client.crt"),
				"--etcd-keyfile=" + filepath.Join(testCertsDir, "apiserver-etcd-client.key"),
			},
		},
		{
			name: "authorization-config extra-args",
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking:      kubeadmapi.Networking{ServiceSubnet: "bar", DNSDomain: "cluster.local"},
				CertificatesDir: testCertsDir,
				APIServer: kubeadmapi.APIServer{
					ControlPlaneComponent: kubeadmapi.ControlPlaneComponent{
						ExtraArgs: []kubeadmapi.Arg{
							{Name: "authorization-config", Value: "/path/to/authorization/config/file"},
						},
					},
				},
			},
			endpoint: &kubeadmapi.APIEndpoint{BindPort: 123, AdvertiseAddress: "1.2.3.4"},
			expected: []string{
				"kube-apiserver",
				"--enable-admission-plugins=NodeRestriction",
				"--service-cluster-ip-range=bar",
				"--service-account-key-file=" + filepath.Join(testCertsDir, "sa.pub"),
				"--service-account-signing-key-file=" + filepath.Join(testCertsDir, "sa.key"),
				"--service-account-issuer=https://kubernetes.default.svc.cluster.local",
				"--client-ca-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--tls-cert-file=" + filepath.Join(testCertsDir, "apiserver.crt"),
				"--tls-private-key-file=" + filepath.Join(testCertsDir, "apiserver.key"),
				"--kubelet-client-certificate=" + filepath.Join(testCertsDir, "apiserver-kubelet-client.crt"),
				"--kubelet-client-key=" + filepath.Join(testCertsDir, "apiserver-kubelet-client.key"),
				"--enable-bootstrap-token-auth=true",
				"--secure-port=123",
				"--allow-privileged=true",
				"--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname",
				"--proxy-client-cert-file=" + filepath.FromSlash("/var/lib/certs/front-proxy-client.crt"),
				"--proxy-client-key-file=" + filepath.FromSlash("/var/lib/certs/front-proxy-client.key"),
				"--requestheader-username-headers=X-Remote-User",
				"--requestheader-group-headers=X-Remote-Group",
				"--requestheader-extra-headers-prefix=X-Remote-Extra-",
				"--requestheader-client-ca-file=" + filepath.Join(testCertsDir, "front-proxy-ca.crt"),
				"--requestheader-allowed-names=front-proxy-client",
				"--authorization-config=/path/to/authorization/config/file",
				"--advertise-address=1.2.3.4",
				fmt.Sprintf("--etcd-servers=https://127.0.0.1:%d", kubeadmconstants.EtcdListenClientPort),
				"--etcd-cafile=" + filepath.Join(testCertsDir, "etcd/ca.crt"),
				"--etcd-certfile=" + filepath.Join(testCertsDir, "apiserver-etcd-client.crt"),
				"--etcd-keyfile=" + filepath.Join(testCertsDir, "apiserver-etcd-client.key"),
			},
		},
		{
			// Note that we do not block it at this level but api server would fail to start.
			name: "authorization-config and authorization-mode extra-args",
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking:      kubeadmapi.Networking{ServiceSubnet: "bar", DNSDomain: "cluster.local"},
				CertificatesDir: testCertsDir,
				APIServer: kubeadmapi.APIServer{
					ControlPlaneComponent: kubeadmapi.ControlPlaneComponent{
						ExtraArgs: []kubeadmapi.Arg{
							{Name: "authorization-config", Value: "/path/to/authorization/config/file"},
							{Name: "authorization-mode", Value: strings.Join([]string{
								kubeadmconstants.ModeNode,
								kubeadmconstants.ModeRBAC,
								kubeadmconstants.ModeWebhook,
							}, ",")},
						},
					},
				},
			},
			endpoint: &kubeadmapi.APIEndpoint{BindPort: 123, AdvertiseAddress: "1.2.3.4"},
			expected: []string{
				"kube-apiserver",
				"--enable-admission-plugins=NodeRestriction",
				"--service-cluster-ip-range=bar",
				"--service-account-key-file=" + filepath.Join(testCertsDir, "sa.pub"),
				"--service-account-signing-key-file=" + filepath.Join(testCertsDir, "sa.key"),
				"--service-account-issuer=https://kubernetes.default.svc.cluster.local",
				"--client-ca-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--tls-cert-file=" + filepath.Join(testCertsDir, "apiserver.crt"),
				"--tls-private-key-file=" + filepath.Join(testCertsDir, "apiserver.key"),
				"--kubelet-client-certificate=" + filepath.Join(testCertsDir, "apiserver-kubelet-client.crt"),
				"--kubelet-client-key=" + filepath.Join(testCertsDir, "apiserver-kubelet-client.key"),
				"--enable-bootstrap-token-auth=true",
				"--secure-port=123",
				"--allow-privileged=true",
				"--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname",
				"--proxy-client-cert-file=" + filepath.FromSlash("/var/lib/certs/front-proxy-client.crt"),
				"--proxy-client-key-file=" + filepath.FromSlash("/var/lib/certs/front-proxy-client.key"),
				"--requestheader-username-headers=X-Remote-User",
				"--requestheader-group-headers=X-Remote-Group",
				"--requestheader-extra-headers-prefix=X-Remote-Extra-",
				"--requestheader-client-ca-file=" + filepath.Join(testCertsDir, "front-proxy-ca.crt"),
				"--requestheader-allowed-names=front-proxy-client",
				"--authorization-config=/path/to/authorization/config/file",
				"--authorization-mode=Node,RBAC,Webhook",
				"--advertise-address=1.2.3.4",
				fmt.Sprintf("--etcd-servers=https://127.0.0.1:%d", kubeadmconstants.EtcdListenClientPort),
				"--etcd-cafile=" + filepath.Join(testCertsDir, "etcd/ca.crt"),
				"--etcd-certfile=" + filepath.Join(testCertsDir, "apiserver-etcd-client.crt"),
				"--etcd-keyfile=" + filepath.Join(testCertsDir, "apiserver-etcd-client.key"),
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
// makes comparing two cmdline (with lots of arguments) easier
func removeCommon(left, right []string) []string {
	origSet := sets.New(left...)
	origSet.Delete(right...)
	return sets.List(origSet)
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
				"--kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
				"--root-ca-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--service-account-private-key-file=" + filepath.Join(testCertsDir, "sa.key"),
				"--cluster-signing-cert-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--cluster-signing-key-file=" + filepath.Join(testCertsDir, "ca.key"),
				"--use-service-account-credentials=true",
				"--controllers=*,bootstrapsigner,tokencleaner",
				"--authentication-kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
				"--authorization-kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
				"--client-ca-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--requestheader-client-ca-file=" + filepath.Join(testCertsDir, "front-proxy-ca.crt"),
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
				"--kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
				"--root-ca-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--service-account-private-key-file=" + filepath.Join(testCertsDir, "sa.key"),
				"--cluster-signing-cert-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--cluster-signing-key-file=" + filepath.Join(testCertsDir, "ca.key"),
				"--use-service-account-credentials=true",
				"--controllers=*,bootstrapsigner,tokencleaner",
				"--authentication-kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
				"--authorization-kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
				"--client-ca-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--requestheader-client-ca-file=" + filepath.Join(testCertsDir, "front-proxy-ca.crt"),
			},
		},
		{
			name: "custom cluster-cidr for " + cpVersion,
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking:        kubeadmapi.Networking{PodSubnet: "10.0.1.15/16", DNSDomain: "cluster.local"},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: cpVersion,
			},
			expected: []string{
				"kube-controller-manager",
				"--bind-address=127.0.0.1",
				"--leader-elect=true",
				"--kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
				"--root-ca-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--service-account-private-key-file=" + filepath.Join(testCertsDir, "sa.key"),
				"--cluster-signing-cert-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--cluster-signing-key-file=" + filepath.Join(testCertsDir, "ca.key"),
				"--use-service-account-credentials=true",
				"--controllers=*,bootstrapsigner,tokencleaner",
				"--authentication-kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
				"--authorization-kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
				"--client-ca-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--requestheader-client-ca-file=" + filepath.Join(testCertsDir, "front-proxy-ca.crt"),
				"--allocate-node-cidrs=true",
				"--cluster-cidr=10.0.1.15/16",
			},
		},
		{
			name: "custom service-cluster-ip-range for " + cpVersion,
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking: kubeadmapi.Networking{
					PodSubnet:     "10.0.1.15/16",
					ServiceSubnet: "172.20.0.0/24",
					DNSDomain:     "cluster.local",
				},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: cpVersion,
			},
			expected: []string{
				"kube-controller-manager",
				"--bind-address=127.0.0.1",
				"--leader-elect=true",
				"--kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
				"--root-ca-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--service-account-private-key-file=" + filepath.Join(testCertsDir, "sa.key"),
				"--cluster-signing-cert-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--cluster-signing-key-file=" + filepath.Join(testCertsDir, "ca.key"),
				"--use-service-account-credentials=true",
				"--controllers=*,bootstrapsigner,tokencleaner",
				"--authentication-kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
				"--authorization-kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
				"--client-ca-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--requestheader-client-ca-file=" + filepath.Join(testCertsDir, "front-proxy-ca.crt"),
				"--allocate-node-cidrs=true",
				"--cluster-cidr=10.0.1.15/16",
				"--service-cluster-ip-range=172.20.0.0/24",
			},
		},
		{
			name: "custom extra-args for " + cpVersion,
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking: kubeadmapi.Networking{PodSubnet: "10.0.1.15/16", DNSDomain: "cluster.local"},
				ControllerManager: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: []kubeadmapi.Arg{{Name: "node-cidr-mask-size", Value: "20"}},
				},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: cpVersion,
			},
			expected: []string{
				"kube-controller-manager",
				"--bind-address=127.0.0.1",
				"--leader-elect=true",
				"--kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
				"--root-ca-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--service-account-private-key-file=" + filepath.Join(testCertsDir, "sa.key"),
				"--cluster-signing-cert-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--cluster-signing-key-file=" + filepath.Join(testCertsDir, "ca.key"),
				"--use-service-account-credentials=true",
				"--controllers=*,bootstrapsigner,tokencleaner",
				"--authentication-kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
				"--authorization-kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
				"--client-ca-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--requestheader-client-ca-file=" + filepath.Join(testCertsDir, "front-proxy-ca.crt"),
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
					DNSDomain:     "cluster.local",
				},

				CertificatesDir:   testCertsDir,
				KubernetesVersion: cpVersion,
			},
			expected: []string{
				"kube-controller-manager",
				"--bind-address=127.0.0.1",
				"--leader-elect=true",
				"--kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
				"--root-ca-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--service-account-private-key-file=" + filepath.Join(testCertsDir, "sa.key"),
				"--cluster-signing-cert-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--cluster-signing-key-file=" + filepath.Join(testCertsDir, "ca.key"),
				"--use-service-account-credentials=true",
				"--controllers=*,bootstrapsigner,tokencleaner",
				"--authentication-kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
				"--authorization-kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
				"--client-ca-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--requestheader-client-ca-file=" + filepath.Join(testCertsDir, "front-proxy-ca.crt"),
				"--allocate-node-cidrs=true",
				"--cluster-cidr=2001:db8::/64",
				"--service-cluster-ip-range=fd03::/112",
			},
		},
		{
			name: "IPv6 networking custom extra-args for " + cpVersion,
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking: kubeadmapi.Networking{
					PodSubnet:     "2001:db8::/64",
					ServiceSubnet: "fd03::/112",
					DNSDomain:     "cluster.local",
				},
				ControllerManager: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: []kubeadmapi.Arg{{Name: "allocate-node-cidrs", Value: "false"}},
				},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: cpVersion,
			},
			expected: []string{
				"kube-controller-manager",
				"--bind-address=127.0.0.1",
				"--leader-elect=true",
				"--kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
				"--root-ca-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--service-account-private-key-file=" + filepath.Join(testCertsDir, "sa.key"),
				"--cluster-signing-cert-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--cluster-signing-key-file=" + filepath.Join(testCertsDir, "ca.key"),
				"--use-service-account-credentials=true",
				"--controllers=*,bootstrapsigner,tokencleaner",
				"--authentication-kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
				"--authorization-kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
				"--client-ca-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--requestheader-client-ca-file=" + filepath.Join(testCertsDir, "front-proxy-ca.crt"),
				"--allocate-node-cidrs=false",
				"--cluster-cidr=2001:db8::/64",
				"--service-cluster-ip-range=fd03::/112",
			},
		},
		{
			name: "dual-stack networking for " + cpVersion,
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking: kubeadmapi.Networking{
					PodSubnet:     "2001:db8::/64,10.1.0.0/16",
					ServiceSubnet: "fd03::/112,192.168.0.0/16",
					DNSDomain:     "cluster.local",
				},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: cpVersion,
			},
			expected: []string{
				"kube-controller-manager",
				"--bind-address=127.0.0.1",
				"--leader-elect=true",
				"--kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
				"--root-ca-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--service-account-private-key-file=" + filepath.Join(testCertsDir, "sa.key"),
				"--cluster-signing-cert-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--cluster-signing-key-file=" + filepath.Join(testCertsDir, "ca.key"),
				"--use-service-account-credentials=true",
				"--controllers=*,bootstrapsigner,tokencleaner",
				"--authentication-kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
				"--authorization-kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
				"--client-ca-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--requestheader-client-ca-file=" + filepath.Join(testCertsDir, "front-proxy-ca.crt"),
				"--allocate-node-cidrs=true",
				"--cluster-cidr=2001:db8::/64,10.1.0.0/16",
				"--service-cluster-ip-range=fd03::/112,192.168.0.0/16",
			},
		},
		{
			name: "dual-stack networking custom extra-args for " + cpVersion,
			cfg: &kubeadmapi.ClusterConfiguration{
				Networking: kubeadmapi.Networking{
					PodSubnet: "10.0.1.15/16,2001:db8::/64",
					DNSDomain: "cluster.local",
				},
				ControllerManager: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: []kubeadmapi.Arg{
						{Name: "node-cidr-mask-size-ipv4", Value: "20"},
						{Name: "node-cidr-mask-size-ipv6", Value: "80"},
					},
				},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: cpVersion,
			},
			expected: []string{
				"kube-controller-manager",
				"--bind-address=127.0.0.1",
				"--leader-elect=true",
				"--kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
				"--root-ca-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--service-account-private-key-file=" + filepath.Join(testCertsDir, "sa.key"),
				"--cluster-signing-cert-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--cluster-signing-key-file=" + filepath.Join(testCertsDir, "ca.key"),
				"--use-service-account-credentials=true",
				"--controllers=*,bootstrapsigner,tokencleaner",
				"--authentication-kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
				"--authorization-kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
				"--client-ca-file=" + filepath.Join(testCertsDir, "ca.crt"),
				"--requestheader-client-ca-file=" + filepath.Join(testCertsDir, "front-proxy-ca.crt"),
				"--allocate-node-cidrs=true",
				"--cluster-cidr=10.0.1.15/16,2001:db8::/64",
				"--node-cidr-mask-size-ipv4=20",
				"--node-cidr-mask-size-ipv6=80",
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
					"--kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
					"--root-ca-file=" + filepath.Join(tmpdir, "ca.crt"),
					"--service-account-private-key-file=" + filepath.Join(tmpdir, "sa.key"),
					"--cluster-signing-cert-file=",
					"--cluster-signing-key-file=",
					"--use-service-account-credentials=true",
					"--controllers=*,bootstrapsigner,tokencleaner",
					"--authentication-kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
					"--authorization-kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
					"--client-ca-file=" + filepath.Join(tmpdir, "ca.crt"),
					"--requestheader-client-ca-file=" + filepath.Join(tmpdir, "front-proxy-ca.crt"),
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
					"--kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
					"--root-ca-file=" + filepath.Join(tmpdir, "ca.crt"),
					"--service-account-private-key-file=" + filepath.Join(tmpdir, "sa.key"),
					"--cluster-signing-cert-file=" + filepath.Join(tmpdir, "ca.crt"),
					"--cluster-signing-key-file=" + filepath.Join(tmpdir, "ca.key"),
					"--use-service-account-credentials=true",
					"--controllers=*,bootstrapsigner,tokencleaner",
					"--authentication-kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
					"--authorization-kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "controller-manager.conf"),
					"--client-ca-file=" + filepath.Join(tmpdir, "ca.crt"),
					"--requestheader-client-ca-file=" + filepath.Join(tmpdir, "front-proxy-ca.crt"),
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			pkiutiltesting.Reset()

			// Create temp folder for the test case
			tmpdir := t.TempDir()
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
				"--kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "scheduler.conf"),
				"--authentication-kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "scheduler.conf"),
				"--authorization-kubeconfig=" + filepath.Join(kubeadmconstants.KubernetesDir, "scheduler.conf"),
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
