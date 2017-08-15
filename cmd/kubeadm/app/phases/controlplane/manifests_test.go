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
	"testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/pkg/util/version"

	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
)

const (
	testCertsDir = "/var/lib/certs"
	etcdDataDir  = "/var/lib/etcd"
)

func TestGetStaticPodSpecs(t *testing.T) {

	// Creates a Master Configuration
	cfg := &kubeadmapi.MasterConfiguration{
		KubernetesVersion: "v1.7.0",
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
				t.Errorf("getKubeConfigSpecs spec for %s contains pod %s, expectes %s", assertion.staticPodName, spec.Spec.Containers[0].Name, assertion.staticPodName)
			}

		} else {
			t.Errorf("getStaticPodSpecs didn't create spec for %s ", assertion.staticPodName)
		}
	}
}

func TestCreateStaticPodFilesAndWrappers(t *testing.T) {

	var tests = []struct {
		createStaticPodFunction func(outDir string, cfg *kubeadmapi.MasterConfiguration) error
		expectedFiles           []string
	}{
		{ // CreateInitStaticPodManifestFiles
			createStaticPodFunction: CreateInitStaticPodManifestFiles,
			expectedFiles:           []string{kubeadmconstants.KubeAPIServer, kubeadmconstants.KubeControllerManager, kubeadmconstants.KubeScheduler},
		},
		{ // CreateAPIServerStaticPodManifestFile
			createStaticPodFunction: CreateAPIServerStaticPodManifestFile,
			expectedFiles:           []string{kubeadmconstants.KubeAPIServer},
		},
		{ // CreateControllerManagerStaticPodManifestFile
			createStaticPodFunction: CreateControllerManagerStaticPodManifestFile,
			expectedFiles:           []string{kubeadmconstants.KubeControllerManager},
		},
		{ // CreateSchedulerStaticPodManifestFile
			createStaticPodFunction: CreateSchedulerStaticPodManifestFile,
			expectedFiles:           []string{kubeadmconstants.KubeScheduler},
		},
	}

	for _, test := range tests {

		// Create temp folder for the test case
		tmpdir := testutil.SetupTempDir(t)
		defer os.RemoveAll(tmpdir)

		// Creates a Master Configuration
		cfg := &kubeadmapi.MasterConfiguration{
			KubernetesVersion: "v1.7.0",
		}

		// Execute createStaticPodFunction
		manifestPath := filepath.Join(tmpdir, kubeadmconstants.ManifestsSubDirName)
		err := test.createStaticPodFunction(manifestPath, cfg)
		if err != nil {
			t.Errorf("Error executing createStaticPodFunction: %v", err)
			continue
		}

		// Assert expected files are there
		testutil.AssertFilesCount(t, manifestPath, len(test.expectedFiles))

		for _, fileName := range test.expectedFiles {
			testutil.AssertFileExists(t, manifestPath, fileName+".yaml")
		}
	}
}

func TestGetAPIServerCommand(t *testing.T) {
	var tests = []struct {
		cfg      *kubeadmapi.MasterConfiguration
		expected []string
	}{
		{
			cfg: &kubeadmapi.MasterConfiguration{
				API:               kubeadmapi.API{BindPort: 123, AdvertiseAddress: "1.2.3.4"},
				Networking:        kubeadmapi.Networking{ServiceSubnet: "bar"},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.7.0",
			},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--admission-control=Initializers,NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,ResourceQuota",
				"--service-cluster-ip-range=bar",
				"--service-account-key-file=" + testCertsDir + "/sa.pub",
				"--client-ca-file=" + testCertsDir + "/ca.crt",
				"--tls-cert-file=" + testCertsDir + "/apiserver.crt",
				"--tls-private-key-file=" + testCertsDir + "/apiserver.key",
				"--kubelet-client-certificate=" + testCertsDir + "/apiserver-kubelet-client.crt",
				"--kubelet-client-key=" + testCertsDir + "/apiserver-kubelet-client.key",
				"--secure-port=123",
				"--allow-privileged=true",
				"--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname",
				"--experimental-bootstrap-token-auth=true",
				"--proxy-client-cert-file=/var/lib/certs/front-proxy-client.crt",
				"--proxy-client-key-file=/var/lib/certs/front-proxy-client.key",
				"--requestheader-username-headers=X-Remote-User",
				"--requestheader-group-headers=X-Remote-Group",
				"--requestheader-extra-headers-prefix=X-Remote-Extra-",
				"--requestheader-client-ca-file=" + testCertsDir + "/front-proxy-ca.crt",
				"--requestheader-allowed-names=front-proxy-client",
				"--authorization-mode=Node,RBAC",
				"--advertise-address=1.2.3.4",
				"--etcd-servers=http://127.0.0.1:2379",
			},
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				API:               kubeadmapi.API{BindPort: 123, AdvertiseAddress: "4.3.2.1"},
				Networking:        kubeadmapi.Networking{ServiceSubnet: "bar"},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.7.1",
			},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--admission-control=Initializers,NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,ResourceQuota",
				"--service-cluster-ip-range=bar",
				"--service-account-key-file=" + testCertsDir + "/sa.pub",
				"--client-ca-file=" + testCertsDir + "/ca.crt",
				"--tls-cert-file=" + testCertsDir + "/apiserver.crt",
				"--tls-private-key-file=" + testCertsDir + "/apiserver.key",
				"--kubelet-client-certificate=" + testCertsDir + "/apiserver-kubelet-client.crt",
				"--kubelet-client-key=" + testCertsDir + "/apiserver-kubelet-client.key",
				"--secure-port=123",
				"--allow-privileged=true",
				"--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname",
				"--experimental-bootstrap-token-auth=true",
				"--proxy-client-cert-file=/var/lib/certs/front-proxy-client.crt",
				"--proxy-client-key-file=/var/lib/certs/front-proxy-client.key",
				"--requestheader-username-headers=X-Remote-User",
				"--requestheader-group-headers=X-Remote-Group",
				"--requestheader-extra-headers-prefix=X-Remote-Extra-",
				"--requestheader-client-ca-file=" + testCertsDir + "/front-proxy-ca.crt",
				"--requestheader-allowed-names=front-proxy-client",
				"--authorization-mode=Node,RBAC",
				"--advertise-address=4.3.2.1",
				"--etcd-servers=http://127.0.0.1:2379",
			},
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				API:               kubeadmapi.API{BindPort: 123, AdvertiseAddress: "4.3.2.1"},
				Networking:        kubeadmapi.Networking{ServiceSubnet: "bar"},
				Etcd:              kubeadmapi.Etcd{CertFile: "fiz", KeyFile: "faz"},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.7.2",
			},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--admission-control=Initializers,NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,ResourceQuota",
				"--service-cluster-ip-range=bar",
				"--service-account-key-file=" + testCertsDir + "/sa.pub",
				"--client-ca-file=" + testCertsDir + "/ca.crt",
				"--tls-cert-file=" + testCertsDir + "/apiserver.crt",
				"--tls-private-key-file=" + testCertsDir + "/apiserver.key",
				"--kubelet-client-certificate=" + testCertsDir + "/apiserver-kubelet-client.crt",
				"--kubelet-client-key=" + testCertsDir + "/apiserver-kubelet-client.key",
				"--secure-port=123",
				"--allow-privileged=true",
				"--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname",
				"--experimental-bootstrap-token-auth=true",
				"--proxy-client-cert-file=/var/lib/certs/front-proxy-client.crt",
				"--proxy-client-key-file=/var/lib/certs/front-proxy-client.key",
				"--requestheader-username-headers=X-Remote-User",
				"--requestheader-group-headers=X-Remote-Group",
				"--requestheader-extra-headers-prefix=X-Remote-Extra-",
				"--requestheader-client-ca-file=" + testCertsDir + "/front-proxy-ca.crt",
				"--requestheader-allowed-names=front-proxy-client",
				"--authorization-mode=Node,RBAC",
				"--advertise-address=4.3.2.1",
				"--etcd-servers=http://127.0.0.1:2379",
				"--etcd-certfile=fiz",
				"--etcd-keyfile=faz",
			},
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				API:               kubeadmapi.API{BindPort: 123, AdvertiseAddress: "4.3.2.1"},
				Networking:        kubeadmapi.Networking{ServiceSubnet: "bar"},
				Etcd:              kubeadmapi.Etcd{CertFile: "fiz", KeyFile: "faz"},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.7.3",
			},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--admission-control=Initializers,NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,ResourceQuota",
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
				"--experimental-bootstrap-token-auth=true",
				"--proxy-client-cert-file=/var/lib/certs/front-proxy-client.crt",
				"--proxy-client-key-file=/var/lib/certs/front-proxy-client.key",
				"--requestheader-username-headers=X-Remote-User",
				"--requestheader-group-headers=X-Remote-Group",
				"--requestheader-extra-headers-prefix=X-Remote-Extra-",
				"--requestheader-client-ca-file=" + testCertsDir + "/front-proxy-ca.crt",
				"--requestheader-allowed-names=front-proxy-client",
				"--authorization-mode=Node,RBAC",
				"--advertise-address=4.3.2.1",
				"--etcd-servers=http://127.0.0.1:2379",
				"--etcd-certfile=fiz",
				"--etcd-keyfile=faz",
			},
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				API:               kubeadmapi.API{BindPort: 123, AdvertiseAddress: "2001:db8::1"},
				Networking:        kubeadmapi.Networking{ServiceSubnet: "bar"},
				Etcd:              kubeadmapi.Etcd{CertFile: "fiz", KeyFile: "faz"},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.7.0",
			},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--admission-control=Initializers,NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,ResourceQuota",
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
				"--experimental-bootstrap-token-auth=true",
				"--proxy-client-cert-file=/var/lib/certs/front-proxy-client.crt",
				"--proxy-client-key-file=/var/lib/certs/front-proxy-client.key",
				"--requestheader-username-headers=X-Remote-User",
				"--requestheader-group-headers=X-Remote-Group",
				"--requestheader-extra-headers-prefix=X-Remote-Extra-",
				"--requestheader-client-ca-file=" + testCertsDir + "/front-proxy-ca.crt",
				"--requestheader-allowed-names=front-proxy-client",
				"--authorization-mode=Node,RBAC",
				"--advertise-address=2001:db8::1",
				"--etcd-servers=http://127.0.0.1:2379",
				"--etcd-certfile=fiz",
				"--etcd-keyfile=faz",
			},
		},
	}

	for _, rt := range tests {
		actual := getAPIServerCommand(rt.cfg, version.MustParseSemantic(rt.cfg.KubernetesVersion))
		sort.Strings(actual)
		sort.Strings(rt.expected)
		if !reflect.DeepEqual(actual, rt.expected) {
			t.Errorf("failed getAPIServerCommand:\nexpected:\n%v\nsaw:\n%v", rt.expected, actual)
		}
	}
}

func TestGetControllerManagerCommand(t *testing.T) {
	var tests = []struct {
		cfg      *kubeadmapi.MasterConfiguration
		expected []string
	}{
		{
			cfg: &kubeadmapi.MasterConfiguration{
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.7.0",
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
			cfg: &kubeadmapi.MasterConfiguration{
				CloudProvider:     "foo",
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.7.0",
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
				"--cloud-provider=foo",
			},
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				Networking:        kubeadmapi.Networking{PodSubnet: "bar"},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.7.0",
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
				"--cluster-cidr=bar",
			},
		},
	}

	for _, rt := range tests {
		actual := getControllerManagerCommand(rt.cfg, version.MustParseSemantic(rt.cfg.KubernetesVersion))
		sort.Strings(actual)
		sort.Strings(rt.expected)
		if !reflect.DeepEqual(actual, rt.expected) {
			t.Errorf("failed getControllerManagerCommand:\nexpected:\n%v\nsaw:\n%v", rt.expected, actual)
		}
	}
}

func TestGetSchedulerCommand(t *testing.T) {
	var tests = []struct {
		cfg      *kubeadmapi.MasterConfiguration
		expected []string
	}{
		{
			cfg: &kubeadmapi.MasterConfiguration{},
			expected: []string{
				"kube-scheduler",
				"--address=127.0.0.1",
				"--leader-elect=true",
				"--kubeconfig=" + kubeadmconstants.KubernetesDir + "/scheduler.conf",
			},
		},
	}

	for _, rt := range tests {
		actual := getSchedulerCommand(rt.cfg)
		sort.Strings(actual)
		sort.Strings(rt.expected)
		if !reflect.DeepEqual(actual, rt.expected) {
			t.Errorf("failed getSchedulerCommand:\nexpected:\n%v\nsaw:\n%v", rt.expected, actual)
		}
	}
}

func TestGetAuthzParameters(t *testing.T) {
	var tests = []struct {
		authMode []string
		expected []string
	}{
		{
			authMode: []string{},
			expected: []string{
				"--authorization-mode=Node,RBAC",
			},
		},
		{
			authMode: []string{"RBAC"},
			expected: []string{
				"--authorization-mode=RBAC",
			},
		},
		{
			authMode: []string{"AlwaysAllow"},
			expected: []string{
				"--authorization-mode=AlwaysAllow",
			},
		},
		{
			authMode: []string{"AlwaysDeny"},
			expected: []string{
				"--authorization-mode=AlwaysDeny",
			},
		},
		{
			authMode: []string{"ABAC"},
			expected: []string{
				"--authorization-mode=ABAC",
				"--authorization-policy-file=/etc/kubernetes/abac_policy.json",
			},
		},
		{
			authMode: []string{"ABAC", "Webhook"},
			expected: []string{
				"--authorization-mode=ABAC,Webhook",
				"--authorization-policy-file=/etc/kubernetes/abac_policy.json",
				"--authorization-webhook-config-file=/etc/kubernetes/webhook_authz.conf",
			},
		},
		{
			authMode: []string{"ABAC", "RBAC", "Webhook"},
			expected: []string{
				"--authorization-mode=ABAC,RBAC,Webhook",
				"--authorization-policy-file=/etc/kubernetes/abac_policy.json",
				"--authorization-webhook-config-file=/etc/kubernetes/webhook_authz.conf",
			},
		},
		{
			authMode: []string{"Node", "RBAC", "Webhook", "ABAC"},
			expected: []string{
				"--authorization-mode=Node,RBAC,Webhook,ABAC",
				"--authorization-policy-file=/etc/kubernetes/abac_policy.json",
				"--authorization-webhook-config-file=/etc/kubernetes/webhook_authz.conf",
			},
		},
	}

	for _, rt := range tests {
		actual := getAuthzParameters(rt.authMode)
		sort.Strings(actual)
		sort.Strings(rt.expected)
		if !reflect.DeepEqual(actual, rt.expected) {
			t.Errorf("failed getAuthzParameters:\nexpected:\n%v\nsaw:\n%v", rt.expected, actual)
		}
	}
}
