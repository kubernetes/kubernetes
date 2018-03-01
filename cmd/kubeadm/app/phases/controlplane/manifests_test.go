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
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	"k8s.io/kubernetes/pkg/master/reconcilers"
	"k8s.io/kubernetes/pkg/util/version"

	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
	utilpointer "k8s.io/kubernetes/pkg/util/pointer"
)

const (
	testCertsDir = "/var/lib/certs"
	etcdDataDir  = "/var/lib/etcd"
)

func TestGetStaticPodSpecs(t *testing.T) {

	// Creates a Master Configuration
	cfg := &kubeadmapi.MasterConfiguration{
		KubernetesVersion: "v1.9.0",
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
			KubernetesVersion: "v1.9.0",
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

func TestCreatePrivilegedContainerForOpenStack(t *testing.T) {
	// Creates a Master Configuration with OpenStack cloud provider
	var staticPodNames = []string{
		kubeadmconstants.KubeAPIServer,
		kubeadmconstants.KubeControllerManager,
	}
	var assertions = []struct {
		cloudProvider     string
		privilegedPods    bool
		expectedPrivilege bool
	}{
		{
			cloudProvider:     "",
			expectedPrivilege: false,
		},
		{
			cloudProvider:     "aws",
			expectedPrivilege: false,
		},
		{
			cloudProvider:     "openstack",
			privilegedPods:    true,
			expectedPrivilege: true,
		},
	}

	for _, assertion := range assertions {
		cfg := &kubeadmapi.MasterConfiguration{
			KubernetesVersion: "v1.9.0",
			CloudProvider:     assertion.cloudProvider,
			PrivilegedPods:    assertion.privilegedPods,
		}

		k8sVersion, _ := version.ParseSemantic(cfg.KubernetesVersion)
		specs := GetStaticPodSpecs(cfg, k8sVersion)

		for _, podname := range staticPodNames {
			spec, _ := specs[podname]
			sc := spec.Spec.Containers[0].SecurityContext
			if assertion.expectedPrivilege == true {
				if sc == nil || sc.Privileged == nil || *sc.Privileged == false {
					t.Errorf("GetStaticPodSpecs did not enable privileged containers in %s pod for provider %s", podname, assertion.cloudProvider)
				}
			} else {
				if sc != nil && sc.Privileged != nil && *sc.Privileged == true {
					t.Errorf("GetStaticPodSpecs enabled privileged containers in %s pod for provider %s", podname, assertion.cloudProvider)
				}
			}
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
				KubernetesVersion: "v1.9.0",
			},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--admission-control=NamespaceLifecycle,LimitRanger,ServiceAccount,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,MutatingAdmissionWebhook,ValidatingAdmissionWebhook,ResourceQuota",
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
				"--etcd-servers=https://127.0.0.1:2379",
				"--etcd-cafile=" + testCertsDir + "/etcd/ca.crt",
				"--etcd-certfile=" + testCertsDir + "/apiserver-etcd-client.crt",
				"--etcd-keyfile=" + testCertsDir + "/apiserver-etcd-client.key",
			},
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				API:               kubeadmapi.API{BindPort: 123, AdvertiseAddress: "1.2.3.4"},
				Networking:        kubeadmapi.Networking{ServiceSubnet: "bar"},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.9.0-beta.0",
			},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--admission-control=NamespaceLifecycle,LimitRanger,ServiceAccount,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,MutatingAdmissionWebhook,ValidatingAdmissionWebhook,ResourceQuota",
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
				"--enable-bootstrap-token-auth=true",
				"--proxy-client-cert-file=/var/lib/certs/front-proxy-client.crt",
				"--proxy-client-key-file=/var/lib/certs/front-proxy-client.key",
				"--requestheader-username-headers=X-Remote-User",
				"--requestheader-group-headers=X-Remote-Group",
				"--requestheader-extra-headers-prefix=X-Remote-Extra-",
				"--requestheader-client-ca-file=" + testCertsDir + "/front-proxy-ca.crt",
				"--requestheader-allowed-names=front-proxy-client",
				"--authorization-mode=Node,RBAC",
				"--advertise-address=1.2.3.4",
				"--etcd-servers=https://127.0.0.1:2379",
				"--etcd-cafile=" + testCertsDir + "/etcd/ca.crt",
				"--etcd-certfile=" + testCertsDir + "/apiserver-etcd-client.crt",
				"--etcd-keyfile=" + testCertsDir + "/apiserver-etcd-client.key",
			},
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				API:               kubeadmapi.API{BindPort: 123, AdvertiseAddress: "4.3.2.1"},
				Networking:        kubeadmapi.Networking{ServiceSubnet: "bar"},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.9.1",
			},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--admission-control=NamespaceLifecycle,LimitRanger,ServiceAccount,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,MutatingAdmissionWebhook,ValidatingAdmissionWebhook,ResourceQuota",
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
				"--advertise-address=4.3.2.1",
				"--etcd-servers=https://127.0.0.1:2379",
				"--etcd-cafile=" + testCertsDir + "/etcd/ca.crt",
				"--etcd-certfile=" + testCertsDir + "/apiserver-etcd-client.crt",
				"--etcd-keyfile=" + testCertsDir + "/apiserver-etcd-client.key",
			},
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				API:               kubeadmapi.API{BindPort: 123, AdvertiseAddress: "4.3.2.1"},
				Networking:        kubeadmapi.Networking{ServiceSubnet: "bar"},
				Etcd:              kubeadmapi.Etcd{CertFile: "fiz", KeyFile: "faz"},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.9.0",
			},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--admission-control=NamespaceLifecycle,LimitRanger,ServiceAccount,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,MutatingAdmissionWebhook,ValidatingAdmissionWebhook,ResourceQuota",
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
				"--advertise-address=4.3.2.1",
				"--etcd-servers=https://127.0.0.1:2379",
				"--etcd-cafile=" + testCertsDir + "/etcd/ca.crt",
				"--etcd-certfile=" + testCertsDir + "/apiserver-etcd-client.crt",
				"--etcd-keyfile=" + testCertsDir + "/apiserver-etcd-client.key",
			},
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				API:               kubeadmapi.API{BindPort: 123, AdvertiseAddress: "4.3.2.1"},
				Networking:        kubeadmapi.Networking{ServiceSubnet: "bar"},
				Etcd:              kubeadmapi.Etcd{CertFile: "fiz", KeyFile: "faz"},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.9.3",
				AuditPolicyConfiguration: kubeadmapi.AuditPolicyConfiguration{
					Path:      "/foo/bar",
					LogDir:    "/foo/baz",
					LogMaxAge: utilpointer.Int32Ptr(10),
				}, // ignored without the feature gate
			},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--admission-control=NamespaceLifecycle,LimitRanger,ServiceAccount,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,MutatingAdmissionWebhook,ValidatingAdmissionWebhook,ResourceQuota",
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
				"--etcd-servers=https://127.0.0.1:2379",
				"--etcd-cafile=" + testCertsDir + "/etcd/ca.crt",
				"--etcd-certfile=" + testCertsDir + "/apiserver-etcd-client.crt",
				"--etcd-keyfile=" + testCertsDir + "/apiserver-etcd-client.key",
			},
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				API:               kubeadmapi.API{BindPort: 123, AdvertiseAddress: "2001:db8::1"},
				Networking:        kubeadmapi.Networking{ServiceSubnet: "bar"},
				Etcd:              kubeadmapi.Etcd{CertFile: "fiz", KeyFile: "faz"},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.9.0",
			},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--admission-control=NamespaceLifecycle,LimitRanger,ServiceAccount,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,MutatingAdmissionWebhook,ValidatingAdmissionWebhook,ResourceQuota",
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
				"--etcd-servers=https://127.0.0.1:2379",
				"--etcd-cafile=" + testCertsDir + "/etcd/ca.crt",
				"--etcd-certfile=" + testCertsDir + "/apiserver-etcd-client.crt",
				"--etcd-keyfile=" + testCertsDir + "/apiserver-etcd-client.key",
			},
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				API:               kubeadmapi.API{BindPort: 123, AdvertiseAddress: "2001:db8::1"},
				Networking:        kubeadmapi.Networking{ServiceSubnet: "bar"},
				Etcd:              kubeadmapi.Etcd{CertFile: "fiz", KeyFile: "faz"},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.9.0-beta.0",
			},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--admission-control=NamespaceLifecycle,LimitRanger,ServiceAccount,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,MutatingAdmissionWebhook,ValidatingAdmissionWebhook,ResourceQuota",
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
				"--etcd-servers=https://127.0.0.1:2379",
				"--etcd-cafile=" + testCertsDir + "/etcd/ca.crt",
				"--etcd-certfile=" + testCertsDir + "/apiserver-etcd-client.crt",
				"--etcd-keyfile=" + testCertsDir + "/apiserver-etcd-client.key",
			},
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				API:               kubeadmapi.API{BindPort: 123, AdvertiseAddress: "2001:db8::1"},
				Networking:        kubeadmapi.Networking{ServiceSubnet: "bar"},
				FeatureGates:      map[string]bool{features.HighAvailability: true},
				Etcd:              kubeadmapi.Etcd{Endpoints: []string{"https://8.6.4.1:2379", "https://8.6.4.2:2379"}, CAFile: "fuz", CertFile: "fiz", KeyFile: "faz"},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.9.0-beta.0",
			},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--admission-control=NamespaceLifecycle,LimitRanger,ServiceAccount,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,MutatingAdmissionWebhook,ValidatingAdmissionWebhook,ResourceQuota",
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
				fmt.Sprintf("--endpoint-reconciler-type=%s", reconcilers.LeaseEndpointReconcilerType),
			},
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				API:               kubeadmapi.API{BindPort: 123, AdvertiseAddress: "2001:db8::1"},
				Networking:        kubeadmapi.Networking{ServiceSubnet: "bar"},
				Etcd:              kubeadmapi.Etcd{Endpoints: []string{"http://127.0.0.1:2379", "http://127.0.0.1:2380"}},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.9.0-beta.0",
			},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--admission-control=NamespaceLifecycle,LimitRanger,ServiceAccount,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,MutatingAdmissionWebhook,ValidatingAdmissionWebhook,ResourceQuota",
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
			cfg: &kubeadmapi.MasterConfiguration{
				API:               kubeadmapi.API{BindPort: 123, AdvertiseAddress: "2001:db8::1"},
				Networking:        kubeadmapi.Networking{ServiceSubnet: "bar"},
				Etcd:              kubeadmapi.Etcd{CAFile: "fuz"},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.9.0-beta.0",
			},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--admission-control=NamespaceLifecycle,LimitRanger,ServiceAccount,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,MutatingAdmissionWebhook,ValidatingAdmissionWebhook,ResourceQuota",
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
				"--etcd-servers=https://127.0.0.1:2379",
				"--etcd-cafile=" + testCertsDir + "/etcd/ca.crt",
				"--etcd-certfile=" + testCertsDir + "/apiserver-etcd-client.crt",
				"--etcd-keyfile=" + testCertsDir + "/apiserver-etcd-client.key",
			},
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				API:               kubeadmapi.API{BindPort: 123, AdvertiseAddress: "2001:db8::1"},
				Networking:        kubeadmapi.Networking{ServiceSubnet: "bar"},
				FeatureGates:      map[string]bool{features.HighAvailability: true, features.Auditing: true},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.9.0-beta.0",
				AuditPolicyConfiguration: kubeadmapi.AuditPolicyConfiguration{
					LogMaxAge: utilpointer.Int32Ptr(0),
				},
			},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--admission-control=NamespaceLifecycle,LimitRanger,ServiceAccount,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,MutatingAdmissionWebhook,ValidatingAdmissionWebhook,ResourceQuota",
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
				"--etcd-servers=https://127.0.0.1:2379",
				"--etcd-cafile=" + testCertsDir + "/etcd/ca.crt",
				"--etcd-certfile=" + testCertsDir + "/apiserver-etcd-client.crt",
				"--etcd-keyfile=" + testCertsDir + "/apiserver-etcd-client.key",
				fmt.Sprintf("--endpoint-reconciler-type=%s", reconcilers.LeaseEndpointReconcilerType),
				"--audit-policy-file=/etc/kubernetes/audit/audit.yaml",
				"--audit-log-path=/var/log/kubernetes/audit/audit.log",
				"--audit-log-maxage=0",
			},
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				API:               kubeadmapi.API{BindPort: 123, AdvertiseAddress: "1.2.3.4"},
				Networking:        kubeadmapi.Networking{ServiceSubnet: "bar"},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.9.0-beta.0",
				CloudProvider:     "gce",
			},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--admission-control=NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,MutatingAdmissionWebhook,ValidatingAdmissionWebhook,ResourceQuota",
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
				"--etcd-servers=https://127.0.0.1:2379",
				"--etcd-cafile=" + testCertsDir + "/etcd/ca.crt",
				"--etcd-certfile=" + testCertsDir + "/apiserver-etcd-client.crt",
				"--etcd-keyfile=" + testCertsDir + "/apiserver-etcd-client.key",
				"--cloud-provider=gce",
			},
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				API:               kubeadmapi.API{BindPort: 123, AdvertiseAddress: "1.2.3.4"},
				Networking:        kubeadmapi.Networking{ServiceSubnet: "bar"},
				CertificatesDir:   testCertsDir,
				KubernetesVersion: "v1.9.0-beta.0",
				CloudProvider:     "aws",
			},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--admission-control=NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,MutatingAdmissionWebhook,ValidatingAdmissionWebhook,ResourceQuota",
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
				"--etcd-servers=https://127.0.0.1:2379",
				"--etcd-cafile=" + testCertsDir + "/etcd/ca.crt",
				"--etcd-certfile=" + testCertsDir + "/apiserver-etcd-client.crt",
				"--etcd-keyfile=" + testCertsDir + "/apiserver-etcd-client.key",
				"--cloud-provider=aws",
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
				Networking:        kubeadmapi.Networking{PodSubnet: "10.0.1.15/16"},
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
				"--cluster-cidr=10.0.1.15/16",
				"--node-cidr-mask-size=24",
			},
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				Networking:        kubeadmapi.Networking{PodSubnet: "2001:db8::/64"},
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
				"--cluster-cidr=2001:db8::/64",
				"--node-cidr-mask-size=80",
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
			name:           "V6: Largest subnet currently supported",
			podSubnet:      "2001:db8::/66",
			expectedPrefix: "80",
		},
		{
			name:           "V6: For /64 pod net, use /80",
			podSubnet:      "2001:db8::/64",
			expectedPrefix: "80",
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
		cfg             *kubeadmapi.MasterConfiguration
		caKeyPresent    bool
		expectedArgFunc func(dir string) []string
	}{
		{
			cfg: &kubeadmapi.MasterConfiguration{
				KubernetesVersion: "v1.7.0",
				API:               kubeadmapi.API{AdvertiseAddress: "1.2.3.4"},
				Networking:        kubeadmapi.Networking{ServiceSubnet: "10.96.0.0/12", DNSDomain: "cluster.local"},
				NodeName:          "valid-hostname",
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
			cfg: &kubeadmapi.MasterConfiguration{
				KubernetesVersion: "v1.7.0",
				API:               kubeadmapi.API{AdvertiseAddress: "1.2.3.4"},
				Networking:        kubeadmapi.Networking{ServiceSubnet: "10.96.0.0/12", DNSDomain: "cluster.local"},
				NodeName:          "valid-hostname",
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

		// delete ca.key if test.caKeyPresent is false
		if !test.caKeyPresent {
			if err := os.Remove(filepath.Join(test.cfg.CertificatesDir, "ca.key")); err != nil {
				t.Errorf("failed removing ca.key: %v", err)
			}
		}

		actual := getControllerManagerCommand(test.cfg, version.MustParseSemantic(test.cfg.KubernetesVersion))
		expected := test.expectedArgFunc(tmpdir)
		sort.Strings(actual)
		sort.Strings(expected)
		if !reflect.DeepEqual(actual, expected) {
			t.Errorf("failed getControllerManagerCommand:\nexpected:\n%v\nsaw:\n%v", expected, actual)
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
