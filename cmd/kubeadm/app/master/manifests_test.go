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

package master

import (
	"fmt"
	"io/ioutil"
	"os"
	"reflect"
	"sort"
	"testing"

	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/yaml"
	api "k8s.io/client-go/pkg/api/v1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

const testCertsDir = "/var/lib/certs"

func TestWriteStaticPodManifests(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	// set up tmp GlobalEnvParams values for testing
	oldEnv := kubeadmapi.GlobalEnvParams
	kubeadmapi.GlobalEnvParams.KubernetesDir = fmt.Sprintf("%s/etc/kubernetes", tmpdir)
	defer func() { kubeadmapi.GlobalEnvParams = oldEnv }()

	var tests = []struct {
		cfg                  *kubeadmapi.MasterConfiguration
		expected             bool
		expectedAPIProbePort int32
	}{
		{
			cfg:      &kubeadmapi.MasterConfiguration{},
			expected: true,
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				API: kubeadmapi.API{
					BindPort: 443,
				},
			},
			expected:             true,
			expectedAPIProbePort: 443,
		},
	}
	for _, rt := range tests {
		actual := WriteStaticPodManifests(rt.cfg)
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed WriteStaticPodManifests with an error:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(actual == nil),
			)
			continue
		}

		if rt.expectedAPIProbePort != 0 {
			manifest, err := os.Open(fmt.Sprintf("%s/manifests/kube-apiserver.yaml", kubeadmapi.GlobalEnvParams.KubernetesDir))
			if err != nil {
				t.Error("WriteStaticPodManifests: error opening manifests/kube-apiserver.yaml")
				continue
			}

			var pod api.Pod
			d := yaml.NewYAMLOrJSONDecoder(manifest, 4096)
			if err := d.Decode(&pod); err != nil {
				t.Error("WriteStaticPodManifests: error decoding manifests/kube-apiserver.yaml into Pod")
				continue
			}

			// Lots of individual checks as we traverse pointers so we don't panic dereferencing a nil on failure
			containers := pod.Spec.Containers
			if containers == nil || len(containers) == 0 {
				t.Error("WriteStaticPodManifests: wrote an apiserver manifest without any containers")
				continue
			}

			probe := containers[0].LivenessProbe
			if probe == nil {
				t.Error("WriteStaticPodManifests: wrote an apiserver manifest without a liveness probe")
				continue
			}

			httpGET := probe.Handler.HTTPGet
			if httpGET == nil {
				t.Error("WriteStaticPodManifests: wrote an apiserver manifest without an HTTP liveness probe")
				continue
			}

			port := httpGET.Port.IntVal
			if rt.expectedAPIProbePort != port {
				t.Errorf("WriteStaticPodManifests: apiserver pod liveness probe port was: %v, wanted %v", port, rt.expectedAPIProbePort)
			}
		}
	}
}

func TestEtcdVolume(t *testing.T) {
	var tests = []struct {
		cfg      *kubeadmapi.MasterConfiguration
		expected api.Volume
	}{
		{
			cfg: &kubeadmapi.MasterConfiguration{},
			expected: api.Volume{
				Name: "etcd",
				VolumeSource: api.VolumeSource{
					HostPath: &api.HostPathVolumeSource{
						Path: kubeadmapi.GlobalEnvParams.HostEtcdPath},
				}},
		},
	}

	for _, rt := range tests {
		actual := etcdVolume(rt.cfg)
		if actual.Name != rt.expected.Name {
			t.Errorf(
				"failed etcdVolume:\n\texpected: %s\n\t  actual: %s",
				rt.expected.Name,
				actual.Name,
			)
		}
		if actual.VolumeSource.HostPath.Path != rt.expected.VolumeSource.HostPath.Path {
			t.Errorf(
				"failed etcdVolume:\n\texpected: %s\n\t  actual: %s",
				rt.expected.VolumeSource.HostPath.Path,
				actual.VolumeSource.HostPath.Path,
			)
		}
	}
}

func TestEtcdVolumeMount(t *testing.T) {
	var tests = []struct {
		expected api.VolumeMount
	}{
		{
			expected: api.VolumeMount{
				Name:      "etcd",
				MountPath: "/var/lib/etcd",
			},
		},
	}

	for _, rt := range tests {
		actual := etcdVolumeMount()
		if actual.Name != rt.expected.Name {
			t.Errorf(
				"failed etcdVolumeMount:\n\texpected: %s\n\t  actual: %s",
				rt.expected.Name,
				actual.Name,
			)
		}
		if actual.MountPath != rt.expected.MountPath {
			t.Errorf(
				"failed etcdVolumeMount:\n\texpected: %s\n\t  actual: %s",
				rt.expected.MountPath,
				actual.MountPath,
			)
		}
	}
}

func TestCertsVolume(t *testing.T) {
	var tests = []struct {
		cfg      *kubeadmapi.MasterConfiguration
		expected api.Volume
	}{
		{
			cfg: &kubeadmapi.MasterConfiguration{},
			expected: api.Volume{
				Name: "certs",
				VolumeSource: api.VolumeSource{
					HostPath: &api.HostPathVolumeSource{
						Path: "/etc/ssl/certs"},
				}},
		},
	}

	for _, rt := range tests {
		actual := certsVolume(rt.cfg)
		if actual.Name != rt.expected.Name {
			t.Errorf(
				"failed certsVolume:\n\texpected: %s\n\t  actual: %s",
				rt.expected.Name,
				actual.Name,
			)
		}
		if actual.VolumeSource.HostPath.Path != rt.expected.VolumeSource.HostPath.Path {
			t.Errorf(
				"failed certsVolume:\n\texpected: %s\n\t  actual: %s",
				rt.expected.VolumeSource.HostPath.Path,
				actual.VolumeSource.HostPath.Path,
			)
		}
	}
}

func TestCertsVolumeMount(t *testing.T) {
	var tests = []struct {
		expected api.VolumeMount
	}{
		{
			expected: api.VolumeMount{
				Name:      "certs",
				MountPath: "/etc/ssl/certs",
			},
		},
	}

	for _, rt := range tests {
		actual := certsVolumeMount()
		if actual.Name != rt.expected.Name {
			t.Errorf(
				"failed certsVolumeMount:\n\texpected: %s\n\t  actual: %s",
				rt.expected.Name,
				actual.Name,
			)
		}
		if actual.MountPath != rt.expected.MountPath {
			t.Errorf(
				"failed certsVolumeMount:\n\texpected: %s\n\t  actual: %s",
				rt.expected.MountPath,
				actual.MountPath,
			)
		}
	}
}

func TestK8sVolume(t *testing.T) {
	var tests = []struct {
		cfg      *kubeadmapi.MasterConfiguration
		expected api.Volume
	}{
		{
			cfg: &kubeadmapi.MasterConfiguration{},
			expected: api.Volume{
				Name: "k8s",
				VolumeSource: api.VolumeSource{
					HostPath: &api.HostPathVolumeSource{
						Path: kubeadmapi.GlobalEnvParams.KubernetesDir},
				}},
		},
	}

	for _, rt := range tests {
		actual := k8sVolume(rt.cfg)
		if actual.Name != rt.expected.Name {
			t.Errorf(
				"failed k8sVolume:\n\texpected: %s\n\t  actual: %s",
				rt.expected.Name,
				actual.Name,
			)
		}
		if actual.VolumeSource.HostPath.Path != rt.expected.VolumeSource.HostPath.Path {
			t.Errorf(
				"failed k8sVolume:\n\texpected: %s\n\t  actual: %s",
				rt.expected.VolumeSource.HostPath.Path,
				actual.VolumeSource.HostPath.Path,
			)
		}
	}
}

func TestK8sVolumeMount(t *testing.T) {
	var tests = []struct {
		expected api.VolumeMount
	}{
		{
			expected: api.VolumeMount{
				Name:      "k8s",
				MountPath: "/etc/kubernetes/",
				ReadOnly:  true,
			},
		},
	}

	for _, rt := range tests {
		actual := k8sVolumeMount()
		if actual.Name != rt.expected.Name {
			t.Errorf(
				"failed k8sVolumeMount:\n\texpected: %s\n\t  actual: %s",
				rt.expected.Name,
				actual.Name,
			)
		}
		if actual.MountPath != rt.expected.MountPath {
			t.Errorf(
				"failed k8sVolumeMount:\n\texpected: %s\n\t  actual: %s",
				rt.expected.MountPath,
				actual.MountPath,
			)
		}
		if actual.ReadOnly != rt.expected.ReadOnly {
			t.Errorf(
				"failed k8sVolumeMount:\n\texpected: %t\n\t  actual: %t",
				rt.expected.ReadOnly,
				actual.ReadOnly,
			)
		}
	}
}

func TestComponentResources(t *testing.T) {
	a := componentResources("250m")
	if a.Requests == nil {
		t.Errorf(
			"failed componentResources, return value was nil",
		)
	}
}

func TestComponentProbe(t *testing.T) {
	var tests = []struct {
		port   int
		path   string
		scheme api.URIScheme
	}{
		{
			port:   1,
			path:   "foo",
			scheme: api.URISchemeHTTP,
		},
		{
			port:   2,
			path:   "bar",
			scheme: api.URISchemeHTTPS,
		},
	}
	for _, rt := range tests {
		actual := componentProbe(rt.port, rt.path, rt.scheme)
		if actual.Handler.HTTPGet.Port != intstr.FromInt(rt.port) {
			t.Errorf(
				"failed componentProbe:\n\texpected: %v\n\t  actual: %v",
				rt.port,
				actual.Handler.HTTPGet.Port,
			)
		}
		if actual.Handler.HTTPGet.Path != rt.path {
			t.Errorf(
				"failed componentProbe:\n\texpected: %s\n\t  actual: %s",
				rt.path,
				actual.Handler.HTTPGet.Path,
			)
		}
		if actual.Handler.HTTPGet.Scheme != rt.scheme {
			t.Errorf(
				"failed componentProbe:\n\texpected: %v\n\t  actual: %v",
				rt.scheme,
				actual.Handler.HTTPGet.Scheme,
			)
		}
	}
}

func TestComponentPod(t *testing.T) {
	var tests = []struct {
		n string
	}{
		{
			n: "foo",
		},
	}

	for _, rt := range tests {
		c := api.Container{Name: rt.n}
		v := api.Volume{}
		actual := componentPod(c, v)
		if actual.ObjectMeta.Name != rt.n {
			t.Errorf(
				"failed componentPod:\n\texpected: %s\n\t  actual: %s",
				rt.n,
				actual.ObjectMeta.Name,
			)
		}
	}
}

func TestGetComponentBaseCommand(t *testing.T) {
	var tests = []struct {
		c        string
		expected []string
	}{
		{
			c:        "foo",
			expected: []string{"kube-foo", "--v=2"},
		},
		{
			c:        "bar",
			expected: []string{"kube-bar", "--v=2"},
		},
	}

	for _, rt := range tests {
		actual := getComponentBaseCommand(rt.c)
		for i := range actual {
			if actual[i] != rt.expected[i] {
				t.Errorf(
					"failed getComponentBaseCommand:\n\texpected: %s\n\t  actual: %s",
					rt.expected[i],
					actual[i],
				)
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
				API:             kubeadm.API{BindPort: 123, AdvertiseAddress: "1.2.3.4"},
				Networking:      kubeadm.Networking{ServiceSubnet: "bar"},
				CertificatesDir: testCertsDir,
			},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--admission-control=NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,ResourceQuota,DefaultTolerationSeconds",
				"--service-cluster-ip-range=bar",
				"--service-account-key-file=" + testCertsDir + "/sa.pub",
				"--client-ca-file=" + testCertsDir + "/ca.crt",
				"--tls-cert-file=" + testCertsDir + "/apiserver.crt",
				"--tls-private-key-file=" + testCertsDir + "/apiserver.key",
				"--kubelet-client-certificate=" + testCertsDir + "/apiserver-kubelet-client.crt",
				"--kubelet-client-key=" + testCertsDir + "/apiserver-kubelet-client.key",
				fmt.Sprintf("--secure-port=%d", 123),
				"--allow-privileged=true",
				"--storage-backend=etcd3",
				"--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname",
				"--experimental-bootstrap-token-auth=true",
				"--proxy-client-cert-file=/var/lib/certs/front-proxy-client.crt",
				"--proxy-client-key-file=/var/lib/certs/front-proxy-client.key",
				"--requestheader-username-headers=X-Remote-User",
				"--requestheader-group-headers=X-Remote-Group",
				"--requestheader-extra-headers-prefix=X-Remote-Extra-",
				"--requestheader-client-ca-file=" + testCertsDir + "/front-proxy-ca.crt",
				"--requestheader-allowed-names=front-proxy-client",
				"--authorization-mode=RBAC",
				"--advertise-address=1.2.3.4",
				"--etcd-servers=http://127.0.0.1:2379",
			},
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				API:             kubeadm.API{BindPort: 123, AdvertiseAddress: "4.3.2.1"},
				Networking:      kubeadm.Networking{ServiceSubnet: "bar"},
				CertificatesDir: testCertsDir,
			},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--admission-control=NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,ResourceQuota,DefaultTolerationSeconds",
				"--service-cluster-ip-range=bar",
				"--service-account-key-file=" + testCertsDir + "/sa.pub",
				"--client-ca-file=" + testCertsDir + "/ca.crt",
				"--tls-cert-file=" + testCertsDir + "/apiserver.crt",
				"--tls-private-key-file=" + testCertsDir + "/apiserver.key",
				"--kubelet-client-certificate=" + testCertsDir + "/apiserver-kubelet-client.crt",
				"--kubelet-client-key=" + testCertsDir + "/apiserver-kubelet-client.key",
				fmt.Sprintf("--secure-port=%d", 123),
				"--allow-privileged=true",
				"--storage-backend=etcd3",
				"--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname",
				"--experimental-bootstrap-token-auth=true",
				"--proxy-client-cert-file=/var/lib/certs/front-proxy-client.crt",
				"--proxy-client-key-file=/var/lib/certs/front-proxy-client.key",
				"--requestheader-username-headers=X-Remote-User",
				"--requestheader-group-headers=X-Remote-Group",
				"--requestheader-extra-headers-prefix=X-Remote-Extra-",
				"--requestheader-client-ca-file=" + testCertsDir + "/front-proxy-ca.crt",
				"--requestheader-allowed-names=front-proxy-client",
				"--authorization-mode=RBAC",
				"--advertise-address=4.3.2.1",
				"--etcd-servers=http://127.0.0.1:2379",
			},
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				API:             kubeadm.API{BindPort: 123, AdvertiseAddress: "4.3.2.1"},
				Networking:      kubeadm.Networking{ServiceSubnet: "bar"},
				Etcd:            kubeadm.Etcd{CertFile: "fiz", KeyFile: "faz"},
				CertificatesDir: testCertsDir,
			},
			expected: []string{
				"kube-apiserver",
				"--insecure-port=0",
				"--admission-control=NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,ResourceQuota,DefaultTolerationSeconds",
				"--service-cluster-ip-range=bar",
				"--service-account-key-file=" + testCertsDir + "/sa.pub",
				"--client-ca-file=" + testCertsDir + "/ca.crt",
				"--tls-cert-file=" + testCertsDir + "/apiserver.crt",
				"--tls-private-key-file=" + testCertsDir + "/apiserver.key",
				"--kubelet-client-certificate=" + testCertsDir + "/apiserver-kubelet-client.crt",
				"--kubelet-client-key=" + testCertsDir + "/apiserver-kubelet-client.key",
				fmt.Sprintf("--secure-port=%d", 123),
				"--allow-privileged=true",
				"--storage-backend=etcd3",
				"--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname",
				"--experimental-bootstrap-token-auth=true",
				"--proxy-client-cert-file=/var/lib/certs/front-proxy-client.crt",
				"--proxy-client-key-file=/var/lib/certs/front-proxy-client.key",
				"--requestheader-username-headers=X-Remote-User",
				"--requestheader-group-headers=X-Remote-Group",
				"--requestheader-extra-headers-prefix=X-Remote-Extra-",
				"--requestheader-client-ca-file=" + testCertsDir + "/front-proxy-ca.crt",
				"--requestheader-allowed-names=front-proxy-client",
				"--authorization-mode=RBAC",
				"--advertise-address=4.3.2.1",
				"--etcd-servers=http://127.0.0.1:2379",
				"--etcd-certfile=fiz",
				"--etcd-keyfile=faz",
			},
		},
	}

	for _, rt := range tests {
		actual := getAPIServerCommand(rt.cfg, false)
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
				CertificatesDir: testCertsDir,
			},
			expected: []string{
				"kube-controller-manager",
				"--address=127.0.0.1",
				"--leader-elect=true",
				"--kubeconfig=" + kubeadmapi.GlobalEnvParams.KubernetesDir + "/controller-manager.conf",
				"--root-ca-file=" + testCertsDir + "/ca.crt",
				"--service-account-private-key-file=" + testCertsDir + "/sa.key",
				"--cluster-signing-cert-file=" + testCertsDir + "/ca.crt",
				"--cluster-signing-key-file=" + testCertsDir + "/ca.key",
				"--insecure-experimental-approve-all-kubelet-csrs-for-group=system:bootstrappers",
				"--use-service-account-credentials=true",
				"--controllers=*,bootstrapsigner,tokencleaner",
			},
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				CloudProvider:   "foo",
				CertificatesDir: testCertsDir,
			},
			expected: []string{
				"kube-controller-manager",
				"--address=127.0.0.1",
				"--leader-elect=true",
				"--kubeconfig=" + kubeadmapi.GlobalEnvParams.KubernetesDir + "/controller-manager.conf",
				"--root-ca-file=" + testCertsDir + "/ca.crt",
				"--service-account-private-key-file=" + testCertsDir + "/sa.key",
				"--cluster-signing-cert-file=" + testCertsDir + "/ca.crt",
				"--cluster-signing-key-file=" + testCertsDir + "/ca.key",
				"--insecure-experimental-approve-all-kubelet-csrs-for-group=system:bootstrappers",
				"--use-service-account-credentials=true",
				"--controllers=*,bootstrapsigner,tokencleaner",
				"--cloud-provider=foo",
			},
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				Networking:      kubeadm.Networking{PodSubnet: "bar"},
				CertificatesDir: testCertsDir,
			},
			expected: []string{
				"kube-controller-manager",
				"--address=127.0.0.1",
				"--leader-elect=true",
				"--kubeconfig=" + kubeadmapi.GlobalEnvParams.KubernetesDir + "/controller-manager.conf",
				"--root-ca-file=" + testCertsDir + "/ca.crt",
				"--service-account-private-key-file=" + testCertsDir + "/sa.key",
				"--cluster-signing-cert-file=" + testCertsDir + "/ca.crt",
				"--cluster-signing-key-file=" + testCertsDir + "/ca.key",
				"--insecure-experimental-approve-all-kubelet-csrs-for-group=system:bootstrappers",
				"--use-service-account-credentials=true",
				"--controllers=*,bootstrapsigner,tokencleaner",
				"--allocate-node-cidrs=true",
				"--cluster-cidr=bar",
			},
		},
	}

	for _, rt := range tests {
		actual := getControllerManagerCommand(rt.cfg, false)
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
				"--kubeconfig=" + kubeadmapi.GlobalEnvParams.KubernetesDir + "/scheduler.conf",
			},
		},
	}

	for _, rt := range tests {
		actual := getSchedulerCommand(rt.cfg, false)
		sort.Strings(actual)
		sort.Strings(rt.expected)
		if !reflect.DeepEqual(actual, rt.expected) {
			t.Errorf("failed getSchedulerCommand:\nexpected:\n%v\nsaw:\n%v", rt.expected, actual)
		}
	}
}

func TestGetAuthzParameters(t *testing.T) {
	var tests = []struct {
		authMode string
		expected []string
	}{
		{
			authMode: "",
			expected: []string{
				"--authorization-mode=RBAC",
			},
		},
		{
			authMode: "RBAC",
			expected: []string{
				"--authorization-mode=RBAC",
			},
		},
		{
			authMode: "AlwaysAllow",
			expected: []string{
				"--authorization-mode=RBAC,AlwaysAllow",
			},
		},
		{
			authMode: "AlwaysDeny",
			expected: []string{
				"--authorization-mode=RBAC,AlwaysDeny",
			},
		},
		{
			authMode: "ABAC",
			expected: []string{
				"--authorization-mode=RBAC,ABAC",
				"--authorization-policy-file=/etc/kubernetes/abac_policy.json",
			},
		},
		{
			authMode: "Webhook",
			expected: []string{
				"--authorization-mode=RBAC,Webhook",
				"--authorization-webhook-config-file=/etc/kubernetes/webhook_authz.conf",
			},
		},
	}

	for _, rt := range tests {
		actual := getAuthzParameters(rt.authMode)
		for i := range actual {
			if actual[i] != rt.expected[i] {
				t.Errorf(
					"failed getAuthzParameters:\n\texpected: %s\n\t  actual: %s",
					rt.expected[i],
					actual[i],
				)
			}
		}
	}
}

func TestGetExtraParameters(t *testing.T) {
	var tests = []struct {
		overrides map[string]string
		defaults  map[string]string
		expected  []string
	}{
		{
			overrides: map[string]string{
				"admission-control": "NamespaceLifecycle,LimitRanger",
			},
			defaults: map[string]string{
				"admission-control":     "NamespaceLifecycle",
				"insecure-bind-address": "127.0.0.1",
				"allow-privileged":      "true",
			},
			expected: []string{
				"--admission-control=NamespaceLifecycle,LimitRanger",
				"--insecure-bind-address=127.0.0.1",
				"--allow-privileged=true",
			},
		},
		{
			overrides: map[string]string{
				"admission-control": "NamespaceLifecycle,LimitRanger",
			},
			defaults: map[string]string{
				"insecure-bind-address": "127.0.0.1",
				"allow-privileged":      "true",
			},
			expected: []string{
				"--admission-control=NamespaceLifecycle,LimitRanger",
				"--insecure-bind-address=127.0.0.1",
				"--allow-privileged=true",
			},
		},
	}

	for _, rt := range tests {
		actual := getExtraParameters(rt.overrides, rt.defaults)
		sort.Strings(actual)
		sort.Strings(rt.expected)
		if !reflect.DeepEqual(actual, rt.expected) {
			t.Errorf("failed getExtraParameters:\nexpected:\n%v\nsaw:\n%v", rt.expected, actual)
		}
	}
}
