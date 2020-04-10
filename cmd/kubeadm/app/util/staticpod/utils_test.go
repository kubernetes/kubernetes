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

package staticpod

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"testing"

	"github.com/lithammer/dedent"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
)

func TestComponentResources(t *testing.T) {
	a := ComponentResources("250m")
	if a.Requests == nil {
		t.Errorf(
			"failed componentResources, return value was nil",
		)
	}
}

func TestGetAPIServerProbeAddress(t *testing.T) {
	tests := []struct {
		desc     string
		endpoint *kubeadmapi.APIEndpoint
		expected string
	}{
		{
			desc:     "nil endpoint returns 127.0.0.1",
			expected: "127.0.0.1",
		},
		{
			desc:     "empty AdvertiseAddress endpoint returns 127.0.0.1",
			endpoint: &kubeadmapi.APIEndpoint{},
			expected: "127.0.0.1",
		},
		{
			desc: "filled in AdvertiseAddress endpoint returns it",
			endpoint: &kubeadmapi.APIEndpoint{
				AdvertiseAddress: "10.10.10.10",
			},
			expected: "10.10.10.10",
		},
		{
			desc: "filled in ipv6 AdvertiseAddress endpoint returns it",
			endpoint: &kubeadmapi.APIEndpoint{
				AdvertiseAddress: "2001:abcd:bcda::1",
			},
			expected: "2001:abcd:bcda::1",
		},
		{
			desc: "filled in 0.0.0.0 AdvertiseAddress endpoint returns empty",
			endpoint: &kubeadmapi.APIEndpoint{
				AdvertiseAddress: "0.0.0.0",
			},
			expected: "",
		},
		{
			desc: "filled in :: AdvertiseAddress endpoint returns empty",
			endpoint: &kubeadmapi.APIEndpoint{
				AdvertiseAddress: "::",
			},
			expected: "",
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			actual := GetAPIServerProbeAddress(test.endpoint)
			if actual != test.expected {
				t.Errorf("Unexpected result from GetAPIServerProbeAddress:\n\texpected: %s\n\tactual: %s", test.expected, actual)
			}
		})
	}
}

func TestGetControllerManagerProbeAddress(t *testing.T) {
	tests := []struct {
		desc     string
		cfg      *kubeadmapi.ClusterConfiguration
		expected string
	}{
		{
			desc: "no controller manager extra args leads to 127.0.0.1 being used",
			cfg: &kubeadmapi.ClusterConfiguration{
				ControllerManager: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: map[string]string{},
				},
			},
			expected: "127.0.0.1",
		},
		{
			desc: "setting controller manager extra address arg to something acknowledges it",
			cfg: &kubeadmapi.ClusterConfiguration{
				ControllerManager: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: map[string]string{
						kubeControllerManagerBindAddressArg: "10.10.10.10",
					},
				},
			},
			expected: "10.10.10.10",
		},
		{
			desc: "setting controller manager extra ipv6 address arg to something acknowledges it",
			cfg: &kubeadmapi.ClusterConfiguration{
				ControllerManager: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: map[string]string{
						kubeControllerManagerBindAddressArg: "2001:abcd:bcda::1",
					},
				},
			},
			expected: "2001:abcd:bcda::1",
		},
		{
			desc: "setting controller manager extra address arg to 0.0.0.0 returns empty",
			cfg: &kubeadmapi.ClusterConfiguration{
				ControllerManager: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: map[string]string{
						kubeControllerManagerBindAddressArg: "0.0.0.0",
					},
				},
			},
			expected: "",
		},
		{
			desc: "setting controller manager extra ipv6 address arg to :: returns empty",
			cfg: &kubeadmapi.ClusterConfiguration{
				ControllerManager: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: map[string]string{
						kubeControllerManagerBindAddressArg: "::",
					},
				},
			},
			expected: "",
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			actual := GetControllerManagerProbeAddress(test.cfg)
			if actual != test.expected {
				t.Errorf("Unexpected result from GetControllerManagerProbeAddress:\n\texpected: %s\n\tactual: %s", test.expected, actual)
			}
		})
	}
}

func TestGetSchedulerProbeAddress(t *testing.T) {
	tests := []struct {
		desc     string
		cfg      *kubeadmapi.ClusterConfiguration
		expected string
	}{
		{
			desc: "no scheduler extra args leads to 127.0.0.1 being used",
			cfg: &kubeadmapi.ClusterConfiguration{
				Scheduler: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: map[string]string{},
				},
			},
			expected: "127.0.0.1",
		},
		{
			desc: "setting scheduler extra address arg to something acknowledges it",
			cfg: &kubeadmapi.ClusterConfiguration{
				Scheduler: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: map[string]string{
						kubeSchedulerBindAddressArg: "10.10.10.10",
					},
				},
			},
			expected: "10.10.10.10",
		},
		{
			desc: "setting scheduler extra ipv6 address arg to something acknowledges it",
			cfg: &kubeadmapi.ClusterConfiguration{
				Scheduler: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: map[string]string{
						kubeSchedulerBindAddressArg: "2001:abcd:bcda::1",
					},
				},
			},
			expected: "2001:abcd:bcda::1",
		},
		{
			desc: "setting scheduler extra ipv6 address arg to 0.0.0.0 returns empty",
			cfg: &kubeadmapi.ClusterConfiguration{
				Scheduler: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: map[string]string{
						kubeSchedulerBindAddressArg: "0.0.0.0",
					},
				},
			},
			expected: "",
		},
		{
			desc: "setting scheduler extra ipv6 address arg to :: returns empty",
			cfg: &kubeadmapi.ClusterConfiguration{
				Scheduler: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: map[string]string{
						kubeSchedulerBindAddressArg: "::",
					},
				},
			},
			expected: "",
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			actual := GetSchedulerProbeAddress(test.cfg)
			if actual != test.expected {
				t.Errorf("Unexpected result from GetSchedulerProbeAddress:\n\texpected: %s\n\tactual: %s", test.expected, actual)
			}
		})
	}
}
func TestGetEtcdProbeEndpoint(t *testing.T) {
	var tests = []struct {
		name             string
		cfg              *kubeadmapi.Etcd
		isIPv6           bool
		expectedHostname string
		expectedPort     int
		expectedScheme   v1.URIScheme
	}{
		{
			name: "etcd probe URL from two URLs",
			cfg: &kubeadmapi.Etcd{
				Local: &kubeadmapi.LocalEtcd{
					ExtraArgs: map[string]string{
						"listen-metrics-urls": "https://1.2.3.4:1234,https://4.3.2.1:2381"},
				},
			},
			isIPv6:           false,
			expectedHostname: "1.2.3.4",
			expectedPort:     1234,
			expectedScheme:   v1.URISchemeHTTPS,
		},
		{
			name: "etcd probe URL with HTTP scheme",
			cfg: &kubeadmapi.Etcd{
				Local: &kubeadmapi.LocalEtcd{
					ExtraArgs: map[string]string{
						"listen-metrics-urls": "http://1.2.3.4:1234"},
				},
			},
			isIPv6:           false,
			expectedHostname: "1.2.3.4",
			expectedPort:     1234,
			expectedScheme:   v1.URISchemeHTTP,
		},
		{
			name: "etcd probe URL without scheme should result in defaults",
			cfg: &kubeadmapi.Etcd{
				Local: &kubeadmapi.LocalEtcd{
					ExtraArgs: map[string]string{
						"listen-metrics-urls": "1.2.3.4"},
				},
			},
			isIPv6:           false,
			expectedHostname: "127.0.0.1",
			expectedPort:     kubeadmconstants.EtcdMetricsPort,
			expectedScheme:   v1.URISchemeHTTP,
		},
		{
			name: "etcd probe URL without port",
			cfg: &kubeadmapi.Etcd{
				Local: &kubeadmapi.LocalEtcd{
					ExtraArgs: map[string]string{
						"listen-metrics-urls": "https://1.2.3.4"},
				},
			},
			isIPv6:           false,
			expectedHostname: "1.2.3.4",
			expectedPort:     kubeadmconstants.EtcdMetricsPort,
			expectedScheme:   v1.URISchemeHTTPS,
		},
		{
			name: "etcd probe URL from two IPv6 URLs",
			cfg: &kubeadmapi.Etcd{
				Local: &kubeadmapi.LocalEtcd{
					ExtraArgs: map[string]string{
						"listen-metrics-urls": "https://[2001:abcd:bcda::1]:1234,https://[2001:abcd:bcda::2]:2381"},
				},
			},
			isIPv6:           true,
			expectedHostname: "2001:abcd:bcda::1",
			expectedPort:     1234,
			expectedScheme:   v1.URISchemeHTTPS,
		},
		{
			name: "etcd probe localhost IPv6 URL with HTTP scheme",
			cfg: &kubeadmapi.Etcd{
				Local: &kubeadmapi.LocalEtcd{
					ExtraArgs: map[string]string{
						"listen-metrics-urls": "http://[::1]:1234"},
				},
			},
			isIPv6:           true,
			expectedHostname: "::1",
			expectedPort:     1234,
			expectedScheme:   v1.URISchemeHTTP,
		},
		{
			name: "etcd probe IPv6 URL with HTTP scheme",
			cfg: &kubeadmapi.Etcd{
				Local: &kubeadmapi.LocalEtcd{
					ExtraArgs: map[string]string{
						"listen-metrics-urls": "http://[2001:abcd:bcda::1]:1234"},
				},
			},
			isIPv6:           true,
			expectedHostname: "2001:abcd:bcda::1",
			expectedPort:     1234,
			expectedScheme:   v1.URISchemeHTTP,
		},
		{
			name: "etcd probe IPv6 URL without port",
			cfg: &kubeadmapi.Etcd{
				Local: &kubeadmapi.LocalEtcd{
					ExtraArgs: map[string]string{
						"listen-metrics-urls": "https://[2001:abcd:bcda::1]"},
				},
			},
			isIPv6:           true,
			expectedHostname: "2001:abcd:bcda::1",
			expectedPort:     kubeadmconstants.EtcdMetricsPort,
			expectedScheme:   v1.URISchemeHTTPS,
		},
		{
			name: "etcd probe URL from defaults",
			cfg: &kubeadmapi.Etcd{
				Local: &kubeadmapi.LocalEtcd{},
			},
			isIPv6:           false,
			expectedHostname: "127.0.0.1",
			expectedPort:     kubeadmconstants.EtcdMetricsPort,
			expectedScheme:   v1.URISchemeHTTP,
		},
		{
			name: "etcd probe URL from defaults if IPv6",
			cfg: &kubeadmapi.Etcd{
				Local: &kubeadmapi.LocalEtcd{},
			},
			isIPv6:           true,
			expectedHostname: "::1",
			expectedPort:     kubeadmconstants.EtcdMetricsPort,
			expectedScheme:   v1.URISchemeHTTP,
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			hostname, port, scheme := GetEtcdProbeEndpoint(rt.cfg, rt.isIPv6)
			if hostname != rt.expectedHostname {
				t.Errorf("%q test case failed:\n\texpected hostname: %s\n\tgot: %s",
					rt.name, rt.expectedHostname, hostname)
			}
			if port != rt.expectedPort {
				t.Errorf("%q test case failed:\n\texpected port: %d\n\tgot: %d",
					rt.name, rt.expectedPort, port)
			}
			if scheme != rt.expectedScheme {
				t.Errorf("%q test case failed:\n\texpected scheme: %v\n\tgot: %v",
					rt.name, rt.expectedScheme, scheme)
			}
		})
	}
}

func TestComponentPod(t *testing.T) {
	var tests = []struct {
		name     string
		expected v1.Pod
	}{
		{
			name: "foo",
			expected: v1.Pod{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "v1",
					Kind:       "Pod",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: "kube-system",
					Labels:    map[string]string{"component": "foo", "tier": "control-plane"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "foo",
						},
					},
					PriorityClassName: "system-node-critical",
					HostNetwork:       true,
					Volumes:           []v1.Volume{},
				},
			},
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			c := v1.Container{Name: rt.name}
			actual := ComponentPod(c, map[string]v1.Volume{}, nil)
			if !reflect.DeepEqual(rt.expected, actual) {
				t.Errorf(
					"failed componentPod:\n\texpected: %v\n\t  actual: %v",
					rt.expected,
					actual,
				)
			}
		})
	}
}

func TestNewVolume(t *testing.T) {
	hostPathDirectoryOrCreate := v1.HostPathDirectoryOrCreate
	var tests = []struct {
		name     string
		path     string
		expected v1.Volume
		pathType *v1.HostPathType
	}{
		{
			name: "foo",
			path: "/etc/foo",
			expected: v1.Volume{
				Name: "foo",
				VolumeSource: v1.VolumeSource{
					HostPath: &v1.HostPathVolumeSource{
						Path: "/etc/foo",
						Type: &hostPathDirectoryOrCreate,
					},
				},
			},
			pathType: &hostPathDirectoryOrCreate,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			actual := NewVolume(rt.name, rt.path, rt.pathType)
			if !reflect.DeepEqual(actual, rt.expected) {
				t.Errorf(
					"failed newVolume:\n\texpected: %v\n\t  actual: %v",
					rt.expected,
					actual,
				)
			}
		})
	}
}

func TestNewVolumeMount(t *testing.T) {
	var tests = []struct {
		name     string
		path     string
		ro       bool
		expected v1.VolumeMount
	}{
		{
			name: "foo",
			path: "/etc/foo",
			ro:   false,
			expected: v1.VolumeMount{
				Name:      "foo",
				MountPath: "/etc/foo",
				ReadOnly:  false,
			},
		},
		{
			name: "bar",
			path: "/etc/foo/bar",
			ro:   true,
			expected: v1.VolumeMount{
				Name:      "bar",
				MountPath: "/etc/foo/bar",
				ReadOnly:  true,
			},
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			actual := NewVolumeMount(rt.name, rt.path, rt.ro)
			if !reflect.DeepEqual(actual, rt.expected) {
				t.Errorf(
					"failed newVolumeMount:\n\texpected: %v\n\t  actual: %v",
					rt.expected,
					actual,
				)
			}
		})
	}
}
func TestVolumeMapToSlice(t *testing.T) {
	testVolumes := map[string]v1.Volume{
		"foo": {
			Name: "foo",
		},
		"bar": {
			Name: "bar",
		},
	}
	volumeSlice := VolumeMapToSlice(testVolumes)
	if len(volumeSlice) != 2 {
		t.Errorf("Expected slice length of 1, got %d", len(volumeSlice))
	}
	if volumeSlice[0].Name != "bar" {
		t.Errorf("Expected first volume name \"bar\", got %s", volumeSlice[0].Name)
	}
	if volumeSlice[1].Name != "foo" {
		t.Errorf("Expected second volume name \"foo\", got %s", volumeSlice[1].Name)
	}
}

func TestVolumeMountMapToSlice(t *testing.T) {
	testVolumeMounts := map[string]v1.VolumeMount{
		"foo": {
			Name: "foo",
		},
		"bar": {
			Name: "bar",
		},
	}
	volumeMountSlice := VolumeMountMapToSlice(testVolumeMounts)
	if len(volumeMountSlice) != 2 {
		t.Errorf("Expected slice length of 1, got %d", len(volumeMountSlice))
	}
	if volumeMountSlice[0].Name != "bar" {
		t.Errorf("Expected first volume mount name \"bar\", got %s", volumeMountSlice[0].Name)
	}
	if volumeMountSlice[1].Name != "foo" {
		t.Errorf("Expected second volume name \"foo\", got %s", volumeMountSlice[1].Name)
	}
}

func TestGetExtraParameters(t *testing.T) {
	var tests = []struct {
		name      string
		overrides map[string]string
		defaults  map[string]string
		expected  []string
	}{
		{
			name: "with admission-control default NamespaceLifecycle",
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
			name: "without admission-control default",
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
		t.Run(rt.name, func(t *testing.T) {
			actual := GetExtraParameters(rt.overrides, rt.defaults)
			sort.Strings(actual)
			sort.Strings(rt.expected)
			if !reflect.DeepEqual(actual, rt.expected) {
				t.Errorf("failed getExtraParameters:\nexpected:\n%v\nsaw:\n%v", rt.expected, actual)
			}
		})
	}
}

const (
	validPod = `
apiVersion: v1
kind: Pod
metadata:
  labels:
    component: etcd
    tier: control-plane
  name: etcd
  namespace: kube-system
spec:
  containers:
  - image: gcr.io/google_containers/etcd-amd64:3.1.11
status: {}
`
	invalidPod = `---{ broken yaml @@@`
)

func TestReadStaticPodFromDisk(t *testing.T) {
	tests := []struct {
		description   string
		podYaml       string
		expectErr     bool
		writeManifest bool
	}{
		{
			description:   "valid pod is marshaled",
			podYaml:       validPod,
			writeManifest: true,
			expectErr:     false,
		},
		{
			description:   "invalid pod fails to unmarshal",
			podYaml:       invalidPod,
			writeManifest: true,
			expectErr:     true,
		},
		{
			description:   "non-existent file returns error",
			podYaml:       ``,
			writeManifest: false,
			expectErr:     true,
		},
	}

	for _, rt := range tests {
		t.Run(rt.description, func(t *testing.T) {
			tmpdir := testutil.SetupTempDir(t)
			defer os.RemoveAll(tmpdir)

			manifestPath := filepath.Join(tmpdir, "pod.yaml")
			if rt.writeManifest {
				err := ioutil.WriteFile(manifestPath, []byte(rt.podYaml), 0644)
				if err != nil {
					t.Fatalf("Failed to write pod manifest\n%s\n\tfatal error: %v", rt.description, err)
				}
			}

			_, actualErr := ReadStaticPodFromDisk(manifestPath)
			if (actualErr != nil) != rt.expectErr {
				t.Errorf(
					"ReadStaticPodFromDisk failed\n%s\n\texpected error: %t\n\tgot: %t\n\tactual error: %v",
					rt.description,
					rt.expectErr,
					(actualErr != nil),
					actualErr,
				)
			}
		})
	}
}

func TestManifestFilesAreEqual(t *testing.T) {
	var tests = []struct {
		description    string
		podYamls       []string
		expectedResult bool
		expectErr      bool
	}{
		{
			description:    "manifests are equal",
			podYamls:       []string{validPod, validPod},
			expectedResult: true,
			expectErr:      false,
		},
		{
			description:    "manifests are not equal",
			podYamls:       []string{validPod, validPod + "\n"},
			expectedResult: false,
			expectErr:      false,
		},
		{
			description:    "first manifest doesn't exist",
			podYamls:       []string{validPod, ""},
			expectedResult: false,
			expectErr:      true,
		},
		{
			description:    "second manifest doesn't exist",
			podYamls:       []string{"", validPod},
			expectedResult: false,
			expectErr:      true,
		},
	}

	for _, rt := range tests {
		t.Run(rt.description, func(t *testing.T) {
			tmpdir := testutil.SetupTempDir(t)
			defer os.RemoveAll(tmpdir)

			// write 2 manifests
			for i := 0; i < 2; i++ {
				if rt.podYamls[i] != "" {
					manifestPath := filepath.Join(tmpdir, strconv.Itoa(i)+".yaml")
					err := ioutil.WriteFile(manifestPath, []byte(rt.podYamls[i]), 0644)
					if err != nil {
						t.Fatalf("Failed to write manifest file\n%s\n\tfatal error: %v", rt.description, err)
					}
				}
			}

			// compare them
			result, actualErr := ManifestFilesAreEqual(filepath.Join(tmpdir, "0.yaml"), filepath.Join(tmpdir, "1.yaml"))
			if result != rt.expectedResult {
				t.Errorf(
					"ManifestFilesAreEqual failed\n%s\nexpected result: %t\nactual result: %t",
					rt.description,
					rt.expectedResult,
					result,
				)
			}
			if (actualErr != nil) != rt.expectErr {
				t.Errorf(
					"ManifestFilesAreEqual failed\n%s\n\texpected error: %t\n\tgot: %t\n\tactual error: %v",
					rt.description,
					rt.expectErr,
					(actualErr != nil),
					actualErr,
				)
			}
		})
	}
}

func TestKustomizeStaticPod(t *testing.T) {
	// Create temp folder for the test case
	tmpdir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpdir)

	patchString := dedent.Dedent(`
    apiVersion: v1
    kind: Pod
    metadata:
        name: kube-apiserver
        namespace: kube-system
        annotations:
            kustomize: patch for kube-apiserver
    `)

	err := ioutil.WriteFile(filepath.Join(tmpdir, "patch.yaml"), []byte(patchString), 0644)
	if err != nil {
		t.Fatalf("WriteFile returned unexpected error: %v", err)
	}

	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "Pod",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "kube-apiserver",
			Namespace: "kube-system",
		},
	}

	kpod, err := KustomizeStaticPod(pod, tmpdir)
	if err != nil {
		t.Errorf("KustomizeStaticPod returned unexpected error: %v", err)
	}

	if _, ok := kpod.ObjectMeta.Annotations["kustomize"]; !ok {
		t.Error("Kustomize did not apply patches corresponding to the resource")
	}
}
