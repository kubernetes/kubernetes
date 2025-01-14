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
	"io"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

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
					ExtraArgs: []kubeadmapi.Arg{},
				},
			},
			expected: "127.0.0.1",
		},
		{
			desc: "setting controller manager extra address arg to something acknowledges it",
			cfg: &kubeadmapi.ClusterConfiguration{
				ControllerManager: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: []kubeadmapi.Arg{
						{Name: kubeControllerManagerBindAddressArg, Value: "10.10.10.10"},
					},
				},
			},
			expected: "10.10.10.10",
		},
		{
			desc: "setting controller manager extra ipv6 address arg to something acknowledges it",
			cfg: &kubeadmapi.ClusterConfiguration{
				ControllerManager: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: []kubeadmapi.Arg{
						{Name: kubeControllerManagerBindAddressArg, Value: "2001:abcd:bcda::1"},
					},
				},
			},
			expected: "2001:abcd:bcda::1",
		},
		{
			desc: "setting controller manager extra address arg to 0.0.0.0 returns empty",
			cfg: &kubeadmapi.ClusterConfiguration{
				ControllerManager: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: []kubeadmapi.Arg{
						{Name: kubeControllerManagerBindAddressArg, Value: "0.0.0.0"},
					},
				},
			},
			expected: "",
		},
		{
			desc: "setting controller manager extra ipv6 address arg to :: returns empty",
			cfg: &kubeadmapi.ClusterConfiguration{
				ControllerManager: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: []kubeadmapi.Arg{
						{Name: kubeControllerManagerBindAddressArg, Value: "::"},
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
					ExtraArgs: []kubeadmapi.Arg{},
				},
			},
			expected: "127.0.0.1",
		},
		{
			desc: "setting scheduler extra address arg to something acknowledges it",
			cfg: &kubeadmapi.ClusterConfiguration{
				Scheduler: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: []kubeadmapi.Arg{
						{Name: kubeSchedulerBindAddressArg, Value: "10.10.10.10"},
					},
				},
			},
			expected: "10.10.10.10",
		},
		{
			desc: "setting scheduler extra ipv6 address arg to something acknowledges it",
			cfg: &kubeadmapi.ClusterConfiguration{
				Scheduler: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: []kubeadmapi.Arg{
						{Name: kubeSchedulerBindAddressArg, Value: "2001:abcd:bcda::1"},
					},
				},
			},
			expected: "2001:abcd:bcda::1",
		},
		{
			desc: "setting scheduler extra ipv6 address arg to 0.0.0.0 returns empty",
			cfg: &kubeadmapi.ClusterConfiguration{
				Scheduler: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: []kubeadmapi.Arg{
						{Name: kubeSchedulerBindAddressArg, Value: "0.0.0.0"},
					},
				},
			},
			expected: "",
		},
		{
			desc: "setting scheduler extra ipv6 address arg to :: returns empty",
			cfg: &kubeadmapi.ClusterConfiguration{
				Scheduler: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: []kubeadmapi.Arg{
						{Name: kubeSchedulerBindAddressArg, Value: "::"},
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
		expectedPort     int32
		expectedScheme   v1.URIScheme
	}{
		{
			name: "etcd probe URL from two URLs",
			cfg: &kubeadmapi.Etcd{
				Local: &kubeadmapi.LocalEtcd{
					ExtraArgs: []kubeadmapi.Arg{
						{Name: "listen-metrics-urls", Value: "https://1.2.3.4:1234,https://4.3.2.1:2381"},
					},
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
					ExtraArgs: []kubeadmapi.Arg{
						{Name: "listen-metrics-urls", Value: "http://1.2.3.4:1234"},
					},
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
					ExtraArgs: []kubeadmapi.Arg{
						{Name: "listen-metrics-urls", Value: "1.2.3.4"},
					},
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
					ExtraArgs: []kubeadmapi.Arg{
						{Name: "listen-metrics-urls", Value: "https://1.2.3.4"},
					},
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
					ExtraArgs: []kubeadmapi.Arg{
						{Name: "listen-metrics-urls", Value: "https://[2001:abcd:bcda::1]:1234,https://[2001:abcd:bcda::2]:2381"},
					},
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
					ExtraArgs: []kubeadmapi.Arg{
						{Name: "listen-metrics-urls", Value: "http://[::1]:1234"},
					},
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
					ExtraArgs: []kubeadmapi.Arg{
						{Name: "listen-metrics-urls", Value: "http://[2001:abcd:bcda::1]:1234"},
					},
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
					ExtraArgs: []kubeadmapi.Arg{
						{Name: "listen-metrics-urls", Value: "https://[2001:abcd:bcda::1]"},
					},
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
	// priority value for system-node-critical class
	priority := int32(2000001000)
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
					SecurityContext: &v1.PodSecurityContext{
						SeccompProfile: &v1.SeccompProfile{
							Type: v1.SeccompProfileTypeRuntimeDefault,
						},
					},
					Containers: []v1.Container{
						{
							Name: "foo",
						},
					},
					Priority:          &priority,
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
	validPodWithDifferentFieldsOrder = `
apiVersion: v1
kind: Pod
metadata:
  labels:
    tier: control-plane
    component: etcd
  name: etcd
  namespace: kube-system
spec:
  containers:
  - image: gcr.io/google_containers/etcd-amd64:3.1.11
status: {}
`
	invalidWithDefaultFields = `
apiVersion: v1
kind: Pod
metadata:
  labels:
    tier: control-plane
    component: etcd
  name: etcd
  namespace: kube-system
spec:
  containers:
  - image: gcr.io/google_containers/etcd-amd64:3.1.11
  restartPolicy: "Always"
status: {}
`

	validPod2 = `
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
  - image: gcr.io/google_containers/etcd-amd64:3.1.12
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
				err := os.WriteFile(manifestPath, []byte(rt.podYaml), 0644)
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

func TestReadMultipleStaticPodsFromDisk(t *testing.T) {
	getTestPod := func(name string) *v1.Pod {
		return &v1.Pod{
			TypeMeta: metav1.TypeMeta{
				Kind:       "Pod",
				APIVersion: "v1",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
		}
	}

	testCases := []struct {
		name                  string
		setup                 func(dir string)
		components            []string
		expected              []*v1.Pod
		expectedErrorContains []string
	}{
		{
			name: "valid: all pods are written and read",
			setup: func(dir string) {
				var pod *v1.Pod
				pod = getTestPod("a")
				_ = WriteStaticPodToDisk(kubeadmconstants.KubeAPIServer, dir, *pod)
				pod = getTestPod("b")
				_ = WriteStaticPodToDisk(kubeadmconstants.KubeControllerManager, dir, *pod)
				pod = getTestPod("c")
				_ = WriteStaticPodToDisk(kubeadmconstants.KubeScheduler, dir, *pod)
			},
			components: kubeadmconstants.ControlPlaneComponents,
			expected: []*v1.Pod{
				getTestPod("a"),
				getTestPod("b"),
				getTestPod("c"),
			},
		},
		{
			name:       "invalid: all pods returned errors",
			setup:      func(dir string) {},
			components: kubeadmconstants.ControlPlaneComponents,
			expectedErrorContains: []string{
				"kube-apiserver.yaml: no such file or directory",
				"kube-controller-manager.yaml: no such file or directory",
				"kube-scheduler.yaml: no such file or directory",
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			tc.setup(dir)
			m, err := ReadMultipleStaticPodsFromDisk(dir, tc.components...)
			if err != nil {
				for _, ec := range tc.expectedErrorContains {
					if !strings.Contains(err.Error(), ec) {
						t.Fatalf("expected error to contain string: %s\nerror:\n%v", ec, err)
					}
				}
			}

			// Compare sorted result to expected result.
			var actual []*v1.Pod
			for _, v := range m {
				actual = append(actual, v)
			}
			sort.Slice(actual, func(a, b int) bool {
				return actual[a].Name < actual[b].Name
			})
			sort.Slice(tc.expected, func(a, b int) bool {
				return actual[a].Name < actual[b].Name
			})

			if diff := cmp.Diff(tc.expected, actual); diff != "" {
				t.Fatalf("unexpected difference (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestManifestFilesAreEqual(t *testing.T) {
	var tests = []struct {
		description    string
		podYamls       []string
		expectedResult bool
		expectedDiff   string
		expectErr      bool
	}{
		{
			description:    "manifests are equal",
			podYamls:       []string{validPod, validPod},
			expectedResult: true,
			expectErr:      false,
		},
		{
			description:    "manifests are equal, ignore different fields order",
			podYamls:       []string{validPod, validPodWithDifferentFieldsOrder},
			expectedResult: true,
			expectErr:      false,
		},
		{
			description:    "manifests are not equal",
			podYamls:       []string{validPod, validPod2},
			expectedResult: false,
			expectErr:      false,
			expectedDiff: `@@ -12 +12 @@
-  - image: gcr.io/google_containers/etcd-amd64:3.1.11
+  - image: gcr.io/google_containers/etcd-amd64:3.1.12
`,
		},
		{
			description:    "manifests are not equal for adding new defaults",
			podYamls:       []string{validPod, invalidWithDefaultFields},
			expectedResult: false,
			expectErr:      false,
			expectedDiff: `@@ -14,0 +15 @@
+  restartPolicy: Always
`,
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
					err := os.WriteFile(manifestPath, []byte(rt.podYamls[i]), 0644)
					if err != nil {
						t.Fatalf("Failed to write manifest file\n%s\n\tfatal error: %v", rt.description, err)
					}
				}
			}

			// compare them
			result, diff, actualErr := ManifestFilesAreEqual(filepath.Join(tmpdir, "0.yaml"), filepath.Join(tmpdir, "1.yaml"))
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
			if !strings.Contains(diff, rt.expectedDiff) {
				t.Errorf(
					"ManifestFilesAreEqual diff doesn't expected\n%s\n\texpected diff: %s\n\tactual diff: %s",
					rt.description,
					rt.expectedDiff,
					diff,
				)
			}
		})
	}
}

func TestPatchStaticPod(t *testing.T) {
	type file struct {
		name string
		data string
	}

	tests := []struct {
		name          string
		files         []*file
		pod           *v1.Pod
		expectedPod   *v1.Pod
		expectedError bool
	}{
		{
			name: "valid: patch a kube-apiserver target using a couple of ordered patches",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "kube-apiserver",
					Namespace: "foo",
				},
			},
			expectedPod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "kube-apiserver",
					Namespace: "bar2",
				},
			},
			files: []*file{
				{
					name: "kube-apiserver1+merge.json",
					data: `{"metadata":{"namespace":"bar2"}}`,
				},
				{
					name: "kube-apiserver0+json.json",
					data: `[{"op": "replace", "path": "/metadata/namespace", "value": "bar1"}]`,
				},
			},
		},
		{
			name: "invalid: unknown patch target name",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: "bar",
				},
			},
			expectedError: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tempDir, err := os.MkdirTemp("", "patch-files")
			if err != nil {
				t.Fatal(err)
			}
			defer os.RemoveAll(tempDir)

			for _, file := range tc.files {
				filePath := filepath.Join(tempDir, file.name)
				err := os.WriteFile(filePath, []byte(file.data), 0644)
				if err != nil {
					t.Fatalf("could not write temporary file %q", filePath)
				}
			}

			pod, err := PatchStaticPod(tc.pod, tempDir, io.Discard)
			if (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, got: %v, error: %v", tc.expectedError, (err != nil), err)
			}
			if err != nil {
				return
			}

			if tc.expectedPod.String() != pod.String() {
				t.Fatalf("expected object:\n%s\ngot:\n%s", tc.expectedPod.String(), pod.String())
			}
		})
	}
}
