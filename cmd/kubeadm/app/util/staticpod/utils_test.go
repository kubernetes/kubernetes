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

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
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
						kubeControllerManagerAddressArg: "10.10.10.10",
					},
				},
			},
			expected: "10.10.10.10",
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

func TestEtcdProbe(t *testing.T) {
	var tests = []struct {
		name     string
		cfg      *kubeadmapi.Etcd
		port     int
		certsDir string
		cacert   string
		cert     string
		key      string
		expected string
	}{
		{
			name: "valid etcd probe using listen-client-urls IPv4 addresses",
			cfg: &kubeadmapi.Etcd{
				Local: &kubeadmapi.LocalEtcd{
					ExtraArgs: map[string]string{
						"listen-client-urls": "http://1.2.3.4:2379,http://4.3.2.1:2379"},
				},
			},
			port:     1,
			certsDir: "secretsA",
			cacert:   "ca1",
			cert:     "cert1",
			key:      "key1",
			expected: "ETCDCTL_API=3 etcdctl --endpoints=https://[1.2.3.4]:1 --cacert=secretsA/ca1 --cert=secretsA/cert1 --key=secretsA/key1 get foo",
		},
		{
			name: "valid etcd probe using listen-client-urls unspecified IPv6 address",
			cfg: &kubeadmapi.Etcd{
				Local: &kubeadmapi.LocalEtcd{
					ExtraArgs: map[string]string{
						"listen-client-urls": "http://[0:0:0:0:0:0:0:0]:2379"},
				},
			},
			port:     1,
			certsDir: "secretsB",
			cacert:   "ca2",
			cert:     "cert2",
			key:      "key2",
			expected: "ETCDCTL_API=3 etcdctl --endpoints=https://[::1]:1 --cacert=secretsB/ca2 --cert=secretsB/cert2 --key=secretsB/key2 get foo",
		},
		{
			name: "valid etcd probe using listen-client-urls unspecified IPv6 address 2",
			cfg: &kubeadmapi.Etcd{
				Local: &kubeadmapi.LocalEtcd{
					ExtraArgs: map[string]string{
						"listen-client-urls": "http://[::0:0]:2379"},
				},
			},
			port:     1,
			certsDir: "secretsB",
			cacert:   "ca2",
			cert:     "cert2",
			key:      "key2",
			expected: "ETCDCTL_API=3 etcdctl --endpoints=https://[::1]:1 --cacert=secretsB/ca2 --cert=secretsB/cert2 --key=secretsB/key2 get foo",
		},
		{
			name: "valid etcd probe using listen-client-urls unspecified IPv6 address 3",
			cfg: &kubeadmapi.Etcd{
				Local: &kubeadmapi.LocalEtcd{
					ExtraArgs: map[string]string{
						"listen-client-urls": "http://[::]:2379"},
				},
			},
			port:     1,
			certsDir: "secretsB",
			cacert:   "ca2",
			cert:     "cert2",
			key:      "key2",
			expected: "ETCDCTL_API=3 etcdctl --endpoints=https://[::1]:1 --cacert=secretsB/ca2 --cert=secretsB/cert2 --key=secretsB/key2 get foo",
		},
		{
			name: "valid etcd probe using listen-client-urls unspecified IPv4 address",
			cfg: &kubeadmapi.Etcd{
				Local: &kubeadmapi.LocalEtcd{
					ExtraArgs: map[string]string{
						"listen-client-urls": "http://1.2.3.4:2379,http://4.3.2.1:2379"},
				},
			},
			port:     1,
			certsDir: "secretsA",
			cacert:   "ca1",
			cert:     "cert1",
			key:      "key1",
			expected: "ETCDCTL_API=3 etcdctl --endpoints=https://[1.2.3.4]:1 --cacert=secretsA/ca1 --cert=secretsA/cert1 --key=secretsA/key1 get foo",
		},
		{
			name: "valid etcd probe using listen-client-urls IPv6 addresses",
			cfg: &kubeadmapi.Etcd{
				Local: &kubeadmapi.LocalEtcd{
					ExtraArgs: map[string]string{
						"listen-client-urls": "http://[2001:db8::1]:2379,http://[2001:db8::2]:2379"},
				},
			},
			port:     1,
			certsDir: "secretsB",
			cacert:   "ca2",
			cert:     "cert2",
			key:      "key2",
			expected: "ETCDCTL_API=3 etcdctl --endpoints=https://[2001:db8::1]:1 --cacert=secretsB/ca2 --cert=secretsB/cert2 --key=secretsB/key2 get foo",
		},
		{
			name: "valid IPv4 etcd probe using hostname for listen-client-urls",
			cfg: &kubeadmapi.Etcd{
				Local: &kubeadmapi.LocalEtcd{
					ExtraArgs: map[string]string{
						"listen-client-urls": "http://localhost:2379"},
				},
			},
			port:     1,
			certsDir: "secretsC",
			cacert:   "ca3",
			cert:     "cert3",
			key:      "key3",
			expected: "ETCDCTL_API=3 etcdctl --endpoints=https://[127.0.0.1]:1 --cacert=secretsC/ca3 --cert=secretsC/cert3 --key=secretsC/key3 get foo",
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			actual := EtcdProbe(rt.cfg, rt.port, rt.certsDir, rt.cacert, rt.cert, rt.key)
			if actual.Handler.Exec.Command[2] != rt.expected {
				t.Errorf("%s test case failed:\n\texpected: %s\n\t  actual: %s",
					rt.name, rt.expected,
					actual.Handler.Exec.Command[2])
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
					PriorityClassName: "system-cluster-critical",
					HostNetwork:       true,
					Volumes:           []v1.Volume{},
				},
			},
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			c := v1.Container{Name: rt.name}
			actual := ComponentPod(c, map[string]v1.Volume{})
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
