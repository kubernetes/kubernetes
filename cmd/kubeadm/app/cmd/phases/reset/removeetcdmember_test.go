/*
Copyright 2019 The Kubernetes Authors.

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

package phases

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/lithammer/dedent"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
)

const (
	etcdPodContent = `apiVersion: v1
kind: Pod
spec:
  volumes:
  - hostPath:
      path: /path/to/etcd
      type: DirectoryOrCreate
    name: etcd-data
  - hostPath:
      path: /etc/kubernetes/pki/etcd
      type: DirectoryOrCreate
    name: etcd-certs`

	etcdPodInvalidContent = `invalid pod`
)

var (
	pathType                     = v1.HostPathDirectoryOrCreate
	etcdPodWithoutDataVolumeSpec = &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "etcd-certs",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: "/etc/kubernetes/pki/etcd",
							Type: &pathType,
						},
					},
				},
			},
		},
	}
	etcdPodSpec = &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "etcd-data",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: "/path/to/etcd",
							Type: &pathType,
						},
					},
				},
				{
					Name: "etcd-certs",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: "/etc/kubernetes/pki/etcd",
							Type: &pathType,
						},
					},
				},
			},
		},
	}
)

func TestGetEtcdDataDir(t *testing.T) {
	tests := map[string]struct {
		dataDir   string
		podYaml   string
		expectErr bool
		etcdPod   *v1.Pod
	}{
		"non-existent file returns default data dir": {
			expectErr: false,
			dataDir:   "/var/lib/etcd",
		},
		"return etcd data dir": {
			dataDir:   "/path/to/etcd",
			etcdPod:   etcdPodSpec,
			expectErr: false,
		},
		"etcd pod spec without data volume": {
			etcdPod:   etcdPodWithoutDataVolumeSpec,
			expectErr: true,
		},
		"kubeconfig file doesn't exist": {
			dataDir:   "/path/to/etcd",
			etcdPod:   etcdPodSpec,
			expectErr: false,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			tmpdir := testutil.SetupTempDir(t)
			defer os.RemoveAll(tmpdir)

			dataDir, err := getEtcdDataDir(test.etcdPod)

			if (err != nil) != test.expectErr {
				t.Fatalf(dedent.Dedent(
					"getEtcdDataDir failed\n%s\nexpected error: %t\n\tgot: %t\nerror: %v"),
					name,
					test.expectErr,
					(err != nil),
					err,
				)
			}

			if dataDir != test.dataDir {
				t.Fatalf(dedent.Dedent("getEtcdDataDir failed\n%s\n\texpected: %s\ngot: %s"), name, test.dataDir, dataDir)
			}
		})
	}
}
func TestGetEtcdAdvertiseAddress(t *testing.T) {
	tests := map[string]struct {
		etcdPod         *v1.Pod
		podYaml         string
		expectedAddress string
		expectedError   bool
	}{
		"empty etcd advertise client urls": {
			etcdPod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						kubeadmconstants.EtcdAdvertiseClientUrlsAnnotationKey: "",
					},
				},
			},
			expectedAddress: "",
		},
		"valid etcd advertise client urls": {
			etcdPod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						kubeadmconstants.EtcdAdvertiseClientUrlsAnnotationKey: "http://localhost:2379",
					},
				},
			},
			expectedAddress: "localhost",
		},
		"invalid etcd advertise client urls": {
			etcdPod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						kubeadmconstants.EtcdAdvertiseClientUrlsAnnotationKey: "invalid-url",
					},
				},
			},
			expectedAddress: "",
			expectedError:   true,
		},
	}

	for name, test := range tests {
		test := test
		t.Run(name, func(t *testing.T) {
			address, err := getEtcdAdvertiseAddress(test.etcdPod)

			if address != test.expectedAddress {
				t.Fatalf("getEtcdAdvertiseAddress failed\n%s\n\texpected address: %s\ngot: %s", name, test.expectedAddress, address)
			}

			if (err != nil) != test.expectedError {
				t.Fatalf("getEtcdAdvertiseAddress failed\n%s\nexpected error: %t\n\tgot: %t\nerror: %v", name, test.expectedError, (err != nil), err)
			}
		})
	}
}
func TestGetEtcdPod(t *testing.T) {
	tests := map[string]struct {
		expectedPod   bool
		expectedErr   bool
		writeManifest bool
		podYaml       string
	}{
		"non-existent manifest file returns nil pod and no error": {
			expectedPod:   false,
			expectedErr:   false,
			writeManifest: false,
		},
		"existing manifest file returns the pod and no error": {
			expectedPod:   true,
			expectedErr:   false,
			writeManifest: true,
			podYaml:       etcdPodContent,
		},
		"error reading manifest file returns nil pod and the error": {
			expectedPod:   false,
			expectedErr:   true,
			writeManifest: true,
			podYaml:       etcdPodInvalidContent,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			tmpdir := testutil.SetupTempDir(t)
			defer os.RemoveAll(tmpdir)

			manifestPath := filepath.Join(tmpdir, "etcd.yaml")

			if test.writeManifest {
				err := os.WriteFile(manifestPath, []byte(test.podYaml), 0644)
				if err != nil {
					t.Fatalf(dedent.Dedent("failed to write pod manifest\n%s\n\tfatal error: %v"), name, err)
				}
			}

			pod, err := getEtcdPod(manifestPath)

			if (err != nil) != test.expectedErr {
				t.Fatalf(dedent.Dedent(
					"getEtcdPod failed\n%s\nexpected error: %t\n\tgot: %t\nerror: %v"),
					name,
					test.expectedErr,
					(err != nil),
					err,
				)
			}

			if (pod != nil) != test.expectedPod {
				t.Fatalf(dedent.Dedent(
					"getEtcdPod failed\n%s\nexpected pod: %t\n\tgot: %t\npod: %v"),
					name,
					test.expectedPod,
					(pod != nil),
					pod,
				)
			}

		})
	}
}
