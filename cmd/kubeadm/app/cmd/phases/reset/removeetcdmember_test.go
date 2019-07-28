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
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"github.com/lithammer/dedent"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
)

const (
	etcdPod = `apiVersion: v1
kind: Pod
metadata:
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

	etcdPodWithoutDataVolume = `apiVersion: v1
kind: Pod
metadata:
spec:
  volumes:
  - hostPath:
      path: /etc/kubernetes/pki/etcd
      type: DirectoryOrCreate
    name: etcd-certs`

	etcdPodInvalid = `invalid pod`
)

func TestGetEtcdDataDir(t *testing.T) {
	tests := map[string]struct {
		dataDir       string
		podYaml       string
		expectErr     bool
		writeManifest bool
		validConfig   bool
	}{
		"non-existent file returns error": {
			expectErr:     true,
			writeManifest: false,
			validConfig:   true,
		},
		"return etcd data dir": {
			dataDir:       "/path/to/etcd",
			podYaml:       etcdPod,
			expectErr:     false,
			writeManifest: true,
			validConfig:   true,
		},
		"invalid etcd pod": {
			podYaml:       etcdPodInvalid,
			expectErr:     true,
			writeManifest: true,
			validConfig:   true,
		},
		"etcd pod spec without data volume": {
			podYaml:       etcdPodWithoutDataVolume,
			expectErr:     true,
			writeManifest: true,
			validConfig:   true,
		},
		"kubeconfig file doesn't exist": {
			dataDir:       "/path/to/etcd",
			podYaml:       etcdPod,
			expectErr:     false,
			writeManifest: true,
			validConfig:   false,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			tmpdir := testutil.SetupTempDir(t)
			defer os.RemoveAll(tmpdir)

			manifestPath := filepath.Join(tmpdir, "etcd.yaml")
			if test.writeManifest {
				err := ioutil.WriteFile(manifestPath, []byte(test.podYaml), 0644)
				if err != nil {
					t.Fatalf(dedent.Dedent("failed to write pod manifest\n%s\n\tfatal error: %v"), name, err)
				}
			}

			var dataDir string
			var err error
			if test.validConfig {
				cfg := &kubeadmapi.InitConfiguration{}
				dataDir, err = getEtcdDataDir(manifestPath, cfg)
			} else {
				dataDir, err = getEtcdDataDir(manifestPath, nil)
			}

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
