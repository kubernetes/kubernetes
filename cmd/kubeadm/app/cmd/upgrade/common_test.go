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

package upgrade

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/pflag"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/output"
	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
)

const testConfigToken = `apiVersion: v1
clusters:
- cluster:
    certificate-authority-data:
    server: localhost:8000
  name: prod
contexts:
- context:
    cluster: prod
    namespace: default
    user: default-service-account
  name: default
current-context: default
kind: Config
preferences: {}
users:
- name: kubernetes-admin
  user:
    client-certificate-data:
`

func TestEnforceRequirements(t *testing.T) {
	tmpDir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpDir)
	fullPath := filepath.Join(tmpDir, "test-config-file")
	f, err := os.Create(fullPath)
	if err != nil {
		t.Errorf("Unable to create test file %q: %v", fullPath, err)
	}
	defer f.Close()
	if _, err = f.WriteString(testConfigToken); err != nil {
		t.Errorf("Unable to write test file %q: %v", fullPath, err)
	}
	tcases := []struct {
		name               string
		newK8sVersion      string
		dryRun             bool
		flags              applyPlanFlags
		expectedErr        string
		expectedErrNonRoot string
	}{
		{
			name: "Fail pre-flight check",
			flags: applyPlanFlags{
				kubeConfigPath: fullPath,
			},
			expectedErr:        "ERROR CoreDNSUnsupportedPlugins",
			expectedErrNonRoot: "user is not running as", // user is not running as (root || administrator)
		},
		{
			name: "Bogus preflight check specify all with individual check",
			flags: applyPlanFlags{
				ignorePreflightErrors: []string{"bogusvalue", "all"},
				kubeConfigPath:        fullPath,
			},
			expectedErr: "don't specify individual checks if 'all' is used",
		},
		{
			name: "Fail to create client",
			flags: applyPlanFlags{
				ignorePreflightErrors: []string{"all"},
			},
			expectedErr: "couldn't create a Kubernetes client from file",
		},
	}
	for _, tt := range tcases {
		t.Run(tt.name, func(t *testing.T) {
			_, _, _, _, err := enforceRequirements(&pflag.FlagSet{}, &tt.flags, nil, tt.dryRun, false, &output.TextPrinter{})
			if err == nil && len(tt.expectedErr) != 0 {
				t.Error("Expected error, but got success")
			}

			expErr := tt.expectedErr
			// pre-flight check expects the user to be root, so the root and non-root should hit different errors
			isPrivileged := preflight.IsPrivilegedUserCheck{}
			// this will return an array of errors if we're not running as a privileged user.
			_, errors := isPrivileged.Check()
			if len(errors) != 0 && len(tt.expectedErrNonRoot) != 0 {
				expErr = tt.expectedErrNonRoot
			}
			if err != nil && !strings.Contains(err.Error(), expErr) {
				t.Fatalf("enforceRequirements returned unexpected error, expected: %s, got %v", expErr, err)
			}
		})
	}
}

func TestPrintConfiguration(t *testing.T) {
	var tests = []struct {
		name          string
		cfg           *kubeadmapi.ClusterConfiguration
		buf           *bytes.Buffer
		expectedBytes []byte
	}{
		{
			name:          "config is nil",
			cfg:           nil,
			expectedBytes: []byte(""),
		},
		{
			name: "cluster config with local Etcd",
			cfg: &kubeadmapi.ClusterConfiguration{
				KubernetesVersion: "v1.7.1",
				Etcd: kubeadmapi.Etcd{
					Local: &kubeadmapi.LocalEtcd{
						DataDir: "/some/path",
					},
				},
			},
			expectedBytes: []byte(fmt.Sprintf(`[upgrade/config] Configuration used:
	apiServer: {}
	apiVersion: %s
	controllerManager: {}
	dns: {}
	etcd:
	  local:
	    dataDir: /some/path
	kind: ClusterConfiguration
	kubernetesVersion: v1.7.1
	networking: {}
	proxy: {}
	scheduler: {}
`, kubeadmapiv1.SchemeGroupVersion.String())),
		},
		{
			name: "cluster config with ServiceSubnet and external Etcd",
			cfg: &kubeadmapi.ClusterConfiguration{
				KubernetesVersion: "v1.7.1",
				Networking: kubeadmapi.Networking{
					ServiceSubnet: "10.96.0.1/12",
				},
				Etcd: kubeadmapi.Etcd{
					External: &kubeadmapi.ExternalEtcd{
						Endpoints: []string{"https://one-etcd-instance:2379"},
					},
				},
			},
			expectedBytes: []byte(`[upgrade/config] Configuration used:
	apiServer: {}
	apiVersion: ` + kubeadmapiv1.SchemeGroupVersion.String() + `
	controllerManager: {}
	dns: {}
	etcd:
	  external:
	    caFile: ""
	    certFile: ""
	    endpoints:
	    - https://one-etcd-instance:2379
	    keyFile: ""
	kind: ClusterConfiguration
	kubernetesVersion: v1.7.1
	networking:
	  serviceSubnet: 10.96.0.1/12
	proxy: {}
	scheduler: {}
`),
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			rt.buf = bytes.NewBufferString("")
			printConfiguration(rt.cfg, rt.buf, &output.TextPrinter{})
			actualBytes := rt.buf.Bytes()
			if !bytes.Equal(actualBytes, rt.expectedBytes) {
				t.Errorf(
					"failed PrintConfiguration:\n\texpected: %q\n\t  actual: %q",
					string(rt.expectedBytes),
					string(actualBytes),
				)
			}
		})
	}
}
