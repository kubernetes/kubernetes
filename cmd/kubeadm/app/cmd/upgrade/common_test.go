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
	"io/ioutil"
	"os"
	"testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func TestGetK8sVersionFromUserInput(t *testing.T) {
	currentVersion := "v" + constants.CurrentKubernetesVersion.String()
	validConfig := "apiVersion: kubeadm.k8s.io/v1beta2\n" +
		"kind: ClusterConfiguration\n" +
		"kubernetesVersion: " + currentVersion

	var tcases = []struct {
		name               string
		isVersionMandatory bool
		clusterConfig      string
		args               []string
		expectedErr        bool
		expectedVersion    string
	}{
		{
			name:               "No config and version as an argument",
			isVersionMandatory: true,
			args:               []string{"v1.13.1"},
			expectedVersion:    "v1.13.1",
		},
		{
			name:               "Neither config nor version specified",
			isVersionMandatory: true,
			expectedErr:        true,
		},
		{
			name:               "No config and empty version as an argument",
			isVersionMandatory: true,
			args:               []string{""},
			expectedErr:        true,
		},
		{
			name:               "Valid config, but no version specified",
			isVersionMandatory: true,
			clusterConfig:      validConfig,
			expectedVersion:    currentVersion,
		},
		{
			name:               "Valid config and different version specified",
			isVersionMandatory: true,
			clusterConfig:      validConfig,
			args:               []string{"v1.13.1"},
			expectedVersion:    "v1.13.1",
		},
		{
			name: "Version is optional",
		},
	}
	for _, tt := range tcases {
		t.Run(tt.name, func(t *testing.T) {
			flags := &applyPlanFlags{}
			if len(tt.clusterConfig) > 0 {
				file, err := ioutil.TempFile("", "kubeadm-upgrade-common-test-*.yaml")
				if err != nil {
					t.Fatalf("Failed to create test config file: %+v", err)
				}

				tmpFileName := file.Name()
				defer os.Remove(tmpFileName)

				_, err = file.WriteString(tt.clusterConfig)
				file.Close()
				if err != nil {
					t.Fatalf("Failed to write test config file contents: %+v", err)
				}

				flags.cfgPath = tmpFileName
			}

			userVersion, err := getK8sVersionFromUserInput(flags, tt.args, tt.isVersionMandatory)

			if err == nil && tt.expectedErr {
				t.Error("Expected error, but got success")
			}
			if err != nil && !tt.expectedErr {
				t.Errorf("Unexpected error: %+v", err)
			}
			if userVersion != tt.expectedVersion {
				t.Errorf("Expected %q, but got %q", tt.expectedVersion, userVersion)
			}
		})
	}
}

func TestEnforceRequirements(t *testing.T) {
	tcases := []struct {
		name          string
		newK8sVersion string
		dryRun        bool
		flags         applyPlanFlags
		expectedErr   bool
	}{
		{
			name:        "Fail pre-flight check",
			expectedErr: true,
		},
		{
			name: "Bogus preflight check disabled when also 'all' is specified",
			flags: applyPlanFlags{
				ignorePreflightErrors: []string{"bogusvalue", "all"},
			},
			expectedErr: true,
		},
		{
			name: "Fail to create client",
			flags: applyPlanFlags{
				ignorePreflightErrors: []string{"all"},
			},
			expectedErr: true,
		},
	}
	for _, tt := range tcases {
		t.Run(tt.name, func(t *testing.T) {
			_, _, _, err := enforceRequirements(&tt.flags, tt.dryRun, tt.newK8sVersion)

			if err == nil && tt.expectedErr {
				t.Error("Expected error, but got success")
			}
			if err != nil && !tt.expectedErr {
				t.Errorf("Unexpected error: %+v", err)
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
				DNS: kubeadmapi.DNS{
					Type: kubeadmapi.CoreDNS,
				},
			},
			expectedBytes: []byte(`[upgrade/config] Configuration used:
	apiServer: {}
	apiVersion: kubeadm.k8s.io/v1beta2
	controllerManager: {}
	dns:
	  type: CoreDNS
	etcd:
	  local:
	    dataDir: /some/path
	kind: ClusterConfiguration
	kubernetesVersion: v1.7.1
	networking: {}
	scheduler: {}
`),
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
				DNS: kubeadmapi.DNS{
					Type: kubeadmapi.CoreDNS,
				},
			},
			expectedBytes: []byte(`[upgrade/config] Configuration used:
	apiServer: {}
	apiVersion: kubeadm.k8s.io/v1beta2
	controllerManager: {}
	dns:
	  type: CoreDNS
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
	scheduler: {}
`),
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			rt.buf = bytes.NewBufferString("")
			printConfiguration(rt.cfg, rt.buf)
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
