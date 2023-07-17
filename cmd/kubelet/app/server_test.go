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

package app

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/kubernetes/cmd/kubelet/app/options"
	kubeletconfiginternal "k8s.io/kubernetes/pkg/kubelet/apis/config"
)

func TestValueOfAllocatableResources(t *testing.T) {
	testCases := []struct {
		kubeReserved   map[string]string
		systemReserved map[string]string
		errorExpected  bool
		name           string
	}{
		{
			kubeReserved:   map[string]string{"cpu": "200m", "memory": "-150G", "ephemeral-storage": "10Gi"},
			systemReserved: map[string]string{"cpu": "200m", "memory": "15Ki"},
			errorExpected:  true,
			name:           "negative quantity value",
		},
		{
			kubeReserved:   map[string]string{"cpu": "200m", "memory": "150Gi", "ephemeral-storage": "10Gi"},
			systemReserved: map[string]string{"cpu": "200m", "memory": "15Ky"},
			errorExpected:  true,
			name:           "invalid quantity unit",
		},
		{
			kubeReserved:   map[string]string{"cpu": "200m", "memory": "15G", "ephemeral-storage": "10Gi"},
			systemReserved: map[string]string{"cpu": "200m", "memory": "15Ki"},
			errorExpected:  false,
			name:           "Valid resource quantity",
		},
	}

	for _, test := range testCases {
		_, err1 := parseResourceList(test.kubeReserved)
		_, err2 := parseResourceList(test.systemReserved)
		if test.errorExpected {
			if err1 == nil && err2 == nil {
				t.Errorf("%s: error expected", test.name)
			}
		} else {
			if err1 != nil || err2 != nil {
				t.Errorf("%s: unexpected error: %v, %v", test.name, err1, err2)
			}
		}
	}
}

func TestMergeKubeletConfigurations(t *testing.T) {
	testCases := []struct {
		kubeletConfig           string
		dropin1                 string
		dropin2                 string
		overwrittenConfigFields map[string]interface{}
		cliArgs                 []string
		name                    string
	}{
		{
			kubeletConfig: `
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
port: 9080
readOnlyPort: 10257
`,
			dropin1: `
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
port: 9090
`,
			dropin2: `
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
port: 8080
readOnlyPort: 10255
`,
			overwrittenConfigFields: map[string]interface{}{
				"Port":         int32(8080),
				"ReadOnlyPort": int32(10255),
			},
			name: "kubelet.conf.d overrides kubelet.conf",
		},
		{
			kubeletConfig: `
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
readOnlyPort: 10256
kubeReserved:
	memory: 70Mi
`,
			dropin1: `
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
readOnlyPort: 10255
kubeReserved:
  memory: 150Mi
  cpu: 200m
`,
			dropin2: `
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
readOnlyPort: 10257
kubeReserved:
  memory: 100Mi
`,
			overwrittenConfigFields: map[string]interface{}{
				"ReadOnlyPort": int32(10257),
				"KubeReserved": map[string]string{
					"cpu":    "200m",
					"memory": "100Mi",
				},
			},
			name: "kubelet.conf.d overrides kubelet.conf with subfield override",
		},
		{
			kubeletConfig: `
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
port: 9090
clusterDNS:
	- 192.168.1.3
	- 192.168.1.4
`,
			dropin1: `
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
port: 9090
systemReserved:
  memory: 1Gi
`,
			dropin2: `
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
port: 8080
readOnlyPort: 10255
systemReserved:
  memory: 2Gi
clusterDNS:
  - 192.168.1.1
  - 192.168.1.5
  - 192.168.1.8
`,
			overwrittenConfigFields: map[string]interface{}{
				"Port":         int32(8080),
				"ReadOnlyPort": int32(10255),
				"SystemReserved": map[string]string{
					"memory": "2Gi",
				},
				"ClusterDNS": []string{"192.168.1.1", "192.168.1.5", "192.168.1.8"},
			},
			name: "kubelet.conf.d overrides kubelet.conf with slices/lists",
		},
		{
			dropin1: `
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
port: 9090
`,
			dropin2: `
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
port: 8080
readOnlyPort: 10255
`,
			overwrittenConfigFields: map[string]interface{}{
				"Port":         int32(8081),
				"ReadOnlyPort": int32(10256),
			},
			cliArgs: []string{
				"--port=8081",
				"--read-only-port=10256",
			},
			name: "cli args override kubelet.conf.d",
		},
		{
			kubeletConfig: `
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
port: 9090
clusterDNS:
	- 192.168.1.3
`,
			overwrittenConfigFields: map[string]interface{}{
				"Port":       int32(9090),
				"ClusterDNS": []string{"192.168.1.2"},
			},
			cliArgs: []string{
				"--port=9090",
				"--cluster-dns=192.168.1.2",
			},
			name: "cli args override kubelet.conf",
		},
	}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			// Prepare a temporary directory for testing
			tempDir := t.TempDir()

			kubeletConfig := &kubeletconfiginternal.KubeletConfiguration{}
			kubeletFlags := &options.KubeletFlags{}

			if len(test.kubeletConfig) > 0 {
				// Create the Kubeletconfig
				kubeletConfFile := filepath.Join(tempDir, "kubelet.conf")
				err := os.WriteFile(kubeletConfFile, []byte(test.kubeletConfig), 0644)
				require.NoError(t, err, "failed to create config from a yaml file")
				kubeletFlags.KubeletConfigFile = kubeletConfFile
			}
			if len(test.dropin1) > 0 || len(test.dropin2) > 0 {
				// Create kubelet.conf.d directory and drop-in configuration files
				kubeletConfDir := filepath.Join(tempDir, "kubelet.conf.d")
				err := os.Mkdir(kubeletConfDir, 0755)
				require.NoError(t, err, "Failed to create kubelet.conf.d directory")

				err = os.WriteFile(filepath.Join(kubeletConfDir, "10-kubelet.conf"), []byte(test.dropin1), 0644)
				require.NoError(t, err, "failed to create config from a yaml file")

				err = os.WriteFile(filepath.Join(kubeletConfDir, "20-kubelet.conf"), []byte(test.dropin2), 0644)
				require.NoError(t, err, "failed to create config from a yaml file")

				// Merge the kubelet configurations
				err = mergeKubeletConfigurations(kubeletConfig, kubeletConfDir)
				require.NoError(t, err, "failed to merge kubelet drop-in configs")
			}

			// Use kubelet config flag precedence
			err := kubeletConfigFlagPrecedence(kubeletConfig, test.cliArgs)
			require.NoError(t, err, "failed to set the kubelet config flag precedence")

			// Verify the merged configuration fields
			for fieldName, expectedValue := range test.overwrittenConfigFields {
				value := reflect.ValueOf(kubeletConfig).Elem()
				field := value.FieldByName(fieldName)
				require.Equal(t, expectedValue, field.Interface(), "Field mismatch: "+fieldName)
			}
		})
	}
}
