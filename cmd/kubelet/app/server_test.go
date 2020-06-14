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
	"fmt"
	"io/ioutil"
	"k8s.io/kubernetes/cmd/kubelet/app/options"
	kubeletconfiginternal "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"os"
	"path/filepath"
	"testing"
)

const addressFlag = "127.0.0.1"
const addressShared = "0.0.0.0"
const addressInstance = "192.168.0.1"

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

func TestMergeInstanceConfiguration(t *testing.T) {
	tmpDir, err := ioutil.TempDir("", "kubelet")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	testCases := []struct {
		name                    string
		flags                   []string
		sharedAddress           string
		expectedAddress         string
		instanceAddress         string
		expectedInstanceAddress string
	}{
		{
			name:                    "default value",
			flags:                   []string{},
			expectedAddress:         addressShared,
			expectedInstanceAddress: addressShared,
		},
		{
			name:                    "flags set overwrite",
			flags:                   []string{fmt.Sprintf("--address=%s", addressFlag)},
			expectedAddress:         addressFlag,
			expectedInstanceAddress: addressFlag,
		},
		{
			name:                    "config with address",
			flags:                   []string{},
			sharedAddress:           addressFlag,
			expectedAddress:         addressFlag,
			expectedInstanceAddress: addressFlag,
		},
		{
			name:                    "config with address and flag overwrite",
			flags:                   []string{fmt.Sprintf("--address=%s", addressFlag)},
			sharedAddress:           addressShared,
			expectedAddress:         addressFlag,
			expectedInstanceAddress: addressFlag,
		},
		{
			name:                    "config instance address",
			flags:                   []string{},
			instanceAddress:         addressInstance,
			expectedAddress:         addressInstance,
			expectedInstanceAddress: addressInstance,
		},
		{
			name:                    "config instance with flags overwrite",
			flags:                   []string{fmt.Sprintf("--address=%s", addressFlag)},
			instanceAddress:         addressInstance,
			expectedAddress:         addressFlag,
			expectedInstanceAddress: addressFlag,
		},
		{
			name:                    "config and config instance with instance overwrite",
			flags:                   []string{},
			sharedAddress:           addressShared,
			instanceAddress:         addressInstance,
			expectedAddress:         addressInstance,
			expectedInstanceAddress: addressInstance,
		},
		{
			name:                    "config and config instance with flag overwrite",
			flags:                   []string{fmt.Sprintf("--address=%s", addressFlag)},
			sharedAddress:           addressShared,
			instanceAddress:         addressInstance,
			expectedAddress:         addressFlag,
			expectedInstanceAddress: addressFlag,
		},
	}
	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			kc, err := loadKubeletConfiguration(tmpDir, test.sharedAddress)
			if err != nil {
				t.Errorf("%s: unexpected error: %v", test.name, err)
			}

			configInstanceFile, err := writeKubeletInstance(tmpDir, test.instanceAddress)
			if err != nil {
				t.Errorf("%s: unexpected error: %v", test.name, err)
			}

			kc, kci, err := mergeInstanceConfiguration(kc, configInstanceFile, test.flags)
			if err != nil {
				t.Errorf("%s: unexpected error: %v", test.name, err)
			}

			if kc.Address != test.expectedAddress && kci.Address != test.expectedInstanceAddress {
				t.Errorf("%s: different address: %v and %v", test.name, kc.Address, kci.Address)
			}
		})
	}
}

func loadKubeletConfiguration(tmpDir, address string) (*kubeletconfiginternal.KubeletConfiguration, error) {
	configFile := filepath.Join(tmpDir, "kubelet.config")
	if err := writeConfigContent(configFile, address, "KubeletConfiguration"); err != nil {
		return nil, err
	}

	if address != "" {
		kc, err := loadConfigFile(configFile)
		if err != nil {
			return nil, err
		}
		return kc, nil
	}

	kc, err := options.NewKubeletConfiguration()
	if err != nil {
		return nil, err
	}
	return kc, nil
}

func writeKubeletInstance(tmpDir, address string) (string, error) {
	if address == "" {
		return "", nil
	}

	configFile := filepath.Join(tmpDir, "kubelet-instance.config")
	if err := writeConfigContent(configFile, address, "KubeletInstanceConfiguration"); err != nil {
		return "", err
	}

	return configFile, nil
}

func writeConfigContent(file, address, kind string) error {
	err := ioutil.WriteFile(file, []byte(fmt.Sprintf(`kind: %s
apiVersion: kubelet.config.k8s.io/v1beta1
address: %v`, kind, address)), os.FileMode(0600))
	if err != nil {
		return err
	}
	return nil
}
