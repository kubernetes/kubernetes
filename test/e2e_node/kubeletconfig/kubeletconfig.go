/*
Copyright 2021 The Kubernetes Authors.

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

package kubeletconfig

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"

	kubeletconfigv1beta1 "k8s.io/kubelet/config/v1beta1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeletconfigscheme "k8s.io/kubernetes/pkg/kubelet/apis/config/scheme"
	kubeletconfigcodec "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/codec"

	"sigs.k8s.io/yaml"
)

func getKubeletConfigFilePath() (string, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("failed to get current working directory: %w", err)
	}

	// DO NOT name this file "kubelet" - you will overwrite the kubelet binary and be very confused :)
	return filepath.Join(cwd, "kubelet-config"), nil
}

// GetCurrentKubeletConfigFromFile returns the current kubelet configuration under the filesystem.
// This method should only run together with e2e node tests, meaning the test executor and the cluster nodes is the
// same machine
func GetCurrentKubeletConfigFromFile() (*kubeletconfig.KubeletConfiguration, error) {
	kubeletConfigFilePath, err := getKubeletConfigFilePath()
	if err != nil {
		return nil, err
	}

	data, err := ioutil.ReadFile(kubeletConfigFilePath)
	if err != nil {
		return nil, fmt.Errorf("failed to get the kubelet config from the file %q: %w", kubeletConfigFilePath, err)
	}

	var kubeletConfigV1Beta1 kubeletconfigv1beta1.KubeletConfiguration
	if err := yaml.Unmarshal(data, &kubeletConfigV1Beta1); err != nil {
		return nil, fmt.Errorf("failed to unmarshal the kubelet config: %w", err)
	}

	scheme, _, err := kubeletconfigscheme.NewSchemeAndCodecs()
	if err != nil {
		return nil, err
	}

	kubeletConfig := kubeletconfig.KubeletConfiguration{}
	err = scheme.Convert(&kubeletConfigV1Beta1, &kubeletConfig, nil)
	if err != nil {
		return nil, err
	}

	return &kubeletConfig, nil
}

// WriteKubeletConfigFile updates the kubelet configuration under the filesystem
// This method should only run together with e2e node tests, meaning the test executor and the cluster nodes is the
// same machine
func WriteKubeletConfigFile(kubeletConfig *kubeletconfig.KubeletConfiguration) error {
	data, err := kubeletconfigcodec.EncodeKubeletConfig(kubeletConfig, kubeletconfigv1beta1.SchemeGroupVersion)
	if err != nil {
		return err
	}

	kubeletConfigFilePath, err := getKubeletConfigFilePath()
	if err != nil {
		return err
	}

	if err := ioutil.WriteFile(kubeletConfigFilePath, data, 0644); err != nil {
		return fmt.Errorf("failed to write the kubelet file to %q: %w", kubeletConfigFilePath, err)
	}

	return nil
}
