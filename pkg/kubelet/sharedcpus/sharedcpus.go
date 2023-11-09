/*
Copyright 2023 The Kubernetes Authors.

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

package sharedcpus

import (
	"encoding/json"
	"errors"
	"os"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
)

const (
	configFileName         = "/etc/kubernetes/openshift-workload-mixed-cpus"
	sharedCpusResourceName = "workload.openshift.io/enable-shared-cpus"
)

var (
	config            Config
	sharedCpusEnabled bool
)

type Config struct {
	sharedCpus `json:"shared_cpus"`
}

type sharedCpus struct {
	// ContainersLimit specify the number of containers that are allowed to access the shared CPU pool`
	ContainersLimit int64 `json:"containers_limit"`
}

func init() {
	parseConfig()
}

func IsEnabled() bool {
	return sharedCpusEnabled
}

func GetResourceName() corev1.ResourceName {
	return sharedCpusResourceName
}

func GetConfig() Config {
	return config
}

func parseConfig() {
	b, err := os.ReadFile(configFileName)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return
		}
		klog.ErrorS(err, "Failed to read configuration file for shared cpus", "fileName", configFileName)
		return
	}
	cfg, err := parseConfigData(b)
	if err != nil {
		return
	}
	config = *cfg
	sharedCpusEnabled = true
}

func parseConfigData(data []byte) (*Config, error) {
	cfg := &Config{}
	err := json.Unmarshal(data, cfg)
	if err != nil {
		klog.ErrorS(err, "Failed to parse configuration file for shared cpus", "fileContent", string(data))
	}
	return cfg, err
}
