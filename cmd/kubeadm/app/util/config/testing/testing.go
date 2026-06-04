/*
Copyright The Kubernetes Authors.

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

// Package testing contains utilities for managing kubeadm configuration in tests.
package testing

import (
	"testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/config"
)

// GetDefaultInternalConfig returns a defaulted kubeadmapi.InitConfiguration
func GetDefaultInternalConfig(t *testing.T) *kubeadmapi.InitConfiguration {
	internalcfg, err := config.DefaultedStaticInitConfiguration()
	if err != nil {
		t.Fatalf("unexpected error getting default config: %v", err)
	}
	return internalcfg
}
