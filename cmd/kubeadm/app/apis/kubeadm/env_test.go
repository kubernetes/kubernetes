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

package kubeadm

import "testing"

func TestSetEnvParams(t *testing.T) {
	oldEnv := GlobalEnvParams
	GlobalEnvParams = &EnvParams{}
	GlobalEnvParams = SetEnvParams()

	// verify that it gets back to its defaults after being changed to nil
	if oldEnv.KubernetesDir != GlobalEnvParams.KubernetesDir {
		t.Errorf(
			"failed KubernetesDir:\n\texpected: %s\n\t  actual: %s",
			oldEnv.KubernetesDir,
			GlobalEnvParams.KubernetesDir,
		)
	}
	if oldEnv.HostPKIPath != GlobalEnvParams.HostPKIPath {
		t.Errorf(
			"failed HostPKIPath:\n\texpected: %s\n\t  actual: %s",
			oldEnv.HostPKIPath,
			GlobalEnvParams.HostPKIPath,
		)
	}
	if oldEnv.HostEtcdPath != GlobalEnvParams.HostEtcdPath {
		t.Errorf(
			"failed HostEtcdPath:\n\texpected: %s\n\t  actual: %s",
			oldEnv.HostEtcdPath,
			GlobalEnvParams.HostEtcdPath,
		)
	}
	if oldEnv.HyperkubeImage != GlobalEnvParams.HyperkubeImage {
		t.Errorf(
			"failed HyperkubeImage:\n\texpected: %s\n\t  actual: %s",
			oldEnv.HyperkubeImage,
			GlobalEnvParams.HyperkubeImage,
		)
	}
	if oldEnv.DiscoveryImage != GlobalEnvParams.DiscoveryImage {
		t.Errorf(
			"failed DiscoveryImage:\n\texpected: %s\n\t  actual: %s",
			oldEnv.DiscoveryImage,
			GlobalEnvParams.DiscoveryImage,
		)
	}
	if oldEnv.EtcdImage != GlobalEnvParams.EtcdImage {
		t.Errorf(
			"failed EtcdImage:\n\texpected: %s\n\t  actual: %s",
			oldEnv.EtcdImage,
			GlobalEnvParams.EtcdImage,
		)
	}
	if oldEnv.ComponentLoglevel != GlobalEnvParams.ComponentLoglevel {
		t.Errorf(
			"failed ComponentLoglevel:\n\texpected: %s\n\t  actual: %s",
			oldEnv.ComponentLoglevel,
			GlobalEnvParams.ComponentLoglevel,
		)
	}
	defer func() { GlobalEnvParams = oldEnv }()
}
