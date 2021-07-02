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

package auth

import (
	"testing"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/capabilities"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	podsecuritytest "k8s.io/pod-security-admission/test"
)

func TestPodSecurity(t *testing.T) {
	// Enable all feature gates needed to allow all fields to be exercised
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ProcMountType, true)()
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WindowsHostProcessContainers, true)()
	// Ensure the PodSecurity feature is enabled
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodSecurity, true)()
	// Start server
	server := startPodSecurityServer(t)
	opts := podsecuritytest.Options{
		ClientConfig: server.ClientConfig,

		// Don't pass in feature-gate info, so all testcases run

		// TODO
		ExemptClient:         nil,
		ExemptNamespaces:     []string{},
		ExemptRuntimeClasses: []string{},
	}
	podsecuritytest.Run(t, opts)
}

// TestPodSecurityGAOnly ensures policies pass with only GA features enabled
func TestPodSecurityGAOnly(t *testing.T) {
	// Disable all alpha and beta features
	for k, v := range utilfeature.DefaultFeatureGate.DeepCopy().GetAll() {
		if v.PreRelease == featuregate.Alpha || v.PreRelease == featuregate.Beta {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, k, false)()
		}
	}
	// Ensure PodSecurity feature is enabled
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodSecurity, true)()
	// Start server
	server := startPodSecurityServer(t)

	opts := podsecuritytest.Options{
		ClientConfig: server.ClientConfig,
		// Pass in feature gate info so negative test cases depending on alpha or beta features can be skipped
		Features: utilfeature.DefaultFeatureGate,
	}
	podsecuritytest.Run(t, opts)
}

func startPodSecurityServer(t *testing.T) *kubeapiservertesting.TestServer {
	// ensure the global is set to allow privileged containers
	capabilities.SetForTests(capabilities.Capabilities{AllowPrivileged: true})

	server := kubeapiservertesting.StartTestServerOrDie(t, kubeapiservertesting.NewDefaultTestServerOptions(), []string{
		"--anonymous-auth=false",
		"--enable-admission-plugins=PodSecurity",
		"--allow-privileged=true",
		// TODO: "--admission-control-config-file=" + admissionConfigFile.Name(),
	}, framework.SharedEtcd())
	t.Cleanup(server.TearDownFn)
	return server
}
