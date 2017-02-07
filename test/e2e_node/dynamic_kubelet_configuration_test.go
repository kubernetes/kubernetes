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

package e2e_node

import (
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

// This test is marked [Disruptive] because the Kubelet temporarily goes down as part of of this test.
var _ = framework.KubeDescribe("DynamicKubeletConfiguration [Feature:DynamicKubeletConfig] [Serial] [Disruptive]", func() {
	f := framework.NewDefaultFramework("dynamic-kubelet-configuration-test")

	Context("When a configmap called `kubelet-{node-name}` with a different file check frequency is added to the `kube-system` namespace", func() {
		tempSetCurrentKubeletConfig(f, func(kubeCfg *componentconfig.KubeletConfiguration) {
			glog.Infof("KubeletConfiguration - Initial values: %+v", *kubeCfg)
			// Change a safe value e.g. file check frequency.
			// Make sure we're providing a value distinct from the current one.
			newFileCheckFrequency := 11 * time.Second
			if kubeCfg.FileCheckFrequency.Duration == newFileCheckFrequency {
				newFileCheckFrequency = 10 * time.Second
			}
			kubeCfg.FileCheckFrequency.Duration = newFileCheckFrequency
		})
		// Dummy It() so that the BeforeEach and AfterEach used by tempSetCurrentKubeletConfig are run
		It("the Kubelet on that node should restart use up the new config", func() {})
	})

	Context("When a configmap called `kubelet-{node-name}` that sets an experimental field (FailSwapOn) is added to the `kube-system` namespace", func() {
		tempSetCurrentKubeletConfig(f, func(kubeCfg *componentconfig.KubeletConfiguration) {
			glog.Infof("KubeletConfiguration - Initial values: %+v", *kubeCfg)
			ekc, err := componentconfig.ExperimentalKubeletConfigurationFromString(kubeCfg.Experimental)
			framework.ExpectNoError(err)
			// Toggle a not-so-dangerous value (FailSwapOn)
			ekc.FailSwapOn = !ekc.FailSwapOn
			// Serialize back into kubeCfg.Experimental
			s, err := componentconfig.ExperimentalKubeletConfigurationToString(ekc)
			framework.ExpectNoError(err)
			kubeCfg.Experimental = s
		})
		// Dummy It() so that the BeforeEach and AfterEach used by tempSetCurrentKubeletConfig are run
		It("the Kubelet on that node should restart and use the new config", func() {})
	})
})
