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

	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

// This test is marked [Disruptive] because the Kubelet temporarily goes down as part of of this test.
var _ = framework.KubeDescribe("DynamicKubeletConfiguration [Feature:DynamicKubeletConfig] [Serial] [Disruptive]", func() {
	f := framework.NewDefaultFramework("dynamic-kubelet-configuration-test")

	Context("When the config source on a Node is updated to point to new config", func() {
		It("The Kubelet on that node should restart to take up the new config", func() {
			// Get the current KubeletConfiguration (known to be valid) by
			// querying the configz endpoint for the current node.
			kubeCfg, err := getCurrentKubeletConfig()
			framework.ExpectNoError(err)
			glog.Infof("KubeletConfiguration - Initial values: %+v", *kubeCfg)

			// Change a safe value e.g. file check frequency.
			// Make sure we're providing a value distinct from the current one.
			oldFileCheckFrequency := kubeCfg.FileCheckFrequency.Duration
			newFileCheckFrequency := 11 * time.Second
			if kubeCfg.FileCheckFrequency.Duration == newFileCheckFrequency {
				newFileCheckFrequency = 10 * time.Second
			}
			kubeCfg.FileCheckFrequency.Duration = newFileCheckFrequency

			// Use the new config to create a new kube-{node-name} configmap in `kube-system` namespace.
			// Note: setKubeletConfiguration will return an error if the Kubelet does not present the
			//       modified configuration via /configz when it comes back up.
			err = setKubeletConfiguration(f, kubeCfg)
			framework.ExpectNoError(err)

			// Change the config back to what it originally was.
			kubeCfg.FileCheckFrequency.Duration = oldFileCheckFrequency
			err = setKubeletConfiguration(f, kubeCfg)
			framework.ExpectNoError(err)
		})
	})
})
