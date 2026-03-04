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

package e2enode

import (
	"context"
	"fmt"
	"os/exec"
	"path/filepath"
	"time"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeletpullmanager "k8s.io/kubernetes/pkg/kubelet/images/pullmanager"

	"k8s.io/kubernetes/test/e2e/framework"
	e2enodekubelet "k8s.io/kubernetes/test/e2e_node/kubeletconfig"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

// Returns the current KubeletConfiguration
func getCurrentKubeletConfig(ctx context.Context) (*kubeletconfig.KubeletConfiguration, error) {
	// namespace only relevant if useProxy==true, so we don't bother
	return e2enodekubelet.GetCurrentKubeletConfig(ctx, framework.TestContext.NodeName, "", false, framework.TestContext.StandaloneMode)
}

// Must be called within a Context. Allows the function to modify the KubeletConfiguration during the BeforeEach of the context.
// The change is reverted in the AfterEach of the context.
func tempSetCurrentKubeletConfig(f *framework.Framework, updateFunction func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration)) {
	var oldCfg *kubeletconfig.KubeletConfiguration

	ginkgo.BeforeEach(func(ctx context.Context) {
		var err error
		oldCfg, err = getCurrentKubeletConfig(ctx)
		framework.ExpectNoError(err)

		newCfg := oldCfg.DeepCopy()
		updateFunction(ctx, newCfg)
		if apiequality.Semantic.DeepEqual(*newCfg, *oldCfg) {
			return
		}

		updateKubeletConfig(ctx, f, newCfg, true)
	})

	ginkgo.AfterEach(func(ctx context.Context) {
		if oldCfg != nil {
			// Update the Kubelet configuration.
			updateKubeletConfig(ctx, f, oldCfg, true)
		}
	})
}

type updateKubeletOptions struct {
	deleteStateFiles          bool
	ensureConsistentReadyNode bool
	// TODO: add option to use systemctl stop, now we only use systemctl kill for historical reasons
}

func updateKubeletConfigWithOptions(ctx context.Context, f *framework.Framework, kubeletConfig *kubeletconfig.KubeletConfiguration, opts updateKubeletOptions) {
	ginkgo.GinkgoHelper()

	withStoppedKubelet(ctx, f, opts.ensureConsistentReadyNode, func() {
		// Delete CPU and memory manager state files to be sure it will not prevent the kubelet restart
		if opts.deleteStateFiles {
			deleteStateFile(cpuManagerStateFile)
			deleteStateFile(memoryManagerStateFile)
		}

		framework.ExpectNoError(e2enodekubelet.WriteKubeletConfigFile(kubeletConfig))
	})

}

// withStoppedKubelet stops the kubelet, runs the `action`, and starts the kubelet again
func withStoppedKubelet(ctx context.Context, f *framework.Framework, ensureConsistentReadyNode bool, action func()) {
	ginkgo.GinkgoHelper()

	kubeletStart := mustStopKubelet(ctx, f)
	defer func() {
		kubeletStart(ctx)
		if ensureConsistentReadyNode {
			gomega.Consistently(ctx, func(ctx context.Context) bool {
				return getNodeReadyStatus(ctx, f) && kubeletHealthCheck(kubeletHealthCheckURL)
			}).WithTimeout(2 * time.Minute).WithPolling(2 * time.Second).Should(gomega.BeTrueBecause("node keeps reporting ready status"))
		}
	}()

	action()
}

func tempRemoveImagePulledRecord(f *framework.Framework, imageID *string) {
	ginkgo.BeforeEach(func(ctx context.Context) {
		pulledRecordsDir := "/var/lib/kubelet/image_manager/pulled"
		fileName := kubeletpullmanager.CacheFilename(*imageID)
		origPath := filepath.Join(pulledRecordsDir, fileName)
		tempPath := filepath.Join(pulledRecordsDir, "temp_"+fileName)

		withStoppedKubelet(ctx, f, false, func() {
			err := exec.Command("/bin/sh", "-c", fmt.Sprintf("mv %s %s", origPath, tempPath)).Run()
			framework.ExpectNoError(err, "failed to delete the state file")

			ginkgo.DeferCleanup(func(ctx context.Context) {
				withStoppedKubelet(ctx, f, false, func() {
					err := exec.Command("/bin/sh", "-c", fmt.Sprintf("mv %s %s", tempPath, origPath)).Run()
					framework.ExpectNoError(err, "failed to delete the state file")
				})
			})
		})
	})
}

func updateKubeletConfig(ctx context.Context, f *framework.Framework, kubeletConfig *kubeletconfig.KubeletConfiguration, deleteStateFiles bool) {
	updateKubeletConfigWithOptions(ctx, f, kubeletConfig, updateKubeletOptions{
		deleteStateFiles: deleteStateFiles,
	})
}

func getNodeReadyStatus(ctx context.Context, f *framework.Framework) bool {
	ginkgo.GinkgoHelper()
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err)
	// Assuming that there is only one node, because this is a node e2e test.
	gomega.Expect(nodeList.Items).To(gomega.HaveLen(1), "the number of nodes is not as expected")
	return isNodeReady(&nodeList.Items[0])
}

// isNodeReady returns true if a node is ready; false otherwise.
func isNodeReady(node *v1.Node) bool {
	for _, c := range node.Status.Conditions {
		if c.Type == v1.NodeReady {
			return c.Status == v1.ConditionTrue
		}
	}
	return false
}
