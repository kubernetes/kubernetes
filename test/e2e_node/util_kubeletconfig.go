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
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"time"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"

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
			deleteStateFile(usernsStateFiles)
		}

		framework.ExpectNoError(e2enodekubelet.WriteKubeletConfigFile(kubeletConfig))
	})

}

func updateKubeletConfig(ctx context.Context, f *framework.Framework, kubeletConfig *kubeletconfig.KubeletConfiguration, deleteStateFiles bool) {
	updateKubeletConfigWithOptions(ctx, f, kubeletConfig, updateKubeletOptions{
		deleteStateFiles: deleteStateFiles,
	})
}

const pulledRecordsDir = "/var/lib/kubelet/image_manager/pulled"

func pulledRecordFilenameFromID(imageID string) string {
	return fmt.Sprintf("sha256-%x", sha256.Sum256([]byte(imageID)))
}

func tempRemoveImagePulledRecord(f *framework.Framework, imageID *string) {
	tempRemoveImagePulledRecordAndSetKubeletConfig(f, imageID, nil)
}

// tempRemoveImagePulledRecordAndSetKubeletConfig removes the image pulled record file and
// optionally updates the kubelet configuration in a single kubelet stop/start cycle.
// This avoids an extra kubelet restart when both operations are needed.
//
// When updateFunction is non-nil, an explicit AfterEach is used for config
// restoration so it runs in the first cleanup pass alongside the outer
// tempSetCurrentKubeletConfig's AfterEach (ordered by descending nesting level).
// DeferCleanup runs in a second pass and would clobber the outer's
// already-restored config.
func tempRemoveImagePulledRecordAndSetKubeletConfig(f *framework.Framework, imageID *string, updateFunction func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration)) {
	var oldCfg *kubeletconfig.KubeletConfiguration

	if updateFunction != nil {
		ginkgo.AfterEach(func(ctx context.Context) {
			if oldCfg != nil {
				updateKubeletConfig(ctx, f, oldCfg, true)
			}
		})
	}

	ginkgo.BeforeEach(func(ctx context.Context) {
		fileName := pulledRecordFilenameFromID(*imageID)
		origPath := filepath.Join(pulledRecordsDir, fileName)
		tempPath := filepath.Join(pulledRecordsDir, "temp_"+fileName)

		// Wait for the kubelet to create the record. Use a generous timeout because
		// the preceding kubelet restart (from tempSetCurrentKubeletConfig) can be slow
		// on loaded CI nodes, and the file must exist on disk before we can rename it.
		err := wait.PollUntilContextTimeout(ctx, time.Second, 30*time.Second, true, func(_ context.Context) (bool, error) {
			if _, err := os.Stat(origPath); err != nil {
				return false, nil
			}
			return true, nil
		})
		framework.ExpectNoError(err, "the file with the pulled record for the image ID never appeared")

		if updateFunction != nil {
			oldCfg, err = getCurrentKubeletConfig(ctx)
			framework.ExpectNoError(err)
		}

		withStoppedKubelet(ctx, f, false, func() {
			err := os.Rename(origPath, tempPath)
			framework.ExpectNoError(err, "failed to move the ImagePulledRecord file")

			if updateFunction != nil {
				newCfg := oldCfg.DeepCopy()
				updateFunction(ctx, newCfg)

				deleteStateFile(cpuManagerStateFile)
				deleteStateFile(memoryManagerStateFile)
				deleteStateFile(usernsStateFiles)
				framework.ExpectNoError(e2enodekubelet.WriteKubeletConfigFile(newCfg))
			}

			// Restore the file on cleanup. Config restoration (when applicable) is
			// handled by the AfterEach above to maintain correct Ginkgo cleanup ordering.
			ginkgo.DeferCleanup(func(ctx context.Context) {
				withStoppedKubelet(ctx, f, false, func() {
					err := os.Rename(tempPath, origPath)
					framework.ExpectNoError(err, "failed to move the ImagePulledRecord file back to its original location")
				})
			})
		})
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
