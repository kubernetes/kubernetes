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
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
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

	nodeIdent := identifyNode(getLocalNode(ctx, f))

	// Update the Kubelet configuration.
	ginkgo.By("Stopping the kubelet on " + nodeIdent)
	restartKubelet := mustStopKubelet(ctx, f)

	// Delete CPU and memory manager state files to be sure it will not prevent the kubelet restart
	if opts.deleteStateFiles {
		ginkgo.By("Deleting the kubelet state files on " + nodeIdent)
		deleteStateFile(cpuManagerStateFile)
		deleteStateFile(memoryManagerStateFile)
	}

	framework.ExpectNoError(e2enodekubelet.WriteKubeletConfigFile(kubeletConfig))

	ginkgo.By("Restarting the kubelet on " + nodeIdent)
	restartKubelet(ctx)

	if opts.ensureConsistentReadyNode {
		gomega.Consistently(ctx, func(ctx context.Context) bool {
			return getNodeReadyStatus(ctx, f) && kubeletHealthCheck(kubeletHealthCheckURL)
		}).WithTimeout(2 * time.Minute).WithPolling(2 * time.Second).Should(gomega.BeTrueBecause("node keeps reporting ready status"))
	}
}

func identifyNode(node *v1.Node) string {
	if node == nil {
		return "localhost"
	}
	var addrs string
	if len(node.Status.Addresses) > 0 {
		var sb strings.Builder
		for _, addr := range node.Status.Addresses {
			fmt.Fprintf(&sb, " %v=%v", addr.Type, addr.Address)
		}
		addrs = " <" + sb.String()[1:] + ">"
	}
	return node.Name + addrs
}

func updateKubeletConfig(ctx context.Context, f *framework.Framework, kubeletConfig *kubeletconfig.KubeletConfiguration, deleteStateFiles bool) {
	updateKubeletConfigWithOptions(ctx, f, kubeletConfig, updateKubeletOptions{
		deleteStateFiles: deleteStateFiles,
	})
}
