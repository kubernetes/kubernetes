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

/*
E2E Node test for DRA (Dynamic Resource Allocation)
This test covers node-specific aspects of DRA
The test can be run locally on Linux this way:
  make test-e2e-node FOCUS='\[NodeAlphaFeature:DynamicResourceAllocation\]' SKIP='\[Flaky\]' PARALLELISM=1 \
       TEST_ARGS='--feature-gates="DynamicResourceAllocation=true" --service-feature-gates="DynamicResourceAllocation=true" --runtime-config=api/all=true'
*/

package e2enode

import (
	"context"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	resourcev1alpha2 "k8s.io/api/resource/v1alpha2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	draplugin "k8s.io/kubernetes/pkg/kubelet/cm/dra/plugin"
	admissionapi "k8s.io/pod-security-admission/api"

	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"

	"k8s.io/dynamic-resource-allocation/kubeletplugin"
	testdriver "k8s.io/kubernetes/test/e2e/dra/test-driver/app"
)

const (
	driverName                = "test-driver.cdi.k8s.io"
	kubeletPlugin1Name        = "test-driver1.cdi.k8s.io"
	kubeletPlugin2Name        = "test-driver2.cdi.k8s.io"
	cdiDir                    = "/var/run/cdi"
	endpointTemplate          = "/var/lib/kubelet/plugins/%s/dra.sock"
	pluginRegistrationPath    = "/var/lib/kubelet/plugins_registry"
	pluginRegistrationTimeout = time.Second * 60 // how long to wait for a node plugin to be registered
	podInPendingStateTimeout  = time.Second * 60 // how long to wait for a pod to stay in pending state
)

var _ = framework.SIGDescribe("node")("DRA", feature.DynamicResourceAllocation, "[NodeAlphaFeature:DynamicResourceAllocation]", func() {
	f := framework.NewDefaultFramework("dra-node")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	var kubeletPlugin, kubeletPlugin1, kubeletPlugin2 *testdriver.ExamplePlugin

	f.Context("Resource Kubelet Plugin", f.WithSerial(), func() {
		ginkgo.BeforeEach(func(ctx context.Context) {
			kubeletPlugin = newKubeletPlugin(ctx, getNodeName(ctx, f), driverName)
		})

		ginkgo.It("must register after Kubelet restart", func(ctx context.Context) {
			oldCalls := kubeletPlugin.GetGRPCCalls()
			getNewCalls := func() []testdriver.GRPCCall {
				calls := kubeletPlugin.GetGRPCCalls()
				return calls[len(oldCalls):]
			}

			ginkgo.By("restarting Kubelet")
			restartKubelet(true)

			ginkgo.By("wait for Kubelet plugin re-registration")
			gomega.Eventually(getNewCalls).WithTimeout(pluginRegistrationTimeout).Should(testdriver.BeRegistered)
		})

		ginkgo.It("must register after plugin restart", func(ctx context.Context) {
			ginkgo.By("restart Kubelet Plugin")
			kubeletPlugin.Stop()
			kubeletPlugin = newKubeletPlugin(ctx, getNodeName(ctx, f), driverName)

			ginkgo.By("wait for Kubelet plugin re-registration")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(pluginRegistrationTimeout).Should(testdriver.BeRegistered)
		})

		ginkgo.It("must process pod created when kubelet is not running", func(ctx context.Context) {
			// Stop Kubelet
			startKubelet := stopKubelet()
			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drapod", true, []string{driverName})
			// Pod must be in pending state
			err := e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "Pending", framework.PodStartShortTimeout, func(pod *v1.Pod) (bool, error) {
				return pod.Status.Phase == v1.PodPending, nil
			})
			framework.ExpectNoError(err)
			// Start Kubelet
			startKubelet()
			// Pod should succeed
			err = e2epod.WaitForPodSuccessInNamespaceTimeout(ctx, f.ClientSet, pod.Name, f.Namespace.Name, framework.PodStartShortTimeout)
			framework.ExpectNoError(err)
		})

		ginkgo.It("must keep pod in pending state if NodePrepareResources times out", func(ctx context.Context) {
			unblock := kubeletPlugin.BlockNodePrepareResources()
			defer unblock()
			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drapod", true, []string{driverName})

			ginkgo.By("wait for pod to be in Pending state")
			err := e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "Pending", framework.PodStartShortTimeout, func(pod *v1.Pod) (bool, error) {
				return pod.Status.Phase == v1.PodPending, nil
			})
			framework.ExpectNoError(err)

			// TODO: Check condition or event when implemented
			// see https://github.com/kubernetes/kubernetes/issues/118468 for details
			ginkgo.By("check that pod is consistently in Pending state")
			gomega.Consistently(ctx, e2epod.Get(f.ClientSet, pod)).WithTimeout(podInPendingStateTimeout).Should(e2epod.BeInPhase(v1.PodPending),
				"Pod should be in Pending state as resource preparation time outed")
		})

		ginkgo.It("must run pod if NodePrepareResources fails and then succeeds", func(ctx context.Context) {
			unset := kubeletPlugin.SetNodePrepareResourcesFailureMode()
			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drapod", true, []string{driverName})

			ginkgo.By("wait for pod to be in Pending state")
			err := e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "Pending", framework.PodStartShortTimeout, func(pod *v1.Pod) (bool, error) {
				return pod.Status.Phase == v1.PodPending, nil
			})
			framework.ExpectNoError(err)

			ginkgo.By("wait for NodePrepareResources call to fail")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodePrepareResourcesFailed)

			unset()

			ginkgo.By("wait for NodePrepareResources call to succeed")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodePrepareResourcesSucceeded)

			ginkgo.By("wait for pod to succeed")
			err = e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
			framework.ExpectNoError(err)
		})

		ginkgo.It("must run pod if NodeUnprepareResources fails and then succeeds", func(ctx context.Context) {
			unset := kubeletPlugin.SetNodeUnprepareResourcesFailureMode()
			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drapod", true, []string{driverName})

			ginkgo.By("wait for NodePrepareResources call to succeed")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodePrepareResourcesSucceeded)

			ginkgo.By("wait for NodeUnprepareResources call to fail")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodeUnprepareResourcesFailed)

			unset()

			ginkgo.By("wait for NodeUnprepareResources call to succeed")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodeUnprepareResourcesSucceeded)

			ginkgo.By("wait for pod to succeed")
			err := e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
			framework.ExpectNoError(err)
		})

		ginkgo.It("must retry NodePrepareResources after Kubelet restart", func(ctx context.Context) {
			unset := kubeletPlugin.SetNodePrepareResourcesFailureMode()
			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drapod", true, []string{driverName})

			ginkgo.By("wait for pod to be in Pending state")
			err := e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "Pending", framework.PodStartShortTimeout, func(pod *v1.Pod) (bool, error) {
				return pod.Status.Phase == v1.PodPending, nil
			})
			framework.ExpectNoError(err)

			ginkgo.By("wait for NodePrepareResources call to fail")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodePrepareResourcesFailed)

			ginkgo.By("stop Kubelet")
			startKubelet := stopKubelet()

			unset()

			ginkgo.By("start Kubelet")
			startKubelet()

			ginkgo.By("wait for NodePrepareResources call to succeed")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodePrepareResourcesSucceeded)

			ginkgo.By("wait for pod to succeed")
			err = e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
			framework.ExpectNoError(err)
		})

		ginkgo.It("must retry NodeUnprepareResources after Kubelet restart", func(ctx context.Context) {
			unset := kubeletPlugin.SetNodeUnprepareResourcesFailureMode()
			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drapod", true, []string{driverName})
			ginkgo.By("wait for NodePrepareResources call to succeed")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodePrepareResourcesSucceeded)

			ginkgo.By("wait for NodeUnprepareResources call to fail")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodeUnprepareResourcesFailed)

			ginkgo.By("stop Kubelet")
			startKubelet := stopKubelet()

			unset()

			ginkgo.By("start Kubelet")
			startKubelet()

			ginkgo.By("wait for NodeUnprepareResources call to succeed")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodeUnprepareResourcesSucceeded)

			ginkgo.By("wait for pod to succeed")
			err := e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
			framework.ExpectNoError(err)
		})

		ginkgo.It("must call NodeUnprepareResources for deleted pod", func(ctx context.Context) {
			unset := kubeletPlugin.SetNodeUnprepareResourcesFailureMode()
			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drapod", false, []string{driverName})

			ginkgo.By("wait for NodePrepareResources call to succeed")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodePrepareResourcesSucceeded)

			ginkgo.By("wait for NodeUnprepareResources call to fail")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodeUnprepareResourcesFailed)

			ginkgo.By("delete pod")
			e2epod.DeletePodOrFail(ctx, f.ClientSet, f.Namespace.Name, pod.Name)

			ginkgo.By("wait for NodeUnprepareResources call to fail")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodeUnprepareResourcesFailed)

			unset()

			ginkgo.By("wait for NodeUnprepareResources call to succeed")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodeUnprepareResourcesSucceeded)
		})

		ginkgo.It("must call NodeUnprepareResources for deleted pod after Kubelet restart", func(ctx context.Context) {
			unset := kubeletPlugin.SetNodeUnprepareResourcesFailureMode()
			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drapod", false, []string{driverName})

			ginkgo.By("wait for NodePrepareResources call to succeed")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodePrepareResourcesSucceeded)

			ginkgo.By("wait for NodeUnprepareResources call to fail")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodeUnprepareResourcesFailed)

			ginkgo.By("delete pod")
			err := e2epod.DeletePodWithGracePeriod(ctx, f.ClientSet, pod, 0)
			framework.ExpectNoError(err)

			ginkgo.By("wait for NodeUnprepareResources call to fail")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodeUnprepareResourcesFailed)

			ginkgo.By("restart Kubelet")
			stopKubelet()()

			ginkgo.By("wait for NodeUnprepareResources call to fail")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodeUnprepareResourcesFailed)

			unset()

			ginkgo.By("wait for NodeUnprepareResources call to succeed")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodeUnprepareResourcesSucceeded)
		})

		ginkgo.It("must not call NodePrepareResources for deleted pod after Kubelet restart", func(ctx context.Context) {
			unblock := kubeletPlugin.BlockNodePrepareResources()
			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drapod", false, []string{driverName})

			ginkgo.By("wait for pod to be in Pending state")
			err := e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "Pending", framework.PodStartShortTimeout, func(pod *v1.Pod) (bool, error) {
				return pod.Status.Phase == v1.PodPending, nil
			})
			framework.ExpectNoError(err)

			ginkgo.By("stop Kubelet")
			startKubelet := stopKubelet()

			ginkgo.By("delete pod")
			e2epod.DeletePodOrFail(ctx, f.ClientSet, f.Namespace.Name, pod.Name)

			unblock()

			ginkgo.By("start Kubelet")
			startKubelet()

			calls := kubeletPlugin.CountCalls("/NodePrepareResources")
			ginkgo.By("make sure NodePrepareResources is not called again")
			gomega.Consistently(kubeletPlugin.CountCalls("/NodePrepareResources")).WithTimeout(draplugin.PluginClientTimeout).Should(gomega.Equal(calls))
		})
	})

	f.Context("Two resource Kubelet Plugins", f.WithSerial(), func() {
		ginkgo.BeforeEach(func(ctx context.Context) {
			kubeletPlugin1 = newKubeletPlugin(ctx, getNodeName(ctx, f), kubeletPlugin1Name)
			kubeletPlugin2 = newKubeletPlugin(ctx, getNodeName(ctx, f), kubeletPlugin2Name)

			ginkgo.By("wait for Kubelet plugin registration")
			gomega.Eventually(kubeletPlugin1.GetGRPCCalls()).WithTimeout(pluginRegistrationTimeout).Should(testdriver.BeRegistered)
			gomega.Eventually(kubeletPlugin2.GetGRPCCalls()).WithTimeout(pluginRegistrationTimeout).Should(testdriver.BeRegistered)
		})

		ginkgo.It("must prepare and unprepare resources", func(ctx context.Context) {
			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drapod", true, []string{kubeletPlugin1Name, kubeletPlugin2Name})

			ginkgo.By("wait for pod to succeed")
			err := e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
			framework.ExpectNoError(err)

			ginkgo.By("wait for NodePrepareResources calls to succeed")
			gomega.Eventually(kubeletPlugin1.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodePrepareResourcesSucceeded)
			gomega.Eventually(kubeletPlugin2.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodePrepareResourcesSucceeded)

			ginkgo.By("wait for NodeUnprepareResources calls to succeed")
			gomega.Eventually(kubeletPlugin1.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodeUnprepareResourcesSucceeded)
			gomega.Eventually(kubeletPlugin2.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodeUnprepareResourcesSucceeded)
		})

		ginkgo.It("must run pod if NodePrepareResources fails for one plugin and then succeeds", func(ctx context.Context) {
			unset := kubeletPlugin2.SetNodePrepareResourcesFailureMode()
			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drapod", true, []string{kubeletPlugin1Name, kubeletPlugin2Name})

			ginkgo.By("wait for pod to be in Pending state")
			err := e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "Pending", framework.PodStartShortTimeout, func(pod *v1.Pod) (bool, error) {
				return pod.Status.Phase == v1.PodPending, nil
			})
			framework.ExpectNoError(err)

			ginkgo.By("wait for plugin2 NodePrepareResources call to fail")
			gomega.Eventually(kubeletPlugin2.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodePrepareResourcesFailed)

			unset()

			ginkgo.By("wait for plugin2 NodePrepareResources call to succeed")
			gomega.Eventually(kubeletPlugin2.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodePrepareResourcesSucceeded)

			ginkgo.By("wait for pod to succeed")
			err = e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
			framework.ExpectNoError(err)
		})

		ginkgo.It("must run pod if NodeUnprepareResources fails for one plugin and then succeeds", func(ctx context.Context) {
			unset := kubeletPlugin2.SetNodeUnprepareResourcesFailureMode()
			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drapod", true, []string{kubeletPlugin1Name, kubeletPlugin2Name})

			ginkgo.By("wait for plugin1 NodePrepareResources call to succeed")
			gomega.Eventually(kubeletPlugin1.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodePrepareResourcesSucceeded)

			ginkgo.By("wait for plugin2 NodePrepareResources call to succeed")
			gomega.Eventually(kubeletPlugin2.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodePrepareResourcesSucceeded)

			ginkgo.By("wait for plugin2 NodeUnprepareResources call to fail")
			gomega.Eventually(kubeletPlugin2.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodeUnprepareResourcesFailed)

			unset()

			ginkgo.By("wait for plugin2 NodeUnprepareResources call to succeed")
			gomega.Eventually(kubeletPlugin2.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodeUnprepareResourcesSucceeded)

			ginkgo.By("wait for pod to succeed")
			err := e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
			framework.ExpectNoError(err)
		})

		ginkgo.It("must run pod if NodePrepareResources is in progress for one plugin when Kubelet restarts", func(ctx context.Context) {
			unblock := kubeletPlugin.BlockNodePrepareResources()
			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drapod", true, []string{kubeletPlugin1Name, kubeletPlugin2Name})

			ginkgo.By("wait for pod to be in Pending state")
			err := e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "Pending", framework.PodStartShortTimeout, func(pod *v1.Pod) (bool, error) {
				return pod.Status.Phase == v1.PodPending, nil
			})
			framework.ExpectNoError(err)

			ginkgo.By("restart Kubelet")
			restartKubelet(true)

			unblock()

			ginkgo.By("wait for plugin2 NodePrepareResources call to succeed")
			gomega.Eventually(kubeletPlugin2.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodePrepareResourcesSucceeded)

			ginkgo.By("wait for pod to succeed")
			err = e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
			framework.ExpectNoError(err)
		})

		ginkgo.It("must call NodeUnprepareResources again if it's in progress for one plugin when Kubelet restarts", func(ctx context.Context) {
			unblock := kubeletPlugin2.BlockNodeUnprepareResources()
			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drapod", true, []string{kubeletPlugin1Name, kubeletPlugin2Name})

			ginkgo.By("wait for plugin1 NodePrepareResources call to succeed")
			gomega.Eventually(kubeletPlugin1.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodePrepareResourcesSucceeded)

			ginkgo.By("wait for plugin2 NodePrepareResources call to succeed")
			gomega.Eventually(kubeletPlugin2.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodePrepareResourcesSucceeded)

			ginkgo.By("restart Kubelet")
			restartKubelet(true)

			unblock()

			ginkgo.By("wait for plugin2 NodeUnprepareResources call to succeed")
			gomega.Eventually(kubeletPlugin2.GetGRPCCalls).WithTimeout(draplugin.PluginClientTimeout * 2).Should(testdriver.NodeUnprepareResourcesSucceeded)

			ginkgo.By("wait for pod to succeed")
			err := e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
			framework.ExpectNoError(err)
		})
	})
})

// Run Kubelet plugin and wait until it's registered
func newKubeletPlugin(ctx context.Context, nodeName, pluginName string) *testdriver.ExamplePlugin {
	ginkgo.By("start Kubelet plugin")
	logger := klog.LoggerWithValues(klog.LoggerWithName(klog.Background(), "kubelet plugin "+pluginName), "node", nodeName)
	ctx = klog.NewContext(ctx, logger)

	// Ensure that directories exist, creating them if necessary. We want
	// to know early if there is a setup problem that would prevent
	// creating those directories.
	err := os.MkdirAll(cdiDir, os.FileMode(0750))
	framework.ExpectNoError(err, "create CDI directory")
	endpoint := fmt.Sprintf(endpointTemplate, pluginName)
	err = os.MkdirAll(filepath.Dir(endpoint), 0750)
	framework.ExpectNoError(err, "create socket directory")

	plugin, err := testdriver.StartPlugin(
		ctx,
		cdiDir,
		pluginName,
		"",
		testdriver.FileOperations{},
		kubeletplugin.PluginSocketPath(endpoint),
		kubeletplugin.RegistrarSocketPath(path.Join(pluginRegistrationPath, pluginName+"-reg.sock")),
		kubeletplugin.KubeletPluginSocketPath(endpoint),
	)
	framework.ExpectNoError(err)

	gomega.Eventually(plugin.GetGRPCCalls).WithTimeout(pluginRegistrationTimeout).Should(testdriver.BeRegistered)

	ginkgo.DeferCleanup(plugin.Stop)

	return plugin
}

// createTestObjects creates objects required by the test
// NOTE: as scheduler and controller manager are not running by the Node e2e,
// the objects must contain all required data to be processed correctly by the API server
// and placed on the node without involving the scheduler and the DRA controller
func createTestObjects(ctx context.Context, clientSet kubernetes.Interface, nodename, namespace, className, claimName, podName string, deferPodDeletion bool, pluginNames []string) *v1.Pod {
	// ResourceClass
	class := &resourcev1alpha2.ResourceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: className,
		},
		DriverName: "controller",
	}
	_, err := clientSet.ResourceV1alpha2().ResourceClasses().Create(ctx, class, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	ginkgo.DeferCleanup(clientSet.ResourceV1alpha2().ResourceClasses().Delete, className, metav1.DeleteOptions{})

	// ResourceClaim
	podClaimName := "resource-claim"
	claim := &resourcev1alpha2.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: claimName,
		},
		Spec: resourcev1alpha2.ResourceClaimSpec{
			ResourceClassName: className,
		},
	}
	createdClaim, err := clientSet.ResourceV1alpha2().ResourceClaims(namespace).Create(ctx, claim, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	ginkgo.DeferCleanup(clientSet.ResourceV1alpha2().ResourceClaims(namespace).Delete, claimName, metav1.DeleteOptions{})

	// Pod
	containerName := "testcontainer"
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: namespace,
		},
		Spec: v1.PodSpec{
			NodeName: nodename, // Assign the node as the scheduler is not running
			ResourceClaims: []v1.PodResourceClaim{
				{
					Name:              podClaimName,
					ResourceClaimName: &claimName,
				},
			},
			Containers: []v1.Container{
				{
					Name:  containerName,
					Image: e2epod.GetDefaultTestImage(),
					Resources: v1.ResourceRequirements{
						Claims: []v1.ResourceClaim{{Name: podClaimName}},
					},
					Command: []string{"/bin/sh", "-c", "env | grep DRA_PARAM1=PARAM1_VALUE"},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
	createdPod, err := clientSet.CoreV1().Pods(namespace).Create(ctx, pod, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	if deferPodDeletion {
		ginkgo.DeferCleanup(clientSet.CoreV1().Pods(namespace).Delete, podName, metav1.DeleteOptions{})
	}

	// Update claim status: set ReservedFor and AllocationResult
	// NOTE: This is usually done by the DRA controller
	resourceHandlers := make([]resourcev1alpha2.ResourceHandle, len(pluginNames))
	for i, pluginName := range pluginNames {
		resourceHandlers[i] = resourcev1alpha2.ResourceHandle{
			DriverName: pluginName,
			Data:       "{\"EnvVars\":{\"DRA_PARAM1\":\"PARAM1_VALUE\"},\"NodeName\":\"\"}",
		}
	}
	createdClaim.Status = resourcev1alpha2.ResourceClaimStatus{
		DriverName: "controller",
		ReservedFor: []resourcev1alpha2.ResourceClaimConsumerReference{
			{Resource: "pods", Name: podName, UID: createdPod.UID},
		},
		Allocation: &resourcev1alpha2.AllocationResult{
			ResourceHandles: resourceHandlers,
		},
	}
	_, err = clientSet.ResourceV1alpha2().ResourceClaims(namespace).UpdateStatus(ctx, createdClaim, metav1.UpdateOptions{})
	framework.ExpectNoError(err)

	return pod
}
