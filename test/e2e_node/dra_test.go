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
    make test-e2e-node FOCUS='\[Feature:DynamicResourceAllocation\]' SKIP='\[Flaky\]' PARALLELISM=1 \
       TEST_ARGS='--feature-gates="DynamicResourceAllocation=true,ResourceHealthStatus=true,DRAConsumableCapacity=true" --service-feature-gates="DynamicResourceAllocation=true,ResourceHealthStatus=true,DRAConsumableCapacity=true" --runtime-config=api/all=true'
*/

package e2enode

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"path"
	"regexp"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	"github.com/onsi/gomega/types"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/client-go/kubernetes"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"

	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"

	"k8s.io/dynamic-resource-allocation/kubeletplugin"
	"k8s.io/dynamic-resource-allocation/resourceslice"
	testdriver "k8s.io/kubernetes/test/e2e/dra/test-driver/app"
	testdrivergomega "k8s.io/kubernetes/test/e2e/dra/test-driver/gomega"
)

const (
	driverName                = "test-driver.cdi.k8s.io"
	kubeletPlugin1Name        = "test-driver1.cdi.k8s.io"
	kubeletPlugin2Name        = "test-driver2.cdi.k8s.io"
	cdiDir                    = "/var/run/cdi"
	pluginRegistrationTimeout = time.Second * 60 // how long to wait for a node plugin to be registered
	podInPendingStateTimeout  = time.Second * 60 // how long to wait for a pod to stay in pending state

	// kubeletRetryPeriod reflects how often the kubelet tries to start a container after
	// some non-fatal failure. This does not not include the time it took for the last attempt
	// itself (?!).
	//
	// Value from https://github.com/kubernetes/kubernetes/commit/0449cef8fd5217d394c5cd331d852bd50983e6b3.
	kubeletRetryPeriod = 90 * time.Second

	// retryTestTimeout is the maximum duration that a test takes for one
	// failed attempt to start a pod followed by another successful
	// attempt.
	//
	// Also used as timeout in other tests because it's a good upper bound
	// even when the test normally completes faster.
	retryTestTimeout = kubeletRetryPeriod + 30*time.Second
)

// Tests depend on container runtime support for CDI and the DRA feature gate.
// The "DRA" label is used to select tests related to DRA in a Ginkgo label filter.
var _ = framework.SIGDescribe("node")(framework.WithLabel("DRA"), feature.DynamicResourceAllocation, framework.WithFeatureGate(features.DynamicResourceAllocation), func() {
	f := framework.NewDefaultFramework("dra-node")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.BeforeEach(func() {
		ginkgo.DeferCleanup(func(ctx context.Context) {
			// When plugin and kubelet get killed at the end of the tests, they leave ResourceSlices behind.
			// Perhaps garbage collection would eventually remove them (not sure how the node instance
			// is managed), but this could take time. Let's clean up explicitly.
			framework.ExpectNoError(f.ClientSet.ResourceV1().ResourceSlices().DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{}))
		})
	})

	f.Context("Resource Kubelet Plugin", f.WithSerial(), func() {
		ginkgo.It("must register after Kubelet restart", func(ctx context.Context) {
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, f.Namespace.Name, getNodeName(ctx, f), driverName)

			oldCalls := kubeletPlugin.GetGRPCCalls()
			getNewCalls := func() []testdriver.GRPCCall {
				calls := kubeletPlugin.GetGRPCCalls()
				return calls[len(oldCalls):]
			}

			ginkgo.By("restarting Kubelet")
			restartKubelet(ctx, true)

			ginkgo.By("wait for Kubelet plugin re-registration")
			gomega.Eventually(getNewCalls).WithTimeout(pluginRegistrationTimeout).Should(testdrivergomega.BeRegistered)
		})

		ginkgo.It("must register after plugin restart", func(ctx context.Context) {
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, f.Namespace.Name, getNodeName(ctx, f), driverName)

			ginkgo.By("restart Kubelet Plugin")
			kubeletPlugin.Stop()
			kubeletPlugin = newKubeletPlugin(ctx, f.ClientSet, f.Namespace.Name, getNodeName(ctx, f), driverName)

			ginkgo.By("wait for Kubelet plugin re-registration")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(pluginRegistrationTimeout).Should(testdrivergomega.BeRegistered)
		})

		// Test that the kubelet plugin manager retries plugin registration
		// when the GetInfo call fails, and succeeds once the call passes.
		ginkgo.It("must recover and register after registration failure", func(ctx context.Context) {
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, f.Namespace.Name, getNodeName(ctx, f), driverName)

			ginkgo.By("set GetInfo failure mode")
			kubeletPlugin.SetGetInfoError(fmt.Errorf("simulated GetInfo failure"))
			kubeletPlugin.ResetGRPCCalls()

			ginkgo.By("restart Kubelet")
			restartKubelet(ctx, true)

			ginkgo.By("wait for Registration call to fail")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.GetInfoFailed())
			gomega.Expect(kubeletPlugin.GetGRPCCalls()).ShouldNot(testdrivergomega.BeRegistered, "Expect plugin not to be registered due to GetInfo failure")

			ginkgo.By("unset registration failure mode")
			kubeletPlugin.SetGetInfoError(nil)

			ginkgo.By("wait for Kubelet plugin re-registration")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(pluginRegistrationTimeout).Should(testdrivergomega.BeRegistered)
		})

		ginkgo.It("must process pod created when kubelet is not running", func(ctx context.Context) {
			newKubeletPlugin(ctx, f.ClientSet, f.Namespace.Name, getNodeName(ctx, f), driverName)

			// Stop Kubelet
			ginkgo.By("stop kubelet")
			restartKubelet := mustStopKubelet(ctx, f)
			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drapod", true, []string{driverName})
			// Pod must be in pending state
			err := e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "Pending", framework.PodStartShortTimeout, func(pod *v1.Pod) (bool, error) {
				return pod.Status.Phase == v1.PodPending, nil
			})
			framework.ExpectNoError(err)
			ginkgo.By("restart kubelet")
			restartKubelet(ctx)
			// Pod should succeed
			err = e2epod.WaitForPodSuccessInNamespaceTimeout(ctx, f.ClientSet, pod.Name, f.Namespace.Name, framework.PodStartShortTimeout)
			framework.ExpectNoError(err)
		})

		ginkgo.It("must keep pod in pending state if NodePrepareResources times out", func(ctx context.Context) {
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, f.Namespace.Name, getNodeName(ctx, f), driverName)

			unblockNodePrepareResources := kubeletPlugin.BlockNodePrepareResources()
			defer unblockNodePrepareResources()
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
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, f.Namespace.Name, getNodeName(ctx, f), driverName)

			unsetNodePrepareResourcesFailureMode := kubeletPlugin.SetNodePrepareResourcesFailureMode()
			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drapod", true, []string{driverName})

			ginkgo.By("wait for pod to be in Pending state")
			err := e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "Pending", framework.PodStartShortTimeout, func(pod *v1.Pod) (bool, error) {
				return pod.Status.Phase == v1.PodPending, nil
			})
			framework.ExpectNoError(err)

			ginkgo.By("wait for NodePrepareResources call to fail")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodePrepareResourcesFailed)

			unsetNodePrepareResourcesFailureMode()

			ginkgo.By("wait for NodePrepareResources call to succeed")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodePrepareResourcesSucceeded)

			ginkgo.By("wait for pod to succeed")
			err = e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
			framework.ExpectNoError(err)
		})

		ginkgo.It("must run pod if NodeUnprepareResources fails and then succeeds", func(ctx context.Context) {
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, f.Namespace.Name, getNodeName(ctx, f), driverName)

			unsetNodeUnprepareResourcesFailureMode := kubeletPlugin.SetNodeUnprepareResourcesFailureMode()
			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drapod", true, []string{driverName})

			ginkgo.By("wait for NodePrepareResources call to succeed")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodePrepareResourcesSucceeded)

			ginkgo.By("wait for NodeUnprepareResources call to fail")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodeUnprepareResourcesFailed)

			unsetNodeUnprepareResourcesFailureMode()

			ginkgo.By("wait for NodeUnprepareResources call to succeed")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodeUnprepareResourcesSucceeded)

			ginkgo.By("wait for pod to succeed")
			err := e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
			framework.ExpectNoError(err)
		})

		ginkgo.It("must retry NodePrepareResources after Kubelet restart", func(ctx context.Context) {
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, f.Namespace.Name, getNodeName(ctx, f), driverName)

			unsetNodePrepareResourcesFailureMode := kubeletPlugin.SetNodePrepareResourcesFailureMode()
			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drapod", true, []string{driverName})

			ginkgo.By("wait for pod to be in Pending state")
			err := e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "Pending", framework.PodStartShortTimeout, func(pod *v1.Pod) (bool, error) {
				return pod.Status.Phase == v1.PodPending, nil
			})
			framework.ExpectNoError(err)

			ginkgo.By("wait for NodePrepareResources call to fail")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodePrepareResourcesFailed)

			ginkgo.By("stop Kubelet")
			restartKubelet := mustStopKubelet(ctx, f)

			unsetNodePrepareResourcesFailureMode()

			ginkgo.By("restart Kubelet")
			restartKubelet(ctx)

			ginkgo.By("wait for NodePrepareResources call to succeed")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodePrepareResourcesSucceeded)

			ginkgo.By("wait for pod to succeed")
			err = e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
			framework.ExpectNoError(err)
		})

		ginkgo.It("must retry NodeUnprepareResources after Kubelet restart", func(ctx context.Context) {
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, f.Namespace.Name, getNodeName(ctx, f), driverName)

			unsetNodeUnprepareResourcesFailureMode := kubeletPlugin.SetNodeUnprepareResourcesFailureMode()
			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drapod", true, []string{driverName})
			ginkgo.By("wait for NodePrepareResources call to succeed")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodePrepareResourcesSucceeded)

			ginkgo.By("wait for NodeUnprepareResources call to fail")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodeUnprepareResourcesFailed)

			ginkgo.By("stop Kubelet")
			restartKubelet := mustStopKubelet(ctx, f)

			unsetNodeUnprepareResourcesFailureMode()

			ginkgo.By("restart Kubelet")
			restartKubelet(ctx)

			ginkgo.By("wait for NodeUnprepareResources call to succeed")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodeUnprepareResourcesSucceeded)

			ginkgo.By("wait for pod to succeed")
			err := e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
			framework.ExpectNoError(err)
		})

		ginkgo.It("must call NodeUnprepareResources for deleted pod", func(ctx context.Context) {
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, f.Namespace.Name, getNodeName(ctx, f), driverName)

			unsetNodeUnprepareResourcesFailureMode := kubeletPlugin.SetNodeUnprepareResourcesFailureMode()
			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drapod", false, []string{driverName})

			ginkgo.By("wait for NodePrepareResources call to succeed")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodePrepareResourcesSucceeded)

			ginkgo.By("wait for NodeUnprepareResources call to fail")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodeUnprepareResourcesFailed)

			ginkgo.By("delete pod")
			e2epod.DeletePodOrFail(ctx, f.ClientSet, f.Namespace.Name, pod.Name)

			ginkgo.By("wait for NodeUnprepareResources call to fail")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodeUnprepareResourcesFailed)

			unsetNodeUnprepareResourcesFailureMode()

			ginkgo.By("wait for NodeUnprepareResources call to succeed")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodeUnprepareResourcesSucceeded)
		})

		ginkgo.It("must call NodeUnprepareResources for deleted pod after Kubelet restart", func(ctx context.Context) {
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, f.Namespace.Name, getNodeName(ctx, f), driverName)

			unsetNodeUnprepareResourcesFailureMode := kubeletPlugin.SetNodeUnprepareResourcesFailureMode()
			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drapod", false, []string{driverName})

			ginkgo.By("wait for NodePrepareResources call to succeed")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodePrepareResourcesSucceeded)

			ginkgo.By("wait for NodeUnprepareResources call to fail")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodeUnprepareResourcesFailed)

			ginkgo.By("delete pod")
			err := e2epod.DeletePodWithGracePeriod(ctx, f.ClientSet, pod, 0)
			framework.ExpectNoError(err)

			ginkgo.By("wait for NodeUnprepareResources call to fail")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodeUnprepareResourcesFailed)

			ginkgo.By("restart Kubelet")
			restartKubelet(ctx, true)

			ginkgo.By("wait for NodeUnprepareResources call to fail")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodeUnprepareResourcesFailed)

			unsetNodeUnprepareResourcesFailureMode()

			ginkgo.By("wait for NodeUnprepareResources call to succeed")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodeUnprepareResourcesSucceeded)
		})

		ginkgo.It("must not call NodePrepareResources for deleted pod after Kubelet restart", func(ctx context.Context) {
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, f.Namespace.Name, getNodeName(ctx, f), driverName)

			unblockNodePrepareResources := kubeletPlugin.BlockNodePrepareResources()
			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drapod", false, []string{driverName})

			ginkgo.By("wait for pod to be in Pending state")
			err := e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "Pending", framework.PodStartShortTimeout, func(pod *v1.Pod) (bool, error) {
				return pod.Status.Phase == v1.PodPending, nil
			})
			framework.ExpectNoError(err)

			ginkgo.By("stop Kubelet")
			restartKubelet := mustStopKubelet(ctx, f)

			ginkgo.By("delete pod")
			e2epod.DeletePodOrFail(ctx, f.ClientSet, f.Namespace.Name, pod.Name)

			unblockNodePrepareResources()

			ginkgo.By("restart Kubelet")
			restartKubelet(ctx)

			calls := kubeletPlugin.CountCalls("/NodePrepareResources")
			ginkgo.By("make sure NodePrepareResources is not called again")
			gomega.Consistently(func() int {
				return kubeletPlugin.CountCalls("/NodePrepareResources")
			}).WithTimeout(retryTestTimeout).Should(gomega.Equal(calls))
		})

		functionalListenAfterRegistration := func(ctx context.Context, datadir string, opts ...any) {
			nodeName := getNodeName(ctx, f)

			ginkgo.By("start DRA registrar")
			registrar := newRegistrar(ctx, f.ClientSet, nodeName, driverName, opts...)

			ginkgo.By("wait for registration to complete")
			gomega.Eventually(registrar.GetGRPCCalls).WithTimeout(pluginRegistrationTimeout).Should(testdrivergomega.BeRegistered)

			ginkgo.By("start DRA plugin service")
			draService := newDRAService(ctx, f.ClientSet, f.Namespace.Name, nodeName, driverName, datadir, opts...)

			pod := createTestObjects(ctx, f.ClientSet, nodeName, f.Namespace.Name, "draclass", "external-claim", "drapod", false, []string{driverName})

			ginkgo.By("wait for NodePrepareResources call to succeed")
			gomega.Eventually(draService.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodePrepareResourcesSucceeded)

			ginkgo.By("wait for pod to succeed")
			err := e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
			framework.ExpectNoError(err)
		}
		ginkgo.DescribeTable("must be functional when plugin starts to listen on a service socket after registration",
			functionalListenAfterRegistration,
			ginkgo.Entry("2 sockets", ""),
			ginkgo.Entry(
				"1 common socket",
				kubeletplugin.KubeletRegistryDir,
				kubeletplugin.PluginDataDirectoryPath(kubeletplugin.KubeletRegistryDir),
				kubeletplugin.PluginSocket(driverName+"-common.sock"),
			),
		)

		functionalAfterServiceReconnect := func(ctx context.Context, datadir string, opts ...any) {
			nodeName := getNodeName(ctx, f)

			ginkgo.By("start DRA registrar")
			registrar := newRegistrar(ctx, f.ClientSet, nodeName, driverName, opts...)

			ginkgo.By("wait for registration to complete")
			gomega.Eventually(registrar.GetGRPCCalls).WithTimeout(pluginRegistrationTimeout).Should(testdrivergomega.BeRegistered)

			ginkgo.By("start DRA plugin service")
			draService := newDRAService(ctx, f.ClientSet, f.Namespace.Name, nodeName, driverName, datadir, opts...)

			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drasleeppod" /* enables sleeping */, false /* pod is deleted below */, []string{driverName})

			ginkgo.By("wait for NodePrepareResources call to succeed")
			gomega.Eventually(draService.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodePrepareResourcesSucceeded)

			ginkgo.By("stop plugin")
			draService.Stop()

			ginkgo.By("waiting for pod to run")
			err := e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod)
			framework.ExpectNoError(err)

			ginkgo.By("wait for ResourceSlice removal, indicating detection of disconnect")
			gomega.Eventually(ctx, listResources(f.ClientSet)).Should(gomega.BeEmpty(), "ResourceSlices without plugin")

			ginkgo.By("restarting plugin")
			draService = newDRAService(ctx, f.ClientSet, f.Namespace.Name, nodeName, driverName, datadir, opts...)

			ginkgo.By("stopping pod and waiting for it to get removed, indicating that NodeUnprepareResources must have been called")
			err = f.ClientSet.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "delete pod")
			gomega.Eventually(draService.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodeUnprepareResourcesSucceeded)
			err = e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace, f.Timeouts.PodDelete)
			framework.ExpectNoError(err, "pod + claim shutdown")
		}
		ginkgo.DescribeTable("must be functional after service reconnect",
			functionalAfterServiceReconnect,
			ginkgo.Entry("2 sockets", ""),
			ginkgo.Entry(
				"1 common socket",
				kubeletplugin.KubeletRegistryDir,
				kubeletplugin.PluginDataDirectoryPath(kubeletplugin.KubeletRegistryDir),
				kubeletplugin.PluginSocket(driverName+"-common.sock"),
			),
		)

		failOnClosedListener := func(
			ctx context.Context,
			service func(ctx context.Context, clientSet kubernetes.Interface, testNamespace, nodeName, driverName, datadir string, opts ...any) *testdriver.ExamplePlugin,
			listenerOptionFun func(listen func(ctx context.Context, path string) (net.Listener, error)) kubeletplugin.Option,
		) {
			ginkgo.By("create a custom listener")
			var listener net.Listener
			errorMsg := "simulated listener failure"
			getListener := func(ctx context.Context, socketPath string) (net.Listener, error) {
				listener = newErrorOnCloseListener(errors.New(errorMsg))
				return listener, nil
			}

			ginkgo.By("create a context with a cancel function")
			tCtx, cancel := context.WithCancelCause(ctx)
			defer cancel(nil)

			ginkgo.By("start service")
			service(
				ctx,
				f.ClientSet,
				f.Namespace.Name,
				getNodeName(ctx, f),
				driverName,
				"",
				listenerOptionFun(getListener),
				testdriver.CancelMainContext(cancel),
			)

			ginkgo.By("close listener to make the grpc.Server.Serve() fail")
			framework.ExpectNoError(listener.Close())

			ginkgo.By("check that the context is canceled with an expected error and cause")
			gomega.Eventually(tCtx.Err).Should(gomega.MatchError(gomega.ContainSubstring("context canceled")), "Context should be canceled by the error handler")
			gomega.Expect(context.Cause(tCtx).Error()).To(gomega.ContainSubstring(errorMsg), "Context should be canceled with the expected cause")
		}
		// The wrappedNewRegistrar function is used to create a new registrar
		// with the same signature as the newDRAService function, so that it can be
		// used in the DescribeTable.
		wrappedNewRegistrar := func(ctx context.Context, clientSet kubernetes.Interface, testNamespace, nodeName, driverName, datadir string, opts ...any) *testdriver.ExamplePlugin {
			return newRegistrar(ctx, clientSet, nodeName, driverName, opts...)
		}
		ginkgo.DescribeTable("must report gRPC serving error",
			failOnClosedListener,
			ginkgo.Entry("for registrar", wrappedNewRegistrar, kubeletplugin.RegistrarListener),
			ginkgo.Entry("for DRA service", newDRAService, kubeletplugin.PluginListener),
		)

		// This test verifies that the Kubelet establishes only a single gRPC connection
		// with the DRA plugin throughout the plugin lifecycle.
		//
		// It does so by:
		// - Using a custom gRPC listener that counts accepted connections.
		// - Sequentially creating pods that trigger NodePrepareResources and NodeUnprepareResources.
		// - Asserting that only one connection was accepted by the listener — since each
		// call to net.Listener.Accept() corresponds to a new incoming connection —
		// ensures that the plugin reused the same gRPC connection for all calls.
		ginkgo.It("must use one gRPC connection for all service calls", func(ctx context.Context) {
			var listener *countingListener
			var err error
			getListener := func(_ context.Context, socketPath string) (net.Listener, error) {
				listener, err = newCountingListener(socketPath)
				return listener, err
			}

			nodeName := getNodeName(ctx, f)

			ginkgo.By("start DRA registrar")
			registrar := newRegistrar(ctx, f.ClientSet, nodeName, driverName)

			ginkgo.By("wait for registration to complete")
			gomega.Eventually(registrar.GetGRPCCalls).WithTimeout(pluginRegistrationTimeout).Should(testdrivergomega.BeRegistered)

			ginkgo.By("start DRA plugin service")
			draService := newDRAService(ctx, f.ClientSet, f.Namespace.Name, nodeName, driverName, "", kubeletplugin.PluginListener(getListener))

			for _, suffix := range []string{"-1", "-2"} {
				draService.ResetGRPCCalls()

				ginkgo.By("create test objects " + suffix)
				pod := createTestObjects(ctx, f.ClientSet, nodeName, f.Namespace.Name, "draclass"+suffix, "external-claim"+suffix, "drapod"+suffix, true, []string{driverName})

				ginkgo.By("wait for NodePrepareResources call to succeed " + suffix)
				gomega.Eventually(draService.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodePrepareResourcesSucceeded)

				ginkgo.By("wait for NodeUnprepareResources call to succeed " + suffix)
				gomega.Eventually(draService.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodeUnprepareResourcesSucceeded)

				ginkgo.By("wait for pod to succeed " + suffix)
				err = e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
				framework.ExpectNoError(err)

				ginkgo.By("check that listener.Accept was called only once " + suffix)
				gomega.Expect(listener.acceptCount()).To(gomega.Equal(1))
			}
		})
	})

	f.Context("Two resource Kubelet Plugins", f.WithSerial(), func() {
		// start creates plugins which will get stopped when the context gets canceled.
		start := func(ctx context.Context) (*testdriver.ExamplePlugin, *testdriver.ExamplePlugin) {
			kubeletPlugin1 := newKubeletPlugin(ctx, f.ClientSet, f.Namespace.Name, getNodeName(ctx, f), kubeletPlugin1Name)
			kubeletPlugin2 := newKubeletPlugin(ctx, f.ClientSet, f.Namespace.Name, getNodeName(ctx, f), kubeletPlugin2Name)

			ginkgo.By("wait for Kubelet plugin registration")
			gomega.Eventually(kubeletPlugin1.GetGRPCCalls()).WithTimeout(pluginRegistrationTimeout).Should(testdrivergomega.BeRegistered)
			gomega.Eventually(kubeletPlugin2.GetGRPCCalls()).WithTimeout(pluginRegistrationTimeout).Should(testdrivergomega.BeRegistered)

			return kubeletPlugin1, kubeletPlugin2
		}

		ginkgo.It("must prepare and unprepare resources", func(ctx context.Context) {
			kubeletPlugin1, kubeletPlugin2 := start(ctx)

			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drapod", true, []string{kubeletPlugin1Name, kubeletPlugin2Name})

			ginkgo.By("wait for pod to succeed")
			err := e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
			framework.ExpectNoError(err)

			ginkgo.By("wait for NodePrepareResources calls to succeed")
			gomega.Eventually(kubeletPlugin1.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodePrepareResourcesSucceeded)
			gomega.Eventually(kubeletPlugin2.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodePrepareResourcesSucceeded)

			ginkgo.By("wait for NodeUnprepareResources calls to succeed")
			gomega.Eventually(kubeletPlugin1.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodeUnprepareResourcesSucceeded)
			gomega.Eventually(kubeletPlugin2.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodeUnprepareResourcesSucceeded)
		})

		ginkgo.It("must provide metrics", func(ctx context.Context) {
			kubeletPlugin1, kubeletPlugin2 := start(ctx)

			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drasleeppod", false, []string{kubeletPlugin1Name, kubeletPlugin2Name})

			ginkgo.By("wait for pod to succeed")
			err := e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod)
			framework.ExpectNoError(err)
			gomega.Expect(kubeletPlugin1.GetGRPCCalls()).Should(testdrivergomega.NodePrepareResourcesSucceeded, "Plugin 1 should have prepared resources.")
			gomega.Expect(kubeletPlugin2.GetGRPCCalls()).Should(testdrivergomega.NodePrepareResourcesSucceeded, "Plugin 2 should have prepared resources.")
			gomega.Expect(getKubeletMetrics(ctx)).Should(gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"dra_resource_claims_in_use": gstruct.MatchAllElements(driverNameLabelKey, gstruct.Elements{
					"<any>":            timelessSample(1),
					kubeletPlugin1Name: timelessSample(1),
					kubeletPlugin2Name: timelessSample(1),
				}),
			}), "metrics while pod is running")

			ginkgo.By("delete pod")
			err = f.ClientSet.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
			err = e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace, f.Timeouts.PodDelete)
			framework.ExpectNoError(err)
			gomega.Expect(kubeletPlugin1.GetGRPCCalls()).Should(testdrivergomega.NodeUnprepareResourcesSucceeded, "Plugin 2 should have unprepared resources.")
			gomega.Expect(kubeletPlugin2.GetGRPCCalls()).Should(testdrivergomega.NodeUnprepareResourcesSucceeded, "Plugin 2 should have unprepared resources.")
			gomega.Expect(getKubeletMetrics(ctx)).Should(gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"dra_resource_claims_in_use": gstruct.MatchAllElements(driverNameLabelKey, gstruct.Elements{
					"<any>": timelessSample(0),
				}),
			}), "metrics after deleting pod")
		})

		ginkgo.It("must run pod if NodePrepareResources fails for one plugin and then succeeds", func(ctx context.Context) {
			_, kubeletPlugin2 := start(ctx)

			unsetNodePrepareResourcesFailureMode := kubeletPlugin2.SetNodePrepareResourcesFailureMode()
			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drapod", true, []string{kubeletPlugin1Name, kubeletPlugin2Name})

			ginkgo.By("wait for pod to be in Pending state")
			err := e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "Pending", framework.PodStartShortTimeout, func(pod *v1.Pod) (bool, error) {
				return pod.Status.Phase == v1.PodPending, nil
			})
			framework.ExpectNoError(err)

			ginkgo.By("wait for plugin2 NodePrepareResources call to fail")
			gomega.Eventually(kubeletPlugin2.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodePrepareResourcesFailed)

			unsetNodePrepareResourcesFailureMode()

			ginkgo.By("wait for plugin2 NodePrepareResources call to succeed")
			gomega.Eventually(kubeletPlugin2.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodePrepareResourcesSucceeded)

			ginkgo.By("wait for pod to succeed")
			err = e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
			framework.ExpectNoError(err)
		})

		ginkgo.It("must run pod if NodeUnprepareResources fails for one plugin and then succeeds", func(ctx context.Context) {
			kubeletPlugin1, kubeletPlugin2 := start(ctx)

			unsetNodeUnprepareResourcesFailureMode := kubeletPlugin2.SetNodeUnprepareResourcesFailureMode()
			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drapod", true, []string{kubeletPlugin1Name, kubeletPlugin2Name})

			ginkgo.By("wait for plugin1 NodePrepareResources call to succeed")
			gomega.Eventually(kubeletPlugin1.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodePrepareResourcesSucceeded)

			ginkgo.By("wait for plugin2 NodePrepareResources call to succeed")
			gomega.Eventually(kubeletPlugin2.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodePrepareResourcesSucceeded)

			ginkgo.By("wait for plugin2 NodeUnprepareResources call to fail")
			gomega.Eventually(kubeletPlugin2.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodeUnprepareResourcesFailed)

			unsetNodeUnprepareResourcesFailureMode()

			ginkgo.By("wait for plugin2 NodeUnprepareResources call to succeed")
			gomega.Eventually(kubeletPlugin2.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodeUnprepareResourcesSucceeded)

			ginkgo.By("wait for pod to succeed")
			err := e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
			framework.ExpectNoError(err)
		})

		ginkgo.It("must run pod if NodePrepareResources is in progress for one plugin when Kubelet restarts", func(ctx context.Context) {
			kubeletPlugin1, kubeletPlugin2 := start(ctx)

			unblockNodePrepareResources := kubeletPlugin1.BlockNodePrepareResources()
			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drapod", true, []string{kubeletPlugin1Name, kubeletPlugin2Name})

			ginkgo.By("wait for pod to be in Pending state")
			err := e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "Pending", framework.PodStartShortTimeout, func(pod *v1.Pod) (bool, error) {
				return pod.Status.Phase == v1.PodPending, nil
			})
			framework.ExpectNoError(err)

			ginkgo.By("restart Kubelet")
			restartKubelet(ctx, true)

			unblockNodePrepareResources()

			ginkgo.By("wait for plugin2 NodePrepareResources call to succeed")
			gomega.Eventually(kubeletPlugin2.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodePrepareResourcesSucceeded)

			ginkgo.By("wait for pod to succeed")
			err = e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
			framework.ExpectNoError(err)
		})

		ginkgo.It("must call NodeUnprepareResources again if it's in progress for one plugin when Kubelet restarts", func(ctx context.Context) {
			kubeletPlugin1, kubeletPlugin2 := start(ctx)

			unblockNodeUnprepareResources := kubeletPlugin2.BlockNodeUnprepareResources()
			pod := createTestObjects(ctx, f.ClientSet, getNodeName(ctx, f), f.Namespace.Name, "draclass", "external-claim", "drapod", true, []string{kubeletPlugin1Name, kubeletPlugin2Name})

			ginkgo.By("wait for plugin1 NodePrepareResources call to succeed")
			gomega.Eventually(kubeletPlugin1.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodePrepareResourcesSucceeded)

			ginkgo.By("wait for plugin2 NodePrepareResources call to succeed")
			gomega.Eventually(kubeletPlugin2.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodePrepareResourcesSucceeded)

			ginkgo.By("restart Kubelet")
			restartKubelet(ctx, true)

			unblockNodeUnprepareResources()

			ginkgo.By("wait for plugin2 NodeUnprepareResources call to succeed")
			gomega.Eventually(kubeletPlugin2.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodeUnprepareResourcesSucceeded)

			ginkgo.By("wait for pod to succeed")
			err := e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
			framework.ExpectNoError(err)
		})
	})

	f.Context("ResourceSlice", f.WithSerial(), func() {
		f.It("must be removed on kubelet startup", f.WithDisruptive(), func(ctx context.Context) {
			ginkgo.By("stop kubelet")
			restartKubelet := mustStopKubelet(ctx, f)
			ginkgo.DeferCleanup(func() {
				if restartKubelet != nil {
					restartKubelet(ctx)
				}
			})

			ginkgo.By("create some ResourceSlices")
			nodeName := getNodeName(ctx, f)
			otherNodeName := nodeName + "-other"
			createTestResourceSlice(ctx, f.ClientSet, nodeName, driverName)
			createTestResourceSlice(ctx, f.ClientSet, nodeName+"-other", driverName)

			matchAll := gomega.ConsistOf(matchResourcesByNodeName(nodeName), matchResourcesByNodeName(otherNodeName))
			matchOtherNode := gomega.ConsistOf(matchResourcesByNodeName(otherNodeName))

			gomega.Consistently(ctx, listResources(f.ClientSet)).WithTimeout(5*time.Second).Should(matchAll, "ResourceSlices without kubelet")

			ginkgo.By("restart kubelet")
			restartKubelet(ctx)
			restartKubelet = nil

			ginkgo.By("wait for exactly the node's ResourceSlice to get deleted")
			gomega.Eventually(ctx, listResources(f.ClientSet)).Should(matchOtherNode, "ResourceSlices with kubelet")
			gomega.Consistently(ctx, listResources(f.ClientSet)).WithTimeout(5*time.Second).Should(matchOtherNode, "ResourceSlices with kubelet")
		})

		f.It("must be removed after plugin unregistration", func(ctx context.Context) {
			nodeName := getNodeName(ctx, f)
			matchNode := gomega.ConsistOf(matchResourcesByNodeName(nodeName))

			ginkgo.By("start plugin and wait for ResourceSlice")
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, f.Namespace.Name, getNodeName(ctx, f), driverName)
			gomega.Eventually(ctx, listResources(f.ClientSet)).Should(matchNode, "ResourceSlice from kubelet plugin")
			gomega.Consistently(ctx, listResources(f.ClientSet)).WithTimeout(5*time.Second).Should(matchNode, "ResourceSlice from kubelet plugin")

			ginkgo.By("stop plugin and wait for ResourceSlice removal")
			kubeletPlugin.Stop()
			gomega.Eventually(ctx, listResources(f.ClientSet)).Should(gomega.BeEmpty(), "ResourceSlices with no plugin")
			gomega.Consistently(ctx, listResources(f.ClientSet)).WithTimeout(5*time.Second).Should(gomega.BeEmpty(), "ResourceSlices with no plugin")
		})

		removedIfPluginStopsAfterRegistration := func(ctx context.Context, datadir string, opts ...any) {
			nodeName := getNodeName(ctx, f)

			ginkgo.By("start DRA registrar")
			registrar := newRegistrar(ctx, f.ClientSet, nodeName, driverName, opts...)

			ginkgo.By("wait for registration to complete")
			gomega.Eventually(registrar.GetGRPCCalls).WithTimeout(pluginRegistrationTimeout).Should(testdrivergomega.BeRegistered)

			ginkgo.By("start DRA plugin service")
			kubeletPlugin := newDRAService(ctx, f.ClientSet, f.Namespace.Name, nodeName, driverName, datadir, opts...)

			ginkgo.By("wait for ResourceSlice to be created by plugin")
			matchNode := gomega.ConsistOf(matchResourcesByNodeName(nodeName))
			gomega.Eventually(ctx, listResources(f.ClientSet)).Should(matchNode, "ResourceSlices")
			gomega.Consistently(ctx, listResources(f.ClientSet)).WithTimeout(5*time.Second).Should(matchNode, "ResourceSlices")

			ginkgo.By("stop plugin")
			kubeletPlugin.Stop()

			ginkgo.By("wait for ResourceSlice removal")
			gomega.Eventually(ctx, listResources(f.ClientSet)).Should(gomega.BeEmpty(), "ResourceSlices")
			gomega.Consistently(ctx, listResources(f.ClientSet)).WithTimeout(5*time.Second).Should(gomega.BeEmpty(), "ResourceSlices")
		}
		ginkgo.DescribeTable("must be removed if plugin stops after registration",
			removedIfPluginStopsAfterRegistration,
			ginkgo.Entry("2 sockets", ""),
			ginkgo.Entry(
				"1 common socket",
				kubeletplugin.KubeletRegistryDir,
				kubeletplugin.PluginDataDirectoryPath(kubeletplugin.KubeletRegistryDir),
				kubeletplugin.PluginSocket(driverName+"-common.sock"),
			),
		)

		f.It("must be removed if plugin is unresponsive after registration", func(ctx context.Context) {
			nodeName := getNodeName(ctx, f)

			ginkgo.By("start DRA registrar")
			registrar := newRegistrar(ctx, f.ClientSet, nodeName, driverName)
			ginkgo.By("wait for registration to complete")
			gomega.Eventually(registrar.GetGRPCCalls).WithTimeout(pluginRegistrationTimeout).Should(testdrivergomega.BeRegistered)

			ginkgo.By("create a ResourceSlice")
			createTestResourceSlice(ctx, f.ClientSet, nodeName, driverName)
			gomega.Eventually(ctx, listResources(f.ClientSet)).Should(gomega.ConsistOf(matchResourcesByNodeName(nodeName)), "ResourceSlices without plugin")

			ginkgo.By("wait for ResourceSlice removal")
			gomega.Eventually(ctx, listResources(f.ClientSet)).Should(gomega.BeEmpty(), "ResourceSlices without plugin")
			gomega.Consistently(ctx, listResources(f.ClientSet)).WithTimeout(5*time.Second).Should(gomega.BeEmpty(), "ResourceSlices without plugin")
		})

		testRemoveIfRestartsQuickly := func(ctx context.Context, datadir string, opts ...any) {
			nodeName := getNodeName(ctx, f)

			ginkgo.By("start DRA registrar")
			registrar := newRegistrar(ctx, f.ClientSet, nodeName, driverName, opts...)

			ginkgo.By("wait for registration to complete")
			gomega.Eventually(registrar.GetGRPCCalls).WithTimeout(pluginRegistrationTimeout).Should(testdrivergomega.BeRegistered)

			ginkgo.By("start DRA plugin service")
			kubeletPlugin := newDRAService(ctx, f.ClientSet, f.Namespace.Name, nodeName, driverName, datadir, opts...)

			ginkgo.By("wait for ResourceSlice to be created by plugin")
			matchNode := gomega.ConsistOf(matchResourcesByNodeName(nodeName))
			gomega.Eventually(ctx, listResources(f.ClientSet)).Should(matchNode, "ResourceSlices")
			var slices []resourceapi.ResourceSlice
			gomega.Consistently(ctx, listAndStoreResources(f.ClientSet, &slices)).WithTimeout(5*time.Second).Should(matchNode, "ResourceSlices")

			ginkgo.By("stop plugin")
			kubeletPlugin.Stop()

			// We know from the "must be removed if plugin is unresponsive after registration" that the kubelet
			// eventually notices the dropped connection. We cannot observe when that happens, we would need
			// a new metric for that ("registered DRA plugins"). Let's give it a few seconds, which is significantly
			// less than the wiping delay.
			time.Sleep(5 * time.Second)

			ginkgo.By("restarting plugin")
			newDRAService(ctx, f.ClientSet, f.Namespace.Name, nodeName, driverName, datadir, opts...)

			ginkgo.By("ensuring unchanged ResourceSlices")
			gomega.Consistently(ctx, listResources(f.ClientSet)).WithTimeout(time.Minute).Should(gomega.Equal(slices), "ResourceSlices")
		}
		ginkgo.DescribeTable("must not be removed if plugin restarts quickly enough",
			testRemoveIfRestartsQuickly,
			ginkgo.Entry("2 sockets", ""),
			ginkgo.Entry(
				"1 common socket",
				kubeletplugin.KubeletRegistryDir,
				kubeletplugin.PluginDataDirectoryPath(kubeletplugin.KubeletRegistryDir),
				kubeletplugin.PluginSocket(driverName+"-common.sock"),
			),
		)
	})

	f.Context("Resource Health", framework.WithFeatureGate(features.ResourceHealthStatus), f.WithSerial(), func() {

		// Verifies that device health transitions (Healthy -> Unhealthy -> Healthy)
		// reported by a DRA plugin are correctly reflected in the Pod's status.
		ginkgo.It("should reflect device health changes in the Pod's status", func(ctx context.Context) {
			ginkgo.By("Starting the test driver with channel-based control")
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, f.Namespace.Name, getNodeName(ctx, f), driverName)

			pod, claimName, poolNameForTest, deviceNameForTest := setupAndVerifyHealthyPod(
				ctx, f, kubeletPlugin, driverName,
				"health-test-class", "health-test-claim", "health-test-pod", "pool-a", "dev-0",
			)

			ginkgo.By("Setting device health to Unhealthy via control channel")
			kubeletPlugin.HealthControlChan <- testdriver.DeviceHealthUpdate{
				PoolName:   poolNameForTest,
				DeviceName: deviceNameForTest,
				Health:     "Unhealthy",
			}

			ginkgo.By("Verifying device health is now Unhealthy")
			gomega.Eventually(ctx, func(ctx context.Context) (string, error) {
				return getDeviceHealthFromAPIServer(f, pod.Namespace, pod.Name, driverName, claimName, poolNameForTest, deviceNameForTest)
			}).WithTimeout(60*time.Second).WithPolling(2*time.Second).Should(gomega.Equal("Unhealthy"), "Device health should update to Unhealthy")

			ginkgo.By("Setting device health back to Healthy via control channel")
			kubeletPlugin.HealthControlChan <- testdriver.DeviceHealthUpdate{
				PoolName:   poolNameForTest,
				DeviceName: deviceNameForTest,
				Health:     "Healthy",
			}

			ginkgo.By("Verifying device health has recovered to Healthy")
			gomega.Eventually(ctx, func(ctx context.Context) (string, error) {
				return getDeviceHealthFromAPIServer(f, pod.Namespace, pod.Name, driverName, claimName, poolNameForTest, deviceNameForTest)
			}).WithTimeout(60*time.Second).WithPolling(2*time.Second).Should(gomega.Equal("Healthy"), "Device health should recover and update to Healthy")
		})

		// This test verifies that the Kubelet establishes only a single gRPC connection
		// with the DRA plugin throughout the plugin lifecycle.
		//
		// It does so by:
		// - Using a custom gRPC listener that counts accepted connections.
		// - Sequentially creating pods that trigger NodePrepareResources and NodeUnprepareResources.
		// - Asserting that only one connection was accepted by the listener — since each
		// call to net.Listener.Accept() corresponds to a new incoming connection —
		// ensures that the plugin reused the same gRPC connection for all calls.
		//
		// NOTE: This is a copy of a similar test to test service connection reuse with enabled
		// ResourceHealth. It should be merged with the original test when the ResourceHealth feature matures.
		ginkgo.It("must reuse one gRPC connection for service and health-monitoring calls", func(ctx context.Context) {
			var listener *countingListener
			var err error
			getListener := func(_ context.Context, socketPath string) (net.Listener, error) {
				listener, err = newCountingListener(socketPath)
				return listener, err
			}

			nodeName := getNodeName(ctx, f)

			ginkgo.By("start DRA registrar")
			registrar := newRegistrar(ctx, f.ClientSet, nodeName, driverName)

			ginkgo.By("wait for registration to complete")
			gomega.Eventually(registrar.GetGRPCCalls).WithTimeout(pluginRegistrationTimeout).Should(testdrivergomega.BeRegistered)

			ginkgo.By("start DRA plugin service")
			draService := newDRAService(ctx, f.ClientSet, f.Namespace.Name, nodeName, driverName, "", kubeletplugin.PluginListener(getListener))

			for _, suffix := range []string{"-1", "-2"} {
				draService.ResetGRPCCalls()

				setupAndVerifyHealthyPod(
					ctx, f, draService, driverName,
					"health-test-class"+suffix,
					"health-test-claim"+suffix,
					"health-test-pod"+suffix,
					"pool-a"+suffix,
					"dev-0"+suffix,
				)

				ginkgo.By("check that listener.Accept was called only once " + suffix)
				gomega.Expect(listener.acceptCount()).To(gomega.Equal(1))
			}
		})

		// Verifies that device health transitions to "Unknown" when a DRA plugin
		// stops and recovers to "Healthy" upon plugin restart.
		ginkgo.It("should update health to Unknown when plugin stops and recover upon restart", func(ctx context.Context) {
			ginkgo.By("Starting the test driver")
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, f.Namespace.Name, getNodeName(ctx, f), driverName)

			pod, claimName, poolNameForTest, deviceNameForTest := setupAndVerifyHealthyPod(
				ctx, f, kubeletPlugin, driverName,
				"unknown-test-class", "unknown-test-claim", "unknown-test-pod", "pool-b", "dev-1",
			)

			ginkgo.By("Stopping the DRA plugin to simulate a crash")
			kubeletPlugin.Stop()

			ginkgo.By("Verifying device health transitions to 'Unknown'")
			gomega.Eventually(ctx, func(ctx context.Context) (string, error) {
				return getDeviceHealthFromAPIServer(f, pod.Namespace, pod.Name, driverName, claimName, poolNameForTest, deviceNameForTest)
			}).WithTimeout(2*time.Minute).WithPolling(5*time.Second).Should(gomega.Equal("Unknown"), "Device health should become Unknown after plugin stops")

			ginkgo.By("Restarting the DRA plugin to simulate recovery")
			// Re-initialize the plugin, which will re-register with the Kubelet.
			kubeletPlugin = newKubeletPlugin(ctx, f.ClientSet, f.Namespace.Name, getNodeName(ctx, f), driverName)

			ginkgo.By("Forcing a 'Healthy' status update after restart")
			kubeletPlugin.HealthControlChan <- testdriver.DeviceHealthUpdate{
				PoolName:   poolNameForTest,
				DeviceName: deviceNameForTest,
				Health:     "Healthy",
			}

			ginkgo.By("Verifying device health recovers to 'Healthy'")
			gomega.Eventually(ctx, func(ctx context.Context) (string, error) {
				return getDeviceHealthFromAPIServer(f, pod.Namespace, pod.Name, driverName, claimName, poolNameForTest, deviceNameForTest)
			}).WithTimeout(60*time.Second).WithPolling(2*time.Second).Should(gomega.Equal("Healthy"), "Device health should recover to Healthy after plugin restarts")
		})

		// Reconnection verification after connection drop
		ginkgo.It("should automatically reconnect both DRA and Health API after connection drop", func(ctx context.Context) {
			ginkgo.By("Starting the test driver")
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, f.Namespace.Name, getNodeName(ctx, f), driverName)

			ginkgo.By("Creating initial pod to establish connection")
			pod, claimName, poolNameForTest, deviceNameForTest := setupAndVerifyHealthyPod(
				ctx, f, kubeletPlugin, driverName,
				"reconnect-test-class", "reconnect-test-claim", "reconnect-test-pod", "pool-reconnect", "dev-reconnect-0",
			)

			ginkgo.By("Simulating connection drop by stopping the plugin")
			kubeletPlugin.Stop()

			ginkgo.By("Verifying health transitions to Unknown after connection drop")
			gomega.Eventually(ctx, func(ctx context.Context) (string, error) {
				return getDeviceHealthFromAPIServer(f, pod.Namespace, pod.Name, driverName, claimName, poolNameForTest, deviceNameForTest)
			}).WithTimeout(2*time.Minute).WithPolling(5*time.Second).Should(gomega.Equal("Unknown"),
				"Health should become Unknown when plugin connection is lost")

			ginkgo.By("Restarting the plugin to simulate reconnection")
			kubeletPlugin = newKubeletPlugin(ctx, f.ClientSet, f.Namespace.Name, getNodeName(ctx, f), driverName)

			ginkgo.By("Waiting for plugin registration after restart")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(pluginRegistrationTimeout).Should(testdrivergomega.BeRegistered,
				"Plugin should re-register after restart")

			ginkgo.By("Creating a new pod to trigger NodePrepareResources after reconnection")
			pod2, claimName2, poolNameForTest2, deviceNameForTest2 := setupAndVerifyHealthyPod(
				ctx, f, kubeletPlugin, driverName,
				"reconnect-test-class-2", "reconnect-test-claim-2", "reconnect-test-pod-2", "pool-reconnect-2", "dev-reconnect-1",
			)

			ginkgo.By("Verifying health updates continue to work")
			kubeletPlugin.HealthControlChan <- testdriver.DeviceHealthUpdate{
				PoolName:   poolNameForTest2,
				DeviceName: deviceNameForTest2,
				Health:     "Unhealthy",
			}

			gomega.Eventually(ctx, func(ctx context.Context) (string, error) {
				return getDeviceHealthFromAPIServer(f, pod2.Namespace, pod2.Name, driverName, claimName2, poolNameForTest2, deviceNameForTest2)
			}).WithTimeout(60*time.Second).WithPolling(2*time.Second).Should(gomega.Equal("Unhealthy"),
				"Health API should continue working after reconnection")
		})

		// Concurrent operations verification
		ginkgo.It("should handle concurrent DRA operations and health monitoring without connection issues", func(ctx context.Context) {
			ginkgo.By("Starting the test driver")
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, f.Namespace.Name, getNodeName(ctx, f), driverName)

			numPods := 3
			pods := make([]*v1.Pod, numPods)
			claimNames := make([]string, numPods)
			poolNames := make([]string, numPods)
			deviceNames := make([]string, numPods)

			ginkgo.By(fmt.Sprintf("Creating %d pods concurrently to stress test connection management", numPods))
			for i := 0; i < numPods; i++ {
				className := fmt.Sprintf("concurrent-class-%d", i)
				claimNames[i] = fmt.Sprintf("concurrent-claim-%d", i)
				podName := fmt.Sprintf("concurrent-pod-%d", i)
				poolNames[i] = fmt.Sprintf("pool-concurrent-%d", i)
				deviceNames[i] = fmt.Sprintf("dev-concurrent-%d", i)

				pods[i] = createHealthTestPodAndClaim(ctx, f, driverName, podName, claimNames[i], className, poolNames[i], deviceNames[i])
			}

			ginkgo.By("Waiting for all pods to be running")
			for i, pod := range pods {
				framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod),
					fmt.Sprintf("Pod %d should be running", i))
			}

			ginkgo.By("Verifying NodePrepareResources was called for all pods")
			gomega.Eventually(func() int {
				return kubeletPlugin.CountCalls("/NodePrepareResources")
			}).WithTimeout(retryTestTimeout).Should(gomega.BeNumerically(">=", numPods),
				"NodePrepareResources should be called at least once per pod")

			ginkgo.By("Sending health updates for all devices concurrently")
			for i := 0; i < numPods; i++ {
				kubeletPlugin.HealthControlChan <- testdriver.DeviceHealthUpdate{
					PoolName:   poolNames[i],
					DeviceName: deviceNames[i],
					Health:     "Healthy",
				}
			}

			ginkgo.By("Verifying all health updates are correctly reflected")
			for i := 0; i < numPods; i++ {
				pod := pods[i]
				poolName := poolNames[i]
				deviceName := deviceNames[i]
				claimName := claimNames[i]

				gomega.Eventually(ctx, func(ctx context.Context) (string, error) {
					return getDeviceHealthFromAPIServer(f, pod.Namespace, pod.Name, driverName, claimName, poolName, deviceName)
				}).WithTimeout(60*time.Second).WithPolling(2*time.Second).Should(gomega.Equal("Healthy"),
					fmt.Sprintf("Health for device %s should be Healthy", deviceName))
			}

			ginkgo.By("Changing health status for all devices to verify continued operation")
			for i := 0; i < numPods; i++ {
				kubeletPlugin.HealthControlChan <- testdriver.DeviceHealthUpdate{
					PoolName:   poolNames[i],
					DeviceName: deviceNames[i],
					Health:     "Unhealthy",
				}
			}

			ginkgo.By("Verifying all health changes are correctly reflected")
			for i := 0; i < numPods; i++ {
				pod := pods[i]
				poolName := poolNames[i]
				deviceName := deviceNames[i]
				claimName := claimNames[i]

				gomega.Eventually(ctx, func(ctx context.Context) (string, error) {
					return getDeviceHealthFromAPIServer(f, pod.Namespace, pod.Name, driverName, claimName, poolName, deviceName)
				}).WithTimeout(60*time.Second).WithPolling(2*time.Second).Should(gomega.Equal("Unhealthy"),
					fmt.Sprintf("Health for device %s should be Unhealthy", deviceName))
			}
		})
	})

	// This matches the "Resource Health" context above, except that it contains tests which need to run
	// when the feature gate is disabled. This has to be checked at runtime because there is no generic
	// way to filter out such tests in advance.
	f.Context("Resource Health", f.WithSerial(), func() {

		ginkgo.BeforeEach(func() {
			if e2eskipper.IsFeatureGateEnabled(features.ResourceHealthStatus) {
				e2eskipper.Skipf("feature %s is enabled", features.ResourceHealthStatus)
			}
		})

		// Verifies that the Kubelet adds no health status to the Pod when the
		// ResourceHealthStatus feature gate is disabled.
		ginkgo.It("should not add health status to Pod when feature gate is disabled", func(ctx context.Context) {

			ginkgo.By("Starting a test driver")
			newKubeletPlugin(ctx, f.ClientSet, f.Namespace.Name, getNodeName(ctx, f), driverName, withHealthService(false))

			className := "gate-disabled-class"
			claimName := "gate-disabled-claim"
			podName := "gate-disabled-pod"
			poolNameForTest := "pool-d"
			deviceNameForTest := "dev-3"

			pod := createHealthTestPodAndClaim(ctx, f, driverName, podName, claimName, className, poolNameForTest, deviceNameForTest)

			ginkgo.By("Waiting for the pod to be running")
			framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod))

			ginkgo.By("Consistently verifying that the allocatedResourcesStatus field remains absent")
			gomega.Consistently(func(ctx context.Context) error {
				p, err := f.ClientSet.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
				if err != nil {
					return err
				}
				for _, containerStatus := range p.Status.ContainerStatuses {
					if containerStatus.Name == "testcontainer" {
						if len(containerStatus.AllocatedResourcesStatus) != 0 {
							return fmt.Errorf("expected allocatedResourcesStatus to be absent, but found %d entries", len(containerStatus.AllocatedResourcesStatus))
						}
						return nil
					}
				}
				return fmt.Errorf("could not find container 'testcontainer' in pod status")
			}).WithContext(ctx).WithTimeout(30*time.Second).WithPolling(2*time.Second).Should(gomega.Succeed(), "The allocatedResourcesStatus field should be absent when the feature gate is disabled")
		})

		f.Context("Device ShareID", framework.WithFeatureGate(features.DRAConsumableCapacity), f.WithSerial(), func() {

			ginkgo.BeforeEach(func() {
				if e2eskipper.IsFeatureGateEnabled(features.DRAConsumableCapacity) {
					e2eskipper.Skipf("feature %s is enabled", features.DRAConsumableCapacity)
				}
			})

			testWithDeviceRequestAllocationResults := func(ctx context.Context, results []resourceapi.DeviceRequestAllocationResult) {
				nodeName := getNodeName(ctx, f)
				kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, f.Namespace.Name, getNodeName(ctx, f), driverName)

				ginkgo.By("wait for registration to complete")
				gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(pluginRegistrationTimeout).Should(testdrivergomega.BeRegistered)

				pod := createTestObjects(ctx, f.ClientSet, nodeName, f.Namespace.Name, "draclass", "external-claim", "drapod", false, []string{driverName}, results...)

				ginkgo.By("wait for NodePrepareResources call to succeed with given allocation results")
				gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodePrepareResourcesSucceeded)

				ginkgo.By("wait for pod to succeed")
				err := e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
				framework.ExpectNoError(err)

				ginkgo.By("delete pod")
				e2epod.DeletePodOrFail(ctx, f.ClientSet, f.Namespace.Name, pod.Name)

				ginkgo.By("wait for NodeUnprepareResources call to succeed")
				gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodeUnprepareResourcesSucceeded)
			}
			ginkgo.DescribeTable("must be functional when plugin starts to listen on a service socket after registration with the specified allocation results",
				testWithDeviceRequestAllocationResults,
				ginkgo.Entry("without ShareID", nil),
				ginkgo.Entry("with a ShareID", []resourceapi.DeviceRequestAllocationResult{
					{
						Driver:      driverName,
						Pool:        "some-pool",
						Device:      "some-device",
						Request:     "my-request",
						ShareID:     ptr.To(uuid.NewUUID()),
						AdminAccess: ptr.To(false),
					},
				},
				),
				ginkgo.Entry("same device with two ShareIDs", []resourceapi.DeviceRequestAllocationResult{
					{
						Driver:      driverName,
						Pool:        "some-pool",
						Device:      "some-device",
						Request:     "my-request",
						ShareID:     ptr.To(uuid.NewUUID()),
						AdminAccess: ptr.To(false),
					},
					{
						Driver:      driverName,
						Pool:        "some-pool",
						Device:      "some-device",
						Request:     "my-request",
						ShareID:     ptr.To(uuid.NewUUID()),
						AdminAccess: ptr.To(false),
					},
				},
				),
			)

		})
	})
})

// pluginOption defines a functional option for configuring the test driver.
type pluginOption func(*testdriver.Options)

// withHealthService is a pluginOption to explicitly enable or disable the health service.
func withHealthService(enabled bool) pluginOption {
	return func(o *testdriver.Options) {
		o.EnableHealthService = enabled
	}
}

// Run Kubelet plugin and wait until it's registered
func newKubeletPlugin(ctx context.Context, clientSet kubernetes.Interface, testNamespace string, nodeName, driverName string, options ...pluginOption) *testdriver.ExamplePlugin {
	ginkgo.By("start Kubelet plugin")
	logger := klog.LoggerWithValues(klog.LoggerWithName(klog.Background(), "DRA kubelet plugin "+driverName), "node", nodeName)
	ctx = klog.NewContext(ctx, logger)

	// Ensure that directories exist, creating them if necessary. We want
	// to know early if there is a setup problem that would prevent
	// creating those directories.
	err := os.MkdirAll(cdiDir, os.FileMode(0750))
	framework.ExpectNoError(err, "create CDI directory")
	datadir := path.Join(kubeletplugin.KubeletPluginsDir, driverName) // The default, not set below.
	err = os.MkdirAll(datadir, 0750)
	framework.ExpectNoError(err, "create DRA socket directory")

	pluginOpts := testdriver.Options{
		EnableHealthService: true,
	}
	for _, option := range options {
		option(&pluginOpts)
	}

	plugin, err := testdriver.StartPlugin(
		context.WithoutCancel(ctx), // Stopping is handled in draServiceCleanup.
		cdiDir,
		driverName,
		clientSet,
		nodeName,
		testdriver.FileOperations{
			DriverResources: &resourceslice.DriverResources{
				Pools: map[string]resourceslice.Pool{
					nodeName: {
						Slices: []resourceslice.Slice{{
							Devices: []resourceapi.Device{
								{
									Name: "device-00",
								},
							},
						}},
					},
				},
			},
		},
		pluginOpts,
	)
	framework.ExpectNoError(err)

	ginkgo.DeferCleanup(func(ctx context.Context) { draServiceCleanup(ctx, clientSet, driverName, testNamespace, plugin) })

	gomega.Eventually(plugin.GetGRPCCalls).WithTimeout(pluginRegistrationTimeout).Should(testdrivergomega.BeRegistered)

	return plugin
}

// newRegistrar starts a registrar for the specified DRA driver, without the DRA gRPC service.
func newRegistrar(ctx context.Context, clientSet kubernetes.Interface, nodeName, driverName string, opts ...any) *testdriver.ExamplePlugin {
	ginkgo.By("start only Kubelet plugin registrar")
	logger := klog.LoggerWithValues(klog.LoggerWithName(klog.Background(), "kubelet plugin registrar "+driverName))
	ctx = klog.NewContext(ctx, logger)

	allOpts := []any{
		testdriver.Options{EnableHealthService: false},
		kubeletplugin.DRAService(false),
	}

	allOpts = append(allOpts, opts...)

	registrar, err := testdriver.StartPlugin(
		context.WithoutCancel(ctx), // It gets stopped during test cleanup or when requested earlier.
		cdiDir,
		driverName,
		clientSet,
		nodeName,
		testdriver.FileOperations{},
		allOpts...,
	)
	framework.ExpectNoError(err, "start only Kubelet plugin registrar")
	ginkgo.DeferCleanup(registrar.Stop)
	return registrar
}

// newDRAService starts the DRA gRPC service for the specified node and driver.
// It ensures that necessary directories exist, starts the plugin and registers
// cleanup functions to remove created resources after the test.
// Parameters:
//   - ctx: The context for controlling cancellation and logging.
//   - clientSet: Kubernetes client interface for interacting with the cluster.
//   - nodeName: The name of the node where the plugin will run.
//   - driverName: The name of the DRA driver.
//   - datadir: The directory for the DRA socket and state files.
//     Must match what is specified via [kubeletplugin.PluginDataDirectoryPath].
//     May be empty if that option is not used, then the default path is used.
//   - opts: Additional options for plugin configuration.
//
// Returns:
//   - A pointer to the started ExamplePlugin instance.
func newDRAService(ctx context.Context, clientSet kubernetes.Interface, testNamespace, nodeName, driverName, datadir string, opts ...any) *testdriver.ExamplePlugin {
	ginkgo.By("start only Kubelet plugin")
	logger := klog.LoggerWithValues(klog.LoggerWithName(klog.Background(), "kubelet plugin "+driverName), "node", nodeName)
	ctx = klog.NewContext(ctx, logger)

	allOpts := []any{
		testdriver.Options{EnableHealthService: true},
		kubeletplugin.RegistrationService(false),
	}
	allOpts = append(allOpts, opts...)

	// Ensure that directories exist, creating them if necessary. We want
	// to know early if there is a setup problem that would prevent
	// creating those directories.
	err := os.MkdirAll(cdiDir, os.FileMode(0750))
	framework.ExpectNoError(err, "create CDI directory")

	// If datadir is not provided, set it to the default and ensure it exists.
	if datadir == "" {
		datadir = path.Join(kubeletplugin.KubeletPluginsDir, driverName)
	}

	err = os.MkdirAll(datadir, 0750)
	framework.ExpectNoError(err, "create DRA socket directory")

	plugin, err := testdriver.StartPlugin(
		context.WithoutCancel(ctx), // Stopping is handled in draServiceCleanup.
		cdiDir,
		driverName,
		clientSet,
		nodeName,
		testdriver.FileOperations{
			DriverResources: &resourceslice.DriverResources{
				Pools: map[string]resourceslice.Pool{
					nodeName: {
						Slices: []resourceslice.Slice{{
							Devices: []resourceapi.Device{
								{
									Name: "device-00",
								},
							},
						}},
					},
				},
			},
		},
		allOpts...,
	)
	framework.ExpectNoError(err)

	ginkgo.DeferCleanup(func(ctx context.Context) { draServiceCleanup(ctx, clientSet, driverName, testNamespace, plugin) })

	return plugin
}

func draServiceCleanup(ctx context.Context, clientSet kubernetes.Interface, driverName string, testNamespace string, plugin *testdriver.ExamplePlugin) {
	// Always stop at the end, even in the case of failures.
	defer func() {
		ginkgo.By("Stopping the driver and deleting ResourceSlices...")
		plugin.Stop()

		// kubelet should do this eventually, but better make sure.
		// A separate test checks this explicitly.
		framework.ExpectNoError(clientSet.ResourceV1().ResourceSlices().DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{FieldSelector: resourceapi.ResourceSliceSelectorDriver + "=" + driverName}))
	}()

	// Some test was leaking an in-use ResourceClaim
	// (https://github.com/kubernetes/kubernetes/issues/133304) because it
	// didn't delete its pod before the driver shutdown. The namespace
	// deletion usually takes care of this, but that happens later and
	// without the driver the pods stay pending. Therefore we delete all of
	// the pods in the test namespace *before* proceeding with the driver
	// shutdown.
	ginkgo.By("Deleting pods and waiting for ResourceClaims to be released...")
	framework.ExpectNoError(clientSet.CoreV1().Pods(testNamespace).DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{}))

	// Failing here is a bit more obvious than in the metric check below.
	// The metrics get checked, too, because we want to be certain that those go down.
	gomega.Eventually(ctx, framework.ListObjects(clientSet.CoreV1().Pods(testNamespace).List, metav1.ListOptions{})).Should(gomega.HaveField("Items", gomega.BeEmpty()), "All pods should be deleted, shut down and removed.")

	// We allow for a delay because it takes a while till the kubelet fully deals with pod deletion.
	// Typically there is no dra_resource_claims_in_use entry for the driver (covered by gstruct.IgnoreMissing).
	// If there is one, it must be zero.
	gomega.Eventually(ctx, getKubeletMetrics).Should(gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
		"dra_resource_claims_in_use": gstruct.MatchElements(driverNameLabelKey, gstruct.IgnoreExtras|gstruct.IgnoreMissing, gstruct.Elements{
			driverName: timelessSample(0),
		}),
	}), "The test should have deleted all pods.\nInstead some claims prepared by the driver %s are still in use at the time when the driver gets uninstalled.", driverName)
}

// createTestObjects creates objects required by the test
// NOTE: as scheduler and controller manager are not running by the Node e2e,
// the objects must contain all required data to be processed correctly by the API server
// and placed on the node without involving the scheduler and the DRA controller.
//
// Instead adding more parameters, the podName determines what the pod does.
func createTestObjects(ctx context.Context, clientSet kubernetes.Interface, nodename, namespace, className, claimName, podName string, deferPodDeletion bool, driverNames []string,
	deviceRequestResults ...resourceapi.DeviceRequestAllocationResult) *v1.Pod {
	// DeviceClass
	class := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: className,
		},
	}
	_, err := clientSet.ResourceV1().DeviceClasses().Create(ctx, class, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	ginkgo.DeferCleanup(clientSet.ResourceV1().DeviceClasses().Delete, className, metav1.DeleteOptions{})

	// ResourceClaim
	podClaimName := "resource-claim"
	claim := &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: claimName,
		},
		Spec: resourceapi.ResourceClaimSpec{
			Devices: resourceapi.DeviceClaim{
				Requests: []resourceapi.DeviceRequest{{
					Name: "my-request",
					Exactly: &resourceapi.ExactDeviceRequest{
						DeviceClassName: className,
					},
				}},
			},
		},
	}
	createdClaim, err := clientSet.ResourceV1().ResourceClaims(namespace).Create(ctx, claim, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	ginkgo.DeferCleanup(clientSet.ResourceV1().ResourceClaims(namespace).Delete, claimName, metav1.DeleteOptions{})

	// The pod checks its own env with grep. Each driver injects its own parameters,
	// with the driver name as part of the variable name. Sorting ensures that a
	// single grep can match the output of env when that gets turned into a single
	// line because the order is deterministic.
	nameToEnv := func(driverName string) string {
		return "DRA_" + regexp.MustCompile(`[^a-z0-9]`).ReplaceAllString(driverName, "_")
	}
	var expectedEnv []string
	sort.Strings(driverNames)
	for _, driverName := range driverNames {
		expectedEnv = append(expectedEnv, nameToEnv(driverName)+"=PARAM1_VALUE")
	}
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
					// If injecting env variables fails, the pod fails and this error shows up in
					// ... Terminated:&ContainerStateTerminated{ExitCode:1,Signal:0,Reason:Error,Message:ERROR: ...
					Command: []string{"/bin/sh", "-c", "if ! echo $(env) | grep -q " + strings.Join(expectedEnv, ".*") + "; then echo ERROR: unexpected env: $(env) >/dev/termination-log; exit 1 ; fi"},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
	if podName == "drasleeppod" {
		// As above, plus infinite sleep.
		pod.Spec.Containers[0].Command[2] += "&& sleep 100000"
	}
	createdPod, err := clientSet.CoreV1().Pods(namespace).Create(ctx, pod, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	if deferPodDeletion {
		ginkgo.DeferCleanup(clientSet.CoreV1().Pods(namespace).Delete, podName, metav1.DeleteOptions{})
	}

	// Update claim status: set ReservedFor and AllocationResult
	// NOTE: This is usually done by the DRA controller or the scheduler.

	var results []resourceapi.DeviceRequestAllocationResult
	if len(deviceRequestResults) > 0 {
		// Specify DeviceRequestAllocationResult to test specific condition of allocation result such as ShareID
		results = deviceRequestResults
	} else {
		// Generate `some-device` DeviceRequestAllocationResult for each driver
		results = make([]resourceapi.DeviceRequestAllocationResult, len(driverNames))
		for i, driverName := range driverNames {
			results[i] = resourceapi.DeviceRequestAllocationResult{
				Driver:      driverName,
				Pool:        "some-pool",
				Device:      "some-device",
				Request:     claim.Spec.Devices.Requests[0].Name,
				AdminAccess: ptr.To(false),
			}
		}
	}

	config := make([]resourceapi.DeviceAllocationConfiguration, len(driverNames))

	for i, driverName := range driverNames {
		config[i] = resourceapi.DeviceAllocationConfiguration{
			Source: resourceapi.AllocationConfigSourceClaim,
			DeviceConfiguration: resourceapi.DeviceConfiguration{
				Opaque: &resourceapi.OpaqueDeviceConfiguration{
					Driver:     driverName,
					Parameters: runtime.RawExtension{Raw: []byte(`{"` + nameToEnv(driverName) + `":"PARAM1_VALUE"}`)},
				},
			},
		}
	}

	createdClaim.Status = resourceapi.ResourceClaimStatus{
		ReservedFor: []resourceapi.ResourceClaimConsumerReference{
			{Resource: "pods", Name: podName, UID: createdPod.UID},
		},
		Allocation: &resourceapi.AllocationResult{
			Devices: resourceapi.DeviceAllocationResult{
				Results: results,
				Config:  config,
			},
		},
	}
	_, err = clientSet.ResourceV1().ResourceClaims(namespace).UpdateStatus(ctx, createdClaim, metav1.UpdateOptions{})
	framework.ExpectNoError(err)

	return pod
}

func createTestResourceSlice(ctx context.Context, clientSet kubernetes.Interface, nodeName, driverName string) {
	slice := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: nodeName,
		},
		Spec: resourceapi.ResourceSliceSpec{
			NodeName: &nodeName,
			Driver:   driverName,
			Pool: resourceapi.ResourcePool{
				Name:               nodeName,
				ResourceSliceCount: 1,
			},
		},
	}

	ginkgo.By(fmt.Sprintf("Creating ResourceSlice %s", nodeName))
	slice, err := clientSet.ResourceV1().ResourceSlices().Create(ctx, slice, metav1.CreateOptions{})
	framework.ExpectNoError(err, "create ResourceSlice")
	ginkgo.DeferCleanup(func(ctx context.Context) {
		ginkgo.By(fmt.Sprintf("Deleting ResourceSlice %s", nodeName))
		err := clientSet.ResourceV1().ResourceSlices().Delete(ctx, slice.Name, metav1.DeleteOptions{})
		if !apierrors.IsNotFound(err) {
			framework.ExpectNoError(err, "delete ResourceSlice")
		}
	})
}

func listResources(client kubernetes.Interface) func(ctx context.Context) ([]resourceapi.ResourceSlice, error) {
	return func(ctx context.Context) ([]resourceapi.ResourceSlice, error) {
		slices, err := client.ResourceV1().ResourceSlices().List(ctx, metav1.ListOptions{})
		if err != nil {
			return nil, err
		}
		return slices.Items, nil
	}
}

func listAndStoreResources(client kubernetes.Interface, lastSlices *[]resourceapi.ResourceSlice) func(ctx context.Context) ([]resourceapi.ResourceSlice, error) {
	return func(ctx context.Context) ([]resourceapi.ResourceSlice, error) {
		slices, err := client.ResourceV1().ResourceSlices().List(ctx, metav1.ListOptions{})
		if err != nil {
			return nil, err
		}
		*lastSlices = slices.Items
		return *lastSlices, nil
	}
}

func matchResourcesByNodeName(nodeName string) types.GomegaMatcher {
	return gomega.HaveField("Spec.NodeName", gstruct.PointTo(gomega.Equal(nodeName)))
}

// This helper function queries the main API server for the pod's status.
func getDeviceHealthFromAPIServer(f *framework.Framework, namespace, podName, driverName, claimName, poolName, deviceName string) (string, error) {
	// Get the Pod object from the API server
	pod, err := f.ClientSet.CoreV1().Pods(namespace).Get(context.Background(), podName, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			return "NotFound", nil
		}
		return "", fmt.Errorf("failed to get pod %s/%s: %w", namespace, podName, err)
	}

	// This is the unique ID for the device based on how Kubelet manager code constructs it.
	expectedResourceID := v1.ResourceID(fmt.Sprintf("%s/%s/%s", driverName, poolName, deviceName))

	expectedResourceStatusNameSimple := v1.ResourceName(fmt.Sprintf("claim:%s", claimName))
	expectedResourceStatusNameWithRequest := v1.ResourceName(fmt.Sprintf("claim:%s/%s", claimName, "my-request"))

	// Loop through container statuses.
	for _, containerStatus := range pod.Status.ContainerStatuses {
		if containerStatus.AllocatedResourcesStatus != nil {
			for _, resourceStatus := range containerStatus.AllocatedResourcesStatus {
				if resourceStatus.Name != expectedResourceStatusNameSimple && resourceStatus.Name != expectedResourceStatusNameWithRequest {
					continue
				}
				for _, resourceHealth := range resourceStatus.Resources {
					if resourceHealth.ResourceID == expectedResourceID || strings.HasPrefix(string(resourceHealth.ResourceID), driverName) {
						return string(resourceHealth.Health), nil
					}
				}
			}
		}
	}

	return "NotFound", nil
}

// createHealthTestPodAndClaim is a specialized helper for the Resource Health test.
// It creates all necessary objects (DeviceClass, ResourceClaim, Pod) and ensures
// the pod is long-running and the claim is allocated from the specified pool.
func createHealthTestPodAndClaim(ctx context.Context, f *framework.Framework, driverName, podName, claimName, className, poolName, deviceName string) *v1.Pod {
	ginkgo.By(fmt.Sprintf("Creating DeviceClass %q", className))
	dc := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: className,
		},
	}
	_, err := f.ClientSet.ResourceV1().DeviceClasses().Create(ctx, dc, metav1.CreateOptions{})
	framework.ExpectNoError(err, "failed to create DeviceClass "+className)
	ginkgo.DeferCleanup(func(ctx context.Context) {
		err := f.ClientSet.ResourceV1().DeviceClasses().Delete(ctx, className, metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			framework.Failf("Failed to delete DeviceClass %s: %v", className, err)
		}
	})
	ginkgo.By(fmt.Sprintf("Creating ResourceClaim %q", claimName))
	claim := &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: claimName,
		},
		Spec: resourceapi.ResourceClaimSpec{
			Devices: resourceapi.DeviceClaim{
				Requests: []resourceapi.DeviceRequest{{
					Name: "my-request",
					Exactly: &resourceapi.ExactDeviceRequest{
						DeviceClassName: className,
					},
				}},
			},
		},
	}

	_, err = f.ClientSet.ResourceV1().ResourceClaims(f.Namespace.Name).Create(ctx, claim, metav1.CreateOptions{})
	framework.ExpectNoError(err, "failed to create ResourceClaim "+claimName)
	ginkgo.DeferCleanup(func(ctx context.Context) {
		err := f.ClientSet.ResourceV1().ResourceClaims(f.Namespace.Name).Delete(ctx, claimName, metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			framework.Failf("Failed to delete ResourceClaim %s: %v", claimName, err)
		}
	})
	ginkgo.By(fmt.Sprintf("Creating long-running Pod %q (without claim allocation yet)", podName))
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: f.Namespace.Name,
		},
		Spec: v1.PodSpec{
			NodeName:      getNodeName(ctx, f),
			RestartPolicy: v1.RestartPolicyNever,
			ResourceClaims: []v1.PodResourceClaim{
				{Name: claimName, ResourceClaimName: &claimName},
			},
			Containers: []v1.Container{
				{
					Name:    "testcontainer",
					Image:   e2epod.GetDefaultTestImage(),
					Command: []string{"/bin/sh", "-c", "sleep 600"},
					Resources: v1.ResourceRequirements{
						Claims: []v1.ResourceClaim{{Name: claimName, Request: "my-request"}},
					},
				},
			},
		},
	}
	// Create the pod on the API server to assign the real UID.
	createdPod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "failed to create Pod "+podName)
	ginkgo.DeferCleanup(func(ctx context.Context) {
		e2epod.DeletePodOrFail(ctx, f.ClientSet, createdPod.Namespace, createdPod.Name)
	})

	// Update the Pod's status to include the ResourceClaimStatuses, mimicking the scheduler.
	podToUpdate, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, createdPod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)
	podToUpdate.Status.ResourceClaimStatuses = []v1.PodResourceClaimStatus{
		{Name: claimName, ResourceClaimName: &claimName},
	}
	_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).UpdateStatus(ctx, podToUpdate, metav1.UpdateOptions{})
	framework.ExpectNoError(err, "failed to update Pod status with ResourceClaimStatuses")

	ginkgo.By(fmt.Sprintf("Allocating claim %q to pod %q with its real UID", claimName, podName))
	// Get the created claim to ensure the latest version before updating.
	claimToUpdate, err := f.ClientSet.ResourceV1().ResourceClaims(f.Namespace.Name).Get(ctx, claimName, metav1.GetOptions{})
	framework.ExpectNoError(err, "failed to get latest version of ResourceClaim "+claimName)

	// Update the claims status to reserve it for the *real* pod UID.
	claimToUpdate.Status = resourceapi.ResourceClaimStatus{
		ReservedFor: []resourceapi.ResourceClaimConsumerReference{
			{Resource: "pods", Name: createdPod.Name, UID: createdPod.UID},
		},
		Allocation: &resourceapi.AllocationResult{
			Devices: resourceapi.DeviceAllocationResult{
				Results: []resourceapi.DeviceRequestAllocationResult{
					{
						Driver:  driverName,
						Pool:    poolName,
						Device:  deviceName,
						Request: "my-request",
					},
				},
			},
		},
	}
	_, err = f.ClientSet.ResourceV1().ResourceClaims(f.Namespace.Name).UpdateStatus(ctx, claimToUpdate, metav1.UpdateOptions{})
	framework.ExpectNoError(err, "failed to update ResourceClaim status for test")

	return createdPod
}

// setupAndVerifyHealthyPod creates a test pod with health monitoring and verifies it reaches a healthy state.
// It encapsulates the repeated pattern of pod creation, waiting for NodePrepareResources,
// pod startup, and health verification that appears throughout the health monitoring tests.
func setupAndVerifyHealthyPod(
	ctx context.Context,
	f *framework.Framework,
	plugin *testdriver.ExamplePlugin,
	driverName string,
	className string,
	claimName string,
	podName string,
	poolName string,
	deviceName string,
) (*v1.Pod, string, string, string) {
	pod := createHealthTestPodAndClaim(ctx, f, driverName, podName, claimName, className, poolName, deviceName)

	ginkgo.By("wait for NodePrepareResources call to succeed")
	gomega.Eventually(plugin.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodePrepareResourcesSucceeded)

	ginkgo.By("Waiting for the pod to be running")
	framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod))

	ginkgo.By("Forcing a 'Healthy' status update to establish a baseline")
	plugin.HealthControlChan <- testdriver.DeviceHealthUpdate{
		PoolName:   poolName,
		DeviceName: deviceName,
		Health:     "Healthy",
	}

	ginkgo.By("Verifying device health is now Healthy in the pod status")
	gomega.Eventually(ctx, func(ctx context.Context) (string, error) {
		return getDeviceHealthFromAPIServer(f, pod.Namespace, pod.Name, driverName, claimName, poolName, deviceName)
	}).WithTimeout(30*time.Second).WithPolling(1*time.Second).Should(gomega.Equal("Healthy"), "Device health should be Healthy after explicit update")

	return pod, claimName, poolName, deviceName
}

// errorOnCloseListener is a mock net.Listener that blocks on Accept()
// until Close() is called, at which point Accept() returns a predefined error.
//
// This is useful in tests or simulated environments to trigger grpc.Server.Serve()
// to exit cleanly with a known error, without needing real network activity.
type errorOnCloseListener struct {
	ch     chan struct{}
	closed atomic.Bool
	err    error
}

// newErrorOnCloseListener creates a new listener that causes Accept to fail
// with the given error after Close is called.
func newErrorOnCloseListener(err error) *errorOnCloseListener {
	return &errorOnCloseListener{
		ch:  make(chan struct{}),
		err: err,
	}
}

// Accept blocks until Close is called, then returns the configured error.
func (l *errorOnCloseListener) Accept() (net.Conn, error) {
	<-l.ch
	return nil, l.err
}

// Close unblocks Accept and causes it to return the configured error.
// It is safe to call multiple times.
func (l *errorOnCloseListener) Close() error {
	if l.closed.Swap(true) {
		return nil // already closed
	}
	close(l.ch)
	return nil
}

// Addr returns a dummy Unix address. Required to satisfy net.Listener.
func (*errorOnCloseListener) Addr() net.Addr {
	return &net.UnixAddr{Name: "errorOnCloseListener", Net: "unix"}
}

// driverNameLabelKey maps metric samples to their driver_name label value.
// Used with gstruct.MatchAllElements.
func driverNameLabelKey(element any) string {
	el := element.(*testutil.Sample)
	return string(el.Metric[testutil.LabelName("driver_name")])
}

// countingListener is a wrapper around net.Listener that counts the number of times
// Accept method is called. It embeds a net.Listener and uses atomic counters
// to safely track the number of accept operations.
type countingListener struct {
	net.Listener
	acceptCalls atomic.Int32
}

// newCountingListener creates a new Unix domain socket listener
// wrapped in a countingListener to track accepted connections.
func newCountingListener(socketPath string) (*countingListener, error) {
	addr, err := net.ResolveUnixAddr("unix", socketPath)
	if err != nil {
		return nil, err
	}

	l, err := net.ListenUnix("unix", addr)
	if err != nil {
		return nil, err
	}

	return &countingListener{Listener: l}, nil
}

// Accept waits for and returns the accepted connection.
// It increments the acceptCalls counter each time the call succeeded.
func (l *countingListener) Accept() (net.Conn, error) {
	conn, err := l.Listener.Accept()
	if err == nil {
		l.acceptCalls.Add(1)
	}
	return conn, err
}

// acceptCount returns the number of times the accept method
// has been called on the listener.
func (l *countingListener) acceptCount() int {
	return int(l.acceptCalls.Load())
}
