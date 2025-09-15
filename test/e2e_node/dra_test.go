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
       TEST_ARGS='--feature-gates="DynamicResourceAllocation=true" --service-feature-gates="DynamicResourceAllocation=true" --runtime-config=api/all=true'
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
	resourceapi "k8s.io/api/resource/v1beta1"
	resourceapiv1beta2 "k8s.io/api/resource/v1beta2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"

	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"

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
			framework.ExpectNoError(f.ClientSet.ResourceV1beta1().ResourceSlices().DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{}))
		})
	})

	f.Context("Resource Kubelet Plugin", f.WithSerial(), func() {
		ginkgo.It("must register after Kubelet restart", func(ctx context.Context) {
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, getNodeName(ctx, f), driverName)

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
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, getNodeName(ctx, f), driverName)

			ginkgo.By("restart Kubelet Plugin")
			kubeletPlugin.Stop()
			kubeletPlugin = newKubeletPlugin(ctx, f.ClientSet, getNodeName(ctx, f), driverName)

			ginkgo.By("wait for Kubelet plugin re-registration")
			gomega.Eventually(kubeletPlugin.GetGRPCCalls).WithTimeout(pluginRegistrationTimeout).Should(testdrivergomega.BeRegistered)
		})

		// Test that the kubelet plugin manager retries plugin registration
		// when the GetInfo call fails, and succeeds once the call passes.
		ginkgo.It("must recover and register after registration failure", func(ctx context.Context) {
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, getNodeName(ctx, f), driverName)

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
			newKubeletPlugin(ctx, f.ClientSet, getNodeName(ctx, f), driverName)

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
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, getNodeName(ctx, f), driverName)

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
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, getNodeName(ctx, f), driverName)

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
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, getNodeName(ctx, f), driverName)

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
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, getNodeName(ctx, f), driverName)

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
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, getNodeName(ctx, f), driverName)

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
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, getNodeName(ctx, f), driverName)

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
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, getNodeName(ctx, f), driverName)

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
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, getNodeName(ctx, f), driverName)

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
			draService := newDRAService(ctx, f.ClientSet, nodeName, driverName, datadir, opts...)

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
			draService := newDRAService(ctx, f.ClientSet, nodeName, driverName, datadir, opts...)

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
			draService = newDRAService(ctx, f.ClientSet, nodeName, driverName, datadir, opts...)

			ginkgo.By("stopping pod")
			err = f.ClientSet.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
			gomega.Eventually(draService.GetGRPCCalls).WithTimeout(retryTestTimeout).Should(testdrivergomega.NodeUnprepareResourcesSucceeded)
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
			service func(ctx context.Context, clientSet kubernetes.Interface, nodeName, driverName, datadir string, opts ...any) *testdriver.ExamplePlugin,
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
		wrappedNewRegistrar := func(ctx context.Context, clientSet kubernetes.Interface, nodeName, driverName, datadir string, opts ...any) *testdriver.ExamplePlugin {
			return newRegistrar(ctx, clientSet, nodeName, driverName, opts...)
		}
		ginkgo.DescribeTable("must report gRPC serving error",
			failOnClosedListener,
			ginkgo.Entry("for registrar", wrappedNewRegistrar, kubeletplugin.RegistrarListener),
			ginkgo.Entry("for DRA service", newDRAService, kubeletplugin.PluginListener),
		)
	})

	f.Context("Two resource Kubelet Plugins", f.WithSerial(), func() {
		// start creates plugins which will get stopped when the context gets canceled.
		start := func(ctx context.Context) (*testdriver.ExamplePlugin, *testdriver.ExamplePlugin) {
			kubeletPlugin1 := newKubeletPlugin(ctx, f.ClientSet, getNodeName(ctx, f), kubeletPlugin1Name)
			kubeletPlugin2 := newKubeletPlugin(ctx, f.ClientSet, getNodeName(ctx, f), kubeletPlugin2Name)

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
			driverName := func(element any) string {
				el := element.(*testutil.Sample)
				return string(el.Metric[testutil.LabelName("driver_name")])
			}

			gomega.Expect(getKubeletMetrics(ctx)).Should(gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"dra_resource_claims_in_use": gstruct.MatchAllElements(driverName, gstruct.Elements{
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
				"dra_resource_claims_in_use": gstruct.MatchAllElements(driverName, gstruct.Elements{
					"<any>": timelessSample(0),
				}),
			}), "metrics while pod is running")
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
			kubeletPlugin := newKubeletPlugin(ctx, f.ClientSet, getNodeName(ctx, f), driverName)
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
			kubeletPlugin := newDRAService(ctx, f.ClientSet, nodeName, driverName, datadir, opts...)

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
			kubeletPlugin := newDRAService(ctx, f.ClientSet, nodeName, driverName, datadir, opts...)

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
			newDRAService(ctx, f.ClientSet, nodeName, driverName, datadir, opts...)

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
})

// Run Kubelet plugin and wait until it's registered
func newKubeletPlugin(ctx context.Context, clientSet kubernetes.Interface, nodeName, driverName string) *testdriver.ExamplePlugin {
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

	plugin, err := testdriver.StartPlugin(
		ctx,
		cdiDir,
		driverName,
		clientSet,
		nodeName,
		testdriver.FileOperations{
			DriverResources: &resourceslice.DriverResources{
				Pools: map[string]resourceslice.Pool{
					nodeName: {
						Slices: []resourceslice.Slice{{
							Devices: []resourceapiv1beta2.Device{
								{
									Name: "device-00",
								},
							},
						}},
					},
				},
			},
		},
	)
	framework.ExpectNoError(err)

	gomega.Eventually(plugin.GetGRPCCalls).WithTimeout(pluginRegistrationTimeout).Should(testdrivergomega.BeRegistered)

	ginkgo.DeferCleanup(func(ctx context.Context) {
		// kubelet should do this eventually, but better make sure.
		// A separate test checks this explicitly.
		framework.ExpectNoError(clientSet.ResourceV1beta1().ResourceSlices().DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{FieldSelector: resourceapi.ResourceSliceSelectorDriver + "=" + driverName}))
	})
	ginkgo.DeferCleanup(plugin.Stop)

	return plugin
}

// newRegistrar starts a registrar for the specified DRA driver, without the DRA gRPC service.
func newRegistrar(ctx context.Context, clientSet kubernetes.Interface, nodeName, driverName string, opts ...any) *testdriver.ExamplePlugin {
	ginkgo.By("start only Kubelet plugin registrar")
	logger := klog.LoggerWithValues(klog.LoggerWithName(klog.Background(), "kubelet plugin registrar "+driverName))
	ctx = klog.NewContext(ctx, logger)
	opts = append(opts, kubeletplugin.DRAService(false))
	registrar, err := testdriver.StartPlugin(
		ctx,
		cdiDir,
		driverName,
		clientSet,
		nodeName,
		testdriver.FileOperations{},
		opts...,
	)
	framework.ExpectNoError(err, "start only Kubelet plugin registrar")
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
func newDRAService(ctx context.Context, clientSet kubernetes.Interface, nodeName, driverName, datadir string, opts ...any) *testdriver.ExamplePlugin {
	ginkgo.By("start only Kubelet plugin")
	logger := klog.LoggerWithValues(klog.LoggerWithName(klog.Background(), "kubelet plugin "+driverName), "node", nodeName)
	ctx = klog.NewContext(ctx, logger)

	opts = append(opts, kubeletplugin.RegistrationService(false))

	// Ensure that directories exist, creating them if necessary. We want
	// to know early if there is a setup problem that would prevent
	// creating those directories.
	err := os.MkdirAll(cdiDir, os.FileMode(0750))
	framework.ExpectNoError(err, "create CDI directory")
	if datadir == "" {
		datadir = path.Join(kubeletplugin.KubeletPluginsDir, driverName)
	}
	err = os.MkdirAll(datadir, 0750)
	framework.ExpectNoError(err, "create DRA socket directory")

	plugin, err := testdriver.StartPlugin(
		ctx,
		cdiDir,
		driverName,
		clientSet,
		nodeName,
		testdriver.FileOperations{
			DriverResources: &resourceslice.DriverResources{
				Pools: map[string]resourceslice.Pool{
					nodeName: {
						Slices: []resourceslice.Slice{{
							Devices: []resourceapiv1beta2.Device{
								{
									Name: "device-00",
								},
							},
						}},
					},
				},
			},
		},
		opts...,
	)
	framework.ExpectNoError(err)

	ginkgo.DeferCleanup(func(ctx context.Context) {
		// kubelet should do this eventually, but better make sure.
		// A separate test checks this explicitly.
		framework.ExpectNoError(clientSet.ResourceV1beta1().ResourceSlices().DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{FieldSelector: resourceapi.ResourceSliceSelectorDriver + "=" + driverName}))
	})
	ginkgo.DeferCleanup(plugin.Stop)

	return plugin
}

// createTestObjects creates objects required by the test
// NOTE: as scheduler and controller manager are not running by the Node e2e,
// the objects must contain all required data to be processed correctly by the API server
// and placed on the node without involving the scheduler and the DRA controller.
//
// Instead adding more parameters, the podName determines what the pod does.
func createTestObjects(ctx context.Context, clientSet kubernetes.Interface, nodename, namespace, className, claimName, podName string, deferPodDeletion bool, driverNames []string) *v1.Pod {
	// DeviceClass
	class := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: className,
		},
	}
	_, err := clientSet.ResourceV1beta1().DeviceClasses().Create(ctx, class, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	ginkgo.DeferCleanup(clientSet.ResourceV1beta1().DeviceClasses().Delete, className, metav1.DeleteOptions{})

	// ResourceClaim
	podClaimName := "resource-claim"
	claim := &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: claimName,
		},
		Spec: resourceapi.ResourceClaimSpec{
			Devices: resourceapi.DeviceClaim{
				Requests: []resourceapi.DeviceRequest{{
					Name:            "my-request",
					DeviceClassName: className,
				}},
			},
		},
	}
	createdClaim, err := clientSet.ResourceV1beta1().ResourceClaims(namespace).Create(ctx, claim, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	ginkgo.DeferCleanup(clientSet.ResourceV1beta1().ResourceClaims(namespace).Delete, claimName, metav1.DeleteOptions{})

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
	results := make([]resourceapi.DeviceRequestAllocationResult, len(driverNames))
	config := make([]resourceapi.DeviceAllocationConfiguration, len(driverNames))
	for i, driverName := range driverNames {
		results[i] = resourceapi.DeviceRequestAllocationResult{
			Driver:      driverName,
			Pool:        "some-pool",
			Device:      "some-device",
			Request:     claim.Spec.Devices.Requests[0].Name,
			AdminAccess: ptr.To(false),
		}
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
	_, err = clientSet.ResourceV1beta1().ResourceClaims(namespace).UpdateStatus(ctx, createdClaim, metav1.UpdateOptions{})
	framework.ExpectNoError(err)

	return pod
}

func createTestResourceSlice(ctx context.Context, clientSet kubernetes.Interface, nodeName, driverName string) {
	slice := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: nodeName,
		},
		Spec: resourceapi.ResourceSliceSpec{
			NodeName: nodeName,
			Driver:   driverName,
			Pool: resourceapi.ResourcePool{
				Name:               nodeName,
				ResourceSliceCount: 1,
			},
		},
	}

	ginkgo.By(fmt.Sprintf("Creating ResourceSlice %s", nodeName))
	slice, err := clientSet.ResourceV1beta1().ResourceSlices().Create(ctx, slice, metav1.CreateOptions{})
	framework.ExpectNoError(err, "create ResourceSlice")
	ginkgo.DeferCleanup(func(ctx context.Context) {
		ginkgo.By(fmt.Sprintf("Deleting ResourceSlice %s", nodeName))
		err := clientSet.ResourceV1beta1().ResourceSlices().Delete(ctx, slice.Name, metav1.DeleteOptions{})
		if !apierrors.IsNotFound(err) {
			framework.ExpectNoError(err, "delete ResourceSlice")
		}
	})
}

func listResources(client kubernetes.Interface) func(ctx context.Context) ([]resourceapi.ResourceSlice, error) {
	return func(ctx context.Context) ([]resourceapi.ResourceSlice, error) {
		slices, err := client.ResourceV1beta1().ResourceSlices().List(ctx, metav1.ListOptions{})
		if err != nil {
			return nil, err
		}
		return slices.Items, nil
	}
}

func listAndStoreResources(client kubernetes.Interface, lastSlices *[]resourceapi.ResourceSlice) func(ctx context.Context) ([]resourceapi.ResourceSlice, error) {
	return func(ctx context.Context) ([]resourceapi.ResourceSlice, error) {
		slices, err := client.ResourceV1beta1().ResourceSlices().List(ctx, metav1.ListOptions{})
		if err != nil {
			return nil, err
		}
		*lastSlices = slices.Items
		return *lastSlices, nil
	}
}

func matchResourcesByNodeName(nodeName string) types.GomegaMatcher {
	return gomega.HaveField("Spec.NodeName", gomega.Equal(nodeName))
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
