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

package node

import (
	"context"
	"fmt"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("Container Lifecycle Hook", func() {
	f := framework.NewDefaultFramework("container-lifecycle-hook")
	// FIXME: This test is being run in the privileged mode because of https://github.com/kubernetes/kubernetes/issues/133091
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var podClient *e2epod.PodClient
	const (
		podCheckInterval     = 1 * time.Second
		postStartWaitTimeout = 2 * time.Minute
		preStopWaitTimeout   = 30 * time.Second
	)
	ginkgo.Context("when create a pod with lifecycle hook", func() {
		var (
			targetIP, targetURL, targetNode string

			httpPorts = []v1.ContainerPort{
				{
					ContainerPort: 8080,
					Protocol:      v1.ProtocolTCP,
				},
			}
			httpsPorts = []v1.ContainerPort{
				{
					ContainerPort: 9090,
					Protocol:      v1.ProtocolTCP,
				},
			}
			httpsArgs = []string{
				"netexec",
				"--http-port", "9090",
				"--udp-port", "9091",
				"--tls-cert-file", "/localhost.crt",
				"--tls-private-key-file", "/localhost.key",
			}
		)

		podHandleHookRequest := e2epod.NewAgnhostPodFromContainers(
			"", "pod-handle-http-request", nil,
			e2epod.NewAgnhostContainer("container-handle-http-request", nil, httpPorts, "netexec"),
			e2epod.NewAgnhostContainer("container-handle-https-request", nil, httpsPorts, httpsArgs...),
		)

		ginkgo.BeforeEach(func(ctx context.Context) {
			node, err := e2enode.GetRandomReadySchedulableNode(ctx, f.ClientSet)
			framework.ExpectNoError(err)
			targetNode = node.Name
			nodeSelection := e2epod.NodeSelection{}
			e2epod.SetAffinity(&nodeSelection, targetNode)
			e2epod.SetNodeSelection(&podHandleHookRequest.Spec, nodeSelection)

			podClient = e2epod.NewPodClient(f)
			ginkgo.By("create the container to handle the HTTPGet hook request.")
			newPod := podClient.CreateSync(ctx, podHandleHookRequest)
			targetIP = newPod.Status.PodIP
			targetURL = targetIP
			if strings.Contains(targetIP, ":") {
				targetURL = fmt.Sprintf("[%s]", targetIP)
			}
		})
		testPodWithHook := func(ctx context.Context, podWithHook *v1.Pod) {
			ginkgo.By("create the pod with lifecycle hook")
			podClient.CreateSync(ctx, podWithHook)
			const (
				defaultHandler = iota
				httpsHandler
			)
			handlerContainer := defaultHandler
			if podWithHook.Spec.Containers[0].Lifecycle.PostStart != nil {
				ginkgo.By("check poststart hook")
				if podWithHook.Spec.Containers[0].Lifecycle.PostStart.HTTPGet != nil {
					if v1.URISchemeHTTPS == podWithHook.Spec.Containers[0].Lifecycle.PostStart.HTTPGet.Scheme {
						handlerContainer = httpsHandler
					}
				}
				gomega.Eventually(ctx, func(ctx context.Context) error {
					return podClient.MatchContainerOutput(ctx, podHandleHookRequest.Name, podHandleHookRequest.Spec.Containers[handlerContainer].Name,
						`GET /echo\?msg=poststart`)
				}, postStartWaitTimeout, podCheckInterval).Should(gomega.BeNil())
			}
			ginkgo.By("delete the pod with lifecycle hook")
			podClient.DeleteSync(ctx, podWithHook.Name, *metav1.NewDeleteOptions(15), f.Timeouts.PodDelete)
			if podWithHook.Spec.Containers[0].Lifecycle.PreStop != nil {
				ginkgo.By("check prestop hook")
				if podWithHook.Spec.Containers[0].Lifecycle.PreStop.HTTPGet != nil {
					if v1.URISchemeHTTPS == podWithHook.Spec.Containers[0].Lifecycle.PreStop.HTTPGet.Scheme {
						handlerContainer = httpsHandler
					}
				}
				gomega.Eventually(ctx, func(ctx context.Context) error {
					return podClient.MatchContainerOutput(ctx, podHandleHookRequest.Name, podHandleHookRequest.Spec.Containers[handlerContainer].Name,
						`GET /echo\?msg=prestop`)
				}, preStopWaitTimeout, podCheckInterval).Should(gomega.BeNil())
			}
		}
		/*
			Release: v1.9
			Testname: Pod Lifecycle, post start exec hook
			Description: When a post start handler is specified in the container lifecycle using a 'Exec' action, then the handler MUST be invoked after the start of the container. A server pod is created that will serve http requests, create a second pod with a container lifecycle specifying a post start that invokes the server pod using ExecAction to validate that the post start is executed.
		*/
		framework.ConformanceIt("should execute poststart exec hook properly", f.WithNodeConformance(), func(ctx context.Context) {
			lifecycle := &v1.Lifecycle{
				PostStart: &v1.LifecycleHandler{
					Exec: &v1.ExecAction{
						Command: []string{"sh", "-c", "curl http://" + targetURL + ":8080/echo?msg=poststart"},
					},
				},
			}
			podWithHook := getPodWithHook("pod-with-poststart-exec-hook", imageutils.GetE2EImage(imageutils.Agnhost), lifecycle)

			testPodWithHook(ctx, podWithHook)
		})
		/*
			Release: v1.9
			Testname: Pod Lifecycle, prestop exec hook
			Description: When a pre-stop handler is specified in the container lifecycle using a 'Exec' action, then the handler MUST be invoked before the container is terminated. A server pod is created that will serve http requests, create a second pod with a container lifecycle specifying a pre-stop that invokes the server pod using ExecAction to validate that the pre-stop is executed.
		*/
		framework.ConformanceIt("should execute prestop exec hook properly", f.WithNodeConformance(), func(ctx context.Context) {
			lifecycle := &v1.Lifecycle{
				PreStop: &v1.LifecycleHandler{
					Exec: &v1.ExecAction{
						Command: []string{"sh", "-c", "curl http://" + targetURL + ":8080/echo?msg=prestop"},
					},
				},
			}
			podWithHook := getPodWithHook("pod-with-prestop-exec-hook", imageutils.GetE2EImage(imageutils.Agnhost), lifecycle)
			testPodWithHook(ctx, podWithHook)
		})
		/*
			Release: v1.9
			Testname: Pod Lifecycle, post start http hook
			Description: When a post start handler is specified in the container lifecycle using a HttpGet action, then the handler MUST be invoked after the start of the container. A server pod is created that will serve http requests, create a second pod on the same node with a container lifecycle specifying a post start that invokes the server pod to validate that the post start is executed.
		*/
		framework.ConformanceIt("should execute poststart http hook properly", f.WithNodeConformance(), func(ctx context.Context) {
			lifecycle := &v1.Lifecycle{
				PostStart: &v1.LifecycleHandler{
					HTTPGet: &v1.HTTPGetAction{
						Path: "/echo?msg=poststart",
						Host: targetIP,
						Port: intstr.FromInt32(8080),
					},
				},
			}
			podWithHook := getPodWithHook("pod-with-poststart-http-hook", imageutils.GetPauseImageName(), lifecycle)
			// make sure we spawn the test pod on the same node as the webserver.
			nodeSelection := e2epod.NodeSelection{}
			e2epod.SetAffinity(&nodeSelection, targetNode)
			e2epod.SetNodeSelection(&podWithHook.Spec, nodeSelection)
			testPodWithHook(ctx, podWithHook)
		})
		/*
			Release : v1.23
			Testname: Pod Lifecycle, poststart https hook
			Description: When a post-start handler is specified in the container lifecycle using a 'HttpGet' action, then the handler MUST be invoked before the container is terminated. A server pod is created that will serve https requests, create a second pod on the same node with a container lifecycle specifying a post-start that invokes the server pod to validate that the post-start is executed.
		*/
		f.It("should execute poststart https hook properly [MinimumKubeletVersion:1.23]", f.WithNodeConformance(), func(ctx context.Context) {
			lifecycle := &v1.Lifecycle{
				PostStart: &v1.LifecycleHandler{
					HTTPGet: &v1.HTTPGetAction{
						Scheme: v1.URISchemeHTTPS,
						Path:   "/echo?msg=poststart",
						Host:   targetIP,
						Port:   intstr.FromInt32(9090),
					},
				},
			}
			podWithHook := getPodWithHook("pod-with-poststart-https-hook", imageutils.GetPauseImageName(), lifecycle)
			// make sure we spawn the test pod on the same node as the webserver.
			nodeSelection := e2epod.NodeSelection{}
			e2epod.SetAffinity(&nodeSelection, targetNode)
			e2epod.SetNodeSelection(&podWithHook.Spec, nodeSelection)
			testPodWithHook(ctx, podWithHook)
		})
		/*
			Release : v1.9
			Testname: Pod Lifecycle, prestop http hook
			Description: When a pre-stop handler is specified in the container lifecycle using a 'HttpGet' action, then the handler MUST be invoked before the container is terminated. A server pod is created that will serve http requests, create a second pod on the same node with a container lifecycle specifying a pre-stop that invokes the server pod to validate that the pre-stop is executed.
		*/
		framework.ConformanceIt("should execute prestop http hook properly", f.WithNodeConformance(), func(ctx context.Context) {
			lifecycle := &v1.Lifecycle{
				PreStop: &v1.LifecycleHandler{
					HTTPGet: &v1.HTTPGetAction{
						Path: "/echo?msg=prestop",
						Host: targetIP,
						Port: intstr.FromInt32(8080),
					},
				},
			}
			podWithHook := getPodWithHook("pod-with-prestop-http-hook", imageutils.GetPauseImageName(), lifecycle)
			// make sure we spawn the test pod on the same node as the webserver.
			nodeSelection := e2epod.NodeSelection{}
			e2epod.SetAffinity(&nodeSelection, targetNode)
			e2epod.SetNodeSelection(&podWithHook.Spec, nodeSelection)
			testPodWithHook(ctx, podWithHook)
		})
		/*
			Release : v1.23
			Testname: Pod Lifecycle, prestop https hook
			Description: When a pre-stop handler is specified in the container lifecycle using a 'HttpGet' action, then the handler MUST be invoked before the container is terminated. A server pod is created that will serve https requests, create a second pod on the same node with a container lifecycle specifying a pre-stop that invokes the server pod to validate that the pre-stop is executed.
		*/
		f.It("should execute prestop https hook properly [MinimumKubeletVersion:1.23]", f.WithNodeConformance(), func(ctx context.Context) {
			lifecycle := &v1.Lifecycle{
				PreStop: &v1.LifecycleHandler{
					HTTPGet: &v1.HTTPGetAction{
						Scheme: v1.URISchemeHTTPS,
						Path:   "/echo?msg=prestop",
						Host:   targetIP,
						Port:   intstr.FromInt32(9090),
					},
				},
			}
			podWithHook := getPodWithHook("pod-with-prestop-https-hook", imageutils.GetPauseImageName(), lifecycle)
			// make sure we spawn the test pod on the same node as the webserver.
			nodeSelection := e2epod.NodeSelection{}
			e2epod.SetAffinity(&nodeSelection, targetNode)
			e2epod.SetNodeSelection(&podWithHook.Spec, nodeSelection)
			testPodWithHook(ctx, podWithHook)
		})
	})
})

var _ = SIGDescribe(feature.SidecarContainers, framework.WithFeatureGate(features.SidecarContainers), "Restartable Init Container Lifecycle Hook", func() {
	f := framework.NewDefaultFramework("restartable-init-container-lifecycle-hook")
	// FIXME: This test is being run in the privileged mode because of https://github.com/kubernetes/kubernetes/issues/133091
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var podClient *e2epod.PodClient
	const (
		podCheckInterval     = 1 * time.Second
		postStartWaitTimeout = 2 * time.Minute
		preStopWaitTimeout   = 30 * time.Second
	)
	ginkgo.Context("when create a pod with lifecycle hook", func() {
		var (
			targetIP, targetURL, targetNode string

			httpPorts = []v1.ContainerPort{
				{
					ContainerPort: 8080,
					Protocol:      v1.ProtocolTCP,
				},
			}
			httpsPorts = []v1.ContainerPort{
				{
					ContainerPort: 9090,
					Protocol:      v1.ProtocolTCP,
				},
			}
			httpsArgs = []string{
				"netexec",
				"--http-port", "9090",
				"--udp-port", "9091",
				"--tls-cert-file", "/localhost.crt",
				"--tls-private-key-file", "/localhost.key",
			}
		)

		podHandleHookRequest := e2epod.NewAgnhostPodFromContainers(
			"", "pod-handle-http-request", nil,
			e2epod.NewAgnhostContainer("container-handle-http-request", nil, httpPorts, "netexec"),
			e2epod.NewAgnhostContainer("container-handle-https-request", nil, httpsPorts, httpsArgs...),
		)

		ginkgo.BeforeEach(func(ctx context.Context) {
			node, err := e2enode.GetRandomReadySchedulableNode(ctx, f.ClientSet)
			framework.ExpectNoError(err)
			targetNode = node.Name
			nodeSelection := e2epod.NodeSelection{}
			e2epod.SetAffinity(&nodeSelection, targetNode)
			e2epod.SetNodeSelection(&podHandleHookRequest.Spec, nodeSelection)

			podClient = e2epod.NewPodClient(f)
			ginkgo.By("create the container to handle the HTTPGet hook request.")
			newPod := podClient.CreateSync(ctx, podHandleHookRequest)
			targetIP = newPod.Status.PodIP
			targetURL = targetIP
			if strings.Contains(targetIP, ":") {
				targetURL = fmt.Sprintf("[%s]", targetIP)
			}
		})
		testPodWithHook := func(ctx context.Context, podWithHook *v1.Pod) {
			ginkgo.By("create the pod with lifecycle hook")
			podClient.CreateSync(ctx, podWithHook)
			const (
				defaultHandler = iota
				httpsHandler
			)
			handlerContainer := defaultHandler
			if podWithHook.Spec.InitContainers[0].Lifecycle.PostStart != nil {
				ginkgo.By("check poststart hook")
				if podWithHook.Spec.InitContainers[0].Lifecycle.PostStart.HTTPGet != nil {
					if v1.URISchemeHTTPS == podWithHook.Spec.InitContainers[0].Lifecycle.PostStart.HTTPGet.Scheme {
						handlerContainer = httpsHandler
					}
				}
				gomega.Eventually(ctx, func(ctx context.Context) error {
					return podClient.MatchContainerOutput(ctx, podHandleHookRequest.Name, podHandleHookRequest.Spec.Containers[handlerContainer].Name,
						`GET /echo\?msg=poststart`)
				}, postStartWaitTimeout, podCheckInterval).Should(gomega.BeNil())
			}
			ginkgo.By("delete the pod with lifecycle hook")
			podClient.DeleteSync(ctx, podWithHook.Name, *metav1.NewDeleteOptions(15), f.Timeouts.PodDelete)
			if podWithHook.Spec.InitContainers[0].Lifecycle.PreStop != nil {
				ginkgo.By("check prestop hook")
				if podWithHook.Spec.InitContainers[0].Lifecycle.PreStop.HTTPGet != nil {
					if v1.URISchemeHTTPS == podWithHook.Spec.InitContainers[0].Lifecycle.PreStop.HTTPGet.Scheme {
						handlerContainer = httpsHandler
					}
				}
				gomega.Eventually(ctx, func(ctx context.Context) error {
					return podClient.MatchContainerOutput(ctx, podHandleHookRequest.Name, podHandleHookRequest.Spec.Containers[handlerContainer].Name,
						`GET /echo\?msg=prestop`)
				}, preStopWaitTimeout, podCheckInterval).Should(gomega.BeNil())
			}
		}
		/*
			Release: v1.28
			Testname: Pod Lifecycle with restartable init container, post start exec hook
			Description: When a post start handler is specified in the container
			lifecycle using a 'Exec' action, then the handler MUST be invoked after
			the start of the container. A server pod is created that will serve http
			requests, create a second pod with a container lifecycle specifying a
			post start that invokes the server pod using ExecAction to validate that
			the post start is executed.
		*/
		ginkgo.It("should execute poststart exec hook properly", func(ctx context.Context) {
			lifecycle := &v1.Lifecycle{
				PostStart: &v1.LifecycleHandler{
					Exec: &v1.ExecAction{
						Command: []string{"sh", "-c", "curl http://" + targetURL + ":8080/echo?msg=poststart"},
					},
				},
			}
			podWithHook := getSidecarPodWithHook("pod-with-poststart-exec-hook", imageutils.GetE2EImage(imageutils.Agnhost), lifecycle)

			testPodWithHook(ctx, podWithHook)
		})
		/*
			Release: v1.28
			Testname: Pod Lifecycle with restartable init container, prestop exec hook
			Description: When a pre-stop handler is specified in the container
			lifecycle using a 'Exec' action, then the handler MUST be invoked before
			the container is terminated. A server pod is created that will serve http
			requests, create a second pod with a container lifecycle specifying a
			pre-stop that invokes the server pod using ExecAction to validate that
			the pre-stop is executed.
		*/
		ginkgo.It("should execute prestop exec hook properly", func(ctx context.Context) {
			lifecycle := &v1.Lifecycle{
				PreStop: &v1.LifecycleHandler{
					Exec: &v1.ExecAction{
						Command: []string{"sh", "-c", "curl http://" + targetURL + ":8080/echo?msg=prestop"},
					},
				},
			}
			podWithHook := getSidecarPodWithHook("pod-with-prestop-exec-hook", imageutils.GetE2EImage(imageutils.Agnhost), lifecycle)
			testPodWithHook(ctx, podWithHook)
		})
		/*
			Release: v1.28
			Testname: Pod Lifecycle with restartable init container, post start http hook
			Description: When a post start handler is specified in the container
			lifecycle using a HttpGet action, then the handler MUST be invoked after
			the start of the container. A server pod is created that will serve http
			requests, create a second pod on the same node with a container lifecycle
			specifying a post start that invokes the server pod to validate that the
			post start is executed.
		*/
		ginkgo.It("should execute poststart http hook properly", func(ctx context.Context) {
			lifecycle := &v1.Lifecycle{
				PostStart: &v1.LifecycleHandler{
					HTTPGet: &v1.HTTPGetAction{
						Path: "/echo?msg=poststart",
						Host: targetIP,
						Port: intstr.FromInt32(8080),
					},
				},
			}
			podWithHook := getSidecarPodWithHook("pod-with-poststart-http-hook", imageutils.GetPauseImageName(), lifecycle)
			// make sure we spawn the test pod on the same node as the webserver.
			nodeSelection := e2epod.NodeSelection{}
			e2epod.SetAffinity(&nodeSelection, targetNode)
			e2epod.SetNodeSelection(&podWithHook.Spec, nodeSelection)
			testPodWithHook(ctx, podWithHook)
		})
		/*
			Release : v1.28
			Testname: Pod Lifecycle with restartable init container, poststart https hook
			Description: When a post-start handler is specified in the container
			lifecycle using a 'HttpGet' action, then the handler MUST be invoked
			before the container is terminated. A server pod is created that will
			serve https requests, create a second pod on the same node with a
			container lifecycle specifying a post-start that invokes the server pod
			to validate that the post-start is executed.
		*/
		ginkgo.It("should execute poststart https hook properly [MinimumKubeletVersion:1.23]", func(ctx context.Context) {
			lifecycle := &v1.Lifecycle{
				PostStart: &v1.LifecycleHandler{
					HTTPGet: &v1.HTTPGetAction{
						Scheme: v1.URISchemeHTTPS,
						Path:   "/echo?msg=poststart",
						Host:   targetIP,
						Port:   intstr.FromInt32(9090),
					},
				},
			}
			podWithHook := getSidecarPodWithHook("pod-with-poststart-https-hook", imageutils.GetPauseImageName(), lifecycle)
			// make sure we spawn the test pod on the same node as the webserver.
			nodeSelection := e2epod.NodeSelection{}
			e2epod.SetAffinity(&nodeSelection, targetNode)
			e2epod.SetNodeSelection(&podWithHook.Spec, nodeSelection)
			testPodWithHook(ctx, podWithHook)
		})
		/*
			Release : v1.28
			Testname: Pod Lifecycle with restartable init container, prestop http hook
			Description: When a pre-stop handler is specified in the container
			lifecycle using a 'HttpGet' action, then the handler MUST be invoked
			before the container is terminated. A server pod is created that will
			serve http requests, create a second pod on the same node with a
			container lifecycle specifying a pre-stop that invokes the server pod to
			validate that the pre-stop is executed.
		*/
		ginkgo.It("should execute prestop http hook properly", func(ctx context.Context) {
			lifecycle := &v1.Lifecycle{
				PreStop: &v1.LifecycleHandler{
					HTTPGet: &v1.HTTPGetAction{
						Path: "/echo?msg=prestop",
						Host: targetIP,
						Port: intstr.FromInt32(8080),
					},
				},
			}
			podWithHook := getSidecarPodWithHook("pod-with-prestop-http-hook", imageutils.GetPauseImageName(), lifecycle)
			// make sure we spawn the test pod on the same node as the webserver.
			nodeSelection := e2epod.NodeSelection{}
			e2epod.SetAffinity(&nodeSelection, targetNode)
			e2epod.SetNodeSelection(&podWithHook.Spec, nodeSelection)
			testPodWithHook(ctx, podWithHook)
		})
		/*
			Release : v1.28
			Testname: Pod Lifecycle with restartable init container, prestop https hook
			Description: When a pre-stop handler is specified in the container
			lifecycle using a 'HttpGet' action, then the handler MUST be invoked
			before the container is terminated. A server pod is created that will
			serve https requests, create a second pod on the same node with a
			container lifecycle specifying a pre-stop that invokes the server pod to
			validate that the pre-stop is executed.
		*/
		ginkgo.It("should execute prestop https hook properly [MinimumKubeletVersion:1.23]", func(ctx context.Context) {
			lifecycle := &v1.Lifecycle{
				PreStop: &v1.LifecycleHandler{
					HTTPGet: &v1.HTTPGetAction{
						Scheme: v1.URISchemeHTTPS,
						Path:   "/echo?msg=prestop",
						Host:   targetIP,
						Port:   intstr.FromInt32(9090),
					},
				},
			}
			podWithHook := getSidecarPodWithHook("pod-with-prestop-https-hook", imageutils.GetPauseImageName(), lifecycle)
			// make sure we spawn the test pod on the same node as the webserver.
			nodeSelection := e2epod.NodeSelection{}
			e2epod.SetAffinity(&nodeSelection, targetNode)
			e2epod.SetNodeSelection(&podWithHook.Spec, nodeSelection)
			testPodWithHook(ctx, podWithHook)
		})
	})
})

func getPodWithHook(name string, image string, lifecycle *v1.Lifecycle) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:      name,
					Image:     image,
					Lifecycle: lifecycle,
				},
			},
		},
	}
}

func getSidecarPodWithHook(name string, image string, lifecycle *v1.Lifecycle) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{
				{
					Name:      name,
					Image:     image,
					Lifecycle: lifecycle,
					RestartPolicy: func() *v1.ContainerRestartPolicy {
						restartPolicy := v1.ContainerRestartPolicyAlways
						return &restartPolicy
					}(),
				},
			},
			Containers: []v1.Container{
				{
					Name:  "main",
					Image: imageutils.GetPauseImageName(),
				},
			},
		},
	}
}

func validDuration(duration time.Duration, low, high int64) bool {
	return duration >= time.Second*time.Duration(low) && duration <= time.Second*time.Duration(high)
}

var _ = SIGDescribe("Lifecycle Sleep Hook", func() {
	f := framework.NewDefaultFramework("pod-lifecycle-sleep-action")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
	var podClient *e2epod.PodClient

	ginkgo.Context("when create a pod with lifecycle hook using sleep action", func() {
		ginkgo.BeforeEach(func(ctx context.Context) {
			podClient = e2epod.NewPodClient(f)
		})

		var finalizer = "test/finalizer"
		/*
			Release : v1.34
			Testname: Pod Lifecycle, prestop sleep hook
			Description: When a pre-stop handler is specified in the container lifecycle using a 'Sleep' action, then the handler MUST be invoked before the container is terminated. A test pod will be created to verify if its termination time aligns with the sleep time specified when it is terminated.
		*/
		ginkgo.It("valid prestop hook using sleep action", func(ctx context.Context) {
			const sleepSeconds = 50
			const gracePeriod = 100
			lifecycle := &v1.Lifecycle{
				PreStop: &v1.LifecycleHandler{
					Sleep: &v1.SleepAction{Seconds: sleepSeconds},
				},
			}
			name := "pod-with-prestop-sleep-hook"
			podWithHook := getPodWithHook(name, imageutils.GetPauseImageName(), lifecycle)
			podWithHook.Finalizers = append(podWithHook.Finalizers, finalizer)
			podWithHook.Spec.TerminationGracePeriodSeconds = ptr.To[int64](gracePeriod)
			ginkgo.By("create the pod with lifecycle hook using sleep action")
			p := podClient.CreateSync(ctx, podWithHook)
			defer podClient.RemoveFinalizer(ctx, name, finalizer)
			ginkgo.By("delete the pod with lifecycle hook using sleep action")
			_ = podClient.Delete(ctx, podWithHook.Name, metav1.DeleteOptions{})
			p, err := podClient.Get(ctx, p.Name, metav1.GetOptions{})
			if err != nil {
				framework.Failf("failed getting pod after deletion")
			}
			// deletionTimestamp equals to delete_time + tgps
			// TODO: reduce sleep_seconds and tgps after issues.k8s.io/132205 is solved
			// we get deletionTimestamp before container become terminated here because of issues.k8s.io/132205
			deletionTS := p.DeletionTimestamp.Time
			if err := e2epod.WaitForContainerTerminated(ctx, f.ClientSet, p.Namespace, p.Name, name, sleepSeconds*2*time.Second); err != nil {
				framework.Failf("failed waiting for container terminated")
			}

			p, err = podClient.Get(ctx, p.Name, metav1.GetOptions{})
			if err != nil {
				framework.Failf("failed getting pod after deletion")
			}
			// finishAt equals to delete_time + sleep_duration
			finishAt := p.Status.ContainerStatuses[0].State.Terminated.FinishedAt

			// sleep_duration = (delete_time + sleep_duration) - (delete_time + tgps) + tgps
			sleepDuration := finishAt.Sub(deletionTS) + time.Second*gracePeriod

			// sleep_duration should be
			// longer than 50 seconds (pod should sleep for 50 seconds)
			// shorter than gracePeriodSeconds (100 seconds here)
			if !validDuration(sleepDuration, sleepSeconds, gracePeriod) {
				framework.Failf("unexpected delay duration before killing the pod, finishAt = %v, deletionAt= %v", finishAt, deletionTS)
			}
		})

		/*
			Release : v1.34
			Testname: Pod Lifecycle, prestop sleep hook with low gracePeriodSeconds
			Description: When a pre-stop handler is specified in the container lifecycle using a 'Sleep' action, then the handler MUST be invoked before the container is terminated. A test pod will be created, and its `gracePeriodSeconds` will be modified to a value less than the sleep time before termination. The termination time will then be checked to ensure it aligns with the `gracePeriodSeconds` value.
		*/
		ginkgo.It("reduce GracePeriodSeconds during runtime", func(ctx context.Context) {
			const sleepSeconds = 50
			lifecycle := &v1.Lifecycle{
				PreStop: &v1.LifecycleHandler{
					Sleep: &v1.SleepAction{Seconds: sleepSeconds},
				},
			}
			name := "pod-with-prestop-sleep-hook"
			podWithHook := getPodWithHook(name, imageutils.GetPauseImageName(), lifecycle)
			podWithHook.Finalizers = append(podWithHook.Finalizers, finalizer)
			podWithHook.Spec.TerminationGracePeriodSeconds = ptr.To[int64](100)
			ginkgo.By("create the pod with lifecycle hook using sleep action")
			p := podClient.CreateSync(ctx, podWithHook)
			defer podClient.RemoveFinalizer(ctx, name, finalizer)
			ginkgo.By("delete the pod with lifecycle hook using sleep action")

			const gracePeriod = 30
			_ = podClient.Delete(ctx, podWithHook.Name, *metav1.NewDeleteOptions(gracePeriod))
			p, err := podClient.Get(ctx, p.Name, metav1.GetOptions{})
			if err != nil {
				framework.Failf("failed getting pod after deletion")
			}
			// deletionTimestamp equals to delete_time + tgps
			// TODO: reduce sleep_seconds and tgps after issues.k8s.io/132205 is solved
			// we get deletionTimestamp before container become terminated here because of issues.k8s.io/132205
			deletionTS := p.DeletionTimestamp.Time
			if err := e2epod.WaitForContainerTerminated(ctx, f.ClientSet, p.Namespace, p.Name, name, sleepSeconds*2*time.Second); err != nil {
				framework.Failf("failed waiting for container terminated")
			}
			p, err = podClient.Get(ctx, p.Name, metav1.GetOptions{})
			if err != nil {
				framework.Failf("failed getting pod after deletion")
			}
			// finishAt equals to delete_time + sleep_duration
			finishAt := p.Status.ContainerStatuses[0].State.Terminated.FinishedAt

			// sleep_duration = (delete_time + sleep_duration) - (delete_time + tgps) + tgps
			sleepDuration := finishAt.Sub(deletionTS) + time.Second*gracePeriod
			// sleep_duration should be
			// longer than 30 seconds (we change gracePeriodSeconds to 30 seconds here, and it's less than sleep action)
			// shorter than sleep action (to make sure it doesn't take effect)
			if !validDuration(sleepDuration, gracePeriod, sleepSeconds) {
				framework.Failf("unexpected delay duration before killing the pod, finishAt = %v, deletionAt= %v", finishAt, deletionTS)
			}
		})

		/*
			Release : v1.34
			Testname: Pod Lifecycle, prestop sleep hook with erroneous startup command
			Description: When a pre-stop handler is specified in the container lifecycle using a 'Sleep' action, then the handler MUST be invoked before the container is terminated. A test pod with an erroneous startup command will be created, and upon termination, it will be checked whether it ignored the sleep time.
		*/
		ginkgo.It("ignore terminated container", func(ctx context.Context) {
			const sleepSeconds = 10
			const gracePeriod = 30
			lifecycle := &v1.Lifecycle{
				PreStop: &v1.LifecycleHandler{
					Sleep: &v1.SleepAction{Seconds: sleepSeconds},
				},
			}
			name := "pod-with-prestop-sleep-hook"
			podWithHook := getPodWithHook(name, imageutils.GetE2EImage(imageutils.BusyBox), lifecycle)
			podWithHook.Spec.TerminationGracePeriodSeconds = ptr.To[int64](gracePeriod)
			podWithHook.Spec.Containers[0].Command = []string{"/bin/sh"}
			// If we exit the container as soon as it's created,
			// finishAt - startedAt can be negative due to some internal race
			// so we need to keep it running for a while
			podWithHook.Spec.Containers[0].Args = []string{"-c", "sleep 3"}
			podWithHook.Spec.RestartPolicy = v1.RestartPolicyNever
			ginkgo.By("create the pod with lifecycle hook using sleep action")
			p := podClient.Create(ctx, podWithHook)
			defer podClient.DeleteSync(ctx, podWithHook.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
			framework.ExpectNoError(e2epod.WaitForContainerTerminated(ctx, f.ClientSet, f.Namespace.Name, p.Name, name, gracePeriod*time.Second))

			p, err := podClient.Get(ctx, p.Name, metav1.GetOptions{})
			if err != nil {
				framework.Failf("failed getting pod after deletion")
			}
			finishAt := p.Status.ContainerStatuses[0].State.Terminated.FinishedAt
			startedAt := p.Status.ContainerStatuses[0].State.Terminated.StartedAt
			cost := finishAt.Sub(startedAt.Time)
			// cost should be
			// shorter than sleep action (container is terminated and sleep action should be ignored)
			if !validDuration(cost, 0, sleepSeconds) {
				framework.Failf("unexpected delay duration before killing the pod, cost = %v", cost)
			}
		})
	})
})

var _ = SIGDescribe("Lifecycle sleep action zero value", func() {
	f := framework.NewDefaultFramework("pod-lifecycle-sleep-action-allow-zero")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
	var podClient *e2epod.PodClient

	ginkgo.Context("when create a pod with lifecycle hook using sleep action with a duration of zero seconds", func() {
		ginkgo.BeforeEach(func(ctx context.Context) {
			podClient = e2epod.NewPodClient(f)
		})
		ginkgo.It("prestop hook using sleep action with zero duration", func(ctx context.Context) {
			lifecycle := &v1.Lifecycle{
				PreStop: &v1.LifecycleHandler{
					Sleep: &v1.SleepAction{Seconds: 0},
				},
			}
			podWithHook := getPodWithHook("pod-with-prestop-sleep-hook-zero-duration", imageutils.GetPauseImageName(), lifecycle)
			ginkgo.By("create the pod with lifecycle hook using sleep action with zero duration")
			podClient.CreateSync(ctx, podWithHook)
			ginkgo.By("delete the pod with lifecycle hook using sleep action with zero duration")
			start := time.Now()
			podClient.DeleteSync(ctx, podWithHook.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
			cost := time.Since(start)
			// cost should be
			// longer than 0 seconds (pod shouldn't sleep and the handler should return immediately)
			// shorter than gracePeriodSeconds (default 30 seconds here)
			if !validDuration(cost, 0, 30) {
				framework.Failf("unexpected delay duration before killing the pod, cost = %v", cost)
			}
		})

	})
})

var _ = SIGDescribe(feature.ContainerStopSignals, framework.WithFeatureGate(features.ContainerStopSignals), func() {
	f := framework.NewDefaultFramework("container-stop-signals")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
	var podClient *e2epod.PodClient
	sigterm := v1.SIGTERM
	podName := "pod-" + utilrand.String(5)

	ginkgo.Context("when create a pod with a StopSignal lifecycle", func() {
		ginkgo.BeforeEach(func(ctx context.Context) {
			podClient = e2epod.NewPodClient(f)
		})
		ginkgo.It("StopSignal defined with pod.OS", func(ctx context.Context) {

			testPod := e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					OS: &v1.PodOS{
						Name: v1.Linux,
					},
					Containers: []v1.Container{
						{
							Name:  "test",
							Image: imageutils.GetPauseImageName(),
							Lifecycle: &v1.Lifecycle{
								StopSignal: &sigterm,
							},
						},
					},
				},
			})

			ginkgo.By("submitting the pod to kubernetes")
			pod := podClient.CreateSync(ctx, testPod)
			framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod), "Pod didn't start when custom StopSignal was passed in Lifecycle")
		})
	})
})
