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

package network

import (
	"context"
	"fmt"
	"net"
	"strconv"
	"time"

	"github.com/onsi/ginkgo/v2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/network/common"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = common.SIGDescribe("Connectivity Pod Lifecycle", func() {

	fr := framework.NewDefaultFramework("podlifecycle")
	fr.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	var (
		cs        clientset.Interface
		ns        string
		podClient *e2epod.PodClient
	)
	ginkgo.BeforeEach(func(ctx context.Context) {
		cs = fr.ClientSet
		ns = fr.Namespace.Name
		podClient = e2epod.NewPodClient(fr)
	})

	ginkgo.It("should be able to connect from a Pod to a terminating Pod", func(ctx context.Context) {
		ginkgo.By("Creating 1 webserver pod able to serve traffic during the grace period of 300 seconds")
		gracePeriod := int64(100)
		webserverPod := e2epod.NewAgnhostPod(ns, "webserver-pod", nil, nil, nil, "netexec", "--http-port=80", fmt.Sprintf("--delay-shutdown=%d", gracePeriod))
		webserverPod.Spec.TerminationGracePeriodSeconds = &gracePeriod
		webserverPod = podClient.CreateSync(ctx, webserverPod)

		ginkgo.By("Creating 1 client pod that will try to connect to the webserver")
		pausePod := e2epod.NewAgnhostPod(ns, "pause-pod-1", nil, nil, nil)
		pausePod = podClient.CreateSync(ctx, pausePod)

		ginkgo.By("Try to connect to the webserver")
		// Wait until we are able to connect to the Pod
		podIPAddress := net.JoinHostPort(webserverPod.Status.PodIP, strconv.Itoa(80))
		execHostnameTest(*pausePod, podIPAddress, webserverPod.Name)

		// webserver should continue to serve traffic after delete
		// since will be gracefully terminating
		err := cs.CoreV1().Pods(ns).Delete(ctx, webserverPod.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "error deleting webserver pod")
		// wait some time to ensure the test hit the pod during the terminating phase
		time.Sleep(15 * time.Second)
		execHostnameTest(*pausePod, podIPAddress, webserverPod.Name)
	})

	ginkgo.It("should be able to connect to other Pod from a terminating Pod", func(ctx context.Context) {
		ginkgo.By("Creating 1 webserver pod able to serve traffic during the grace period of 300 seconds")
		gracePeriod := int64(100)
		webserverPod := e2epod.NewAgnhostPod(ns, "webserver-pod", nil, nil, nil, "netexec", "--http-port=80", fmt.Sprintf("--delay-shutdown=%d", gracePeriod))
		webserverPod = podClient.CreateSync(ctx, webserverPod)

		ginkgo.By("Creating 1 client pod that will try to connect to the webservers")
		pausePod := e2epod.NewAgnhostPod(ns, "pause-pod-1", nil, nil, nil, "netexec", "--http-port=80", fmt.Sprintf("--delay-shutdown=%d", gracePeriod))
		pausePod.Spec.TerminationGracePeriodSeconds = &gracePeriod
		pausePod = podClient.CreateSync(ctx, pausePod)

		ginkgo.By("Try to connect to the webserver")
		// Wait until we are able to connect to the Pod
		podIPAddress := net.JoinHostPort(webserverPod.Status.PodIP, strconv.Itoa(80))
		execHostnameTest(*pausePod, podIPAddress, webserverPod.Name)

		// pod client should continue to connect to the webserver after delete
		// since will be gracefully terminating
		err := cs.CoreV1().Pods(ns).Delete(ctx, pausePod.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "error deleting client pod")
		// wait some time to ensure the test hit the pod during the terminating
		time.Sleep(15 * time.Second)
		execHostnameTest(*pausePod, podIPAddress, webserverPod.Name)
	})

})
