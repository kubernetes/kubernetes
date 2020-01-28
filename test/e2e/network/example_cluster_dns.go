/*
Copyright 2015 The Kubernetes Authors.

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
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/onsi/ginkgo"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
)

const (
	dnsReadyTimeout = time.Minute

	// RespondingTimeout is how long to wait for a service to be responding.
	RespondingTimeout = 2 * time.Minute
)

const queryDNSPythonTemplate string = `
import socket
try:
	socket.gethostbyname('%s')
	print('ok')
except:
	print('err')`

var _ = SIGDescribe("ClusterDns [Feature:Example]", func() {
	f := framework.NewDefaultFramework("cluster-dns")

	var c clientset.Interface
	ginkgo.BeforeEach(func() {
		c = f.ClientSet
	})

	ginkgo.It("should create pod that uses dns", func() {
		mkpath := func(file string) string {
			return filepath.Join(os.Getenv("GOPATH"), "src/k8s.io/examples/staging/cluster-dns", file)
		}

		// contrary to the example, this test does not use contexts, for simplicity
		// namespaces are passed directly.
		// Also, for simplicity, we don't use yamls with namespaces, but we
		// create testing namespaces instead.

		backendRcYaml := mkpath("dns-backend-rc.yaml")
		backendRcName := "dns-backend"
		backendSvcYaml := mkpath("dns-backend-service.yaml")
		backendSvcName := "dns-backend"
		backendPodName := "dns-backend"
		frontendPodYaml := mkpath("dns-frontend-pod.yaml")
		frontendPodName := "dns-frontend"
		frontendPodContainerName := "dns-frontend"

		podOutput := "Hello World!"

		// we need two namespaces anyway, so let's forget about
		// the one created in BeforeEach and create two new ones.
		namespaces := []*v1.Namespace{nil, nil}
		for i := range namespaces {
			var err error
			namespaceName := fmt.Sprintf("dnsexample%d", i)
			namespaces[i], err = f.CreateNamespace(namespaceName, nil)
			framework.ExpectNoError(err, "failed to create namespace: %s", namespaceName)
		}

		for _, ns := range namespaces {
			framework.RunKubectlOrDie(ns.Name, "create", "-f", backendRcYaml, getNsCmdFlag(ns))
		}

		for _, ns := range namespaces {
			framework.RunKubectlOrDie(ns.Name, "create", "-f", backendSvcYaml, getNsCmdFlag(ns))
		}

		// wait for objects
		for _, ns := range namespaces {
			e2epod.WaitForControlledPodsRunning(c, ns.Name, backendRcName, api.Kind("ReplicationController"))
			framework.WaitForService(c, ns.Name, backendSvcName, true, framework.Poll, framework.ServiceStartTimeout)
		}
		// it is not enough that pods are running because they may be set to running, but
		// the application itself may have not been initialized. Just query the application.
		for _, ns := range namespaces {
			label := labels.SelectorFromSet(labels.Set(map[string]string{"name": backendRcName}))
			options := metav1.ListOptions{LabelSelector: label.String()}
			pods, err := c.CoreV1().Pods(ns.Name).List(options)
			framework.ExpectNoError(err, "failed to list pods in namespace: %s", ns.Name)
			err = e2epod.PodsResponding(c, ns.Name, backendPodName, false, pods)
			framework.ExpectNoError(err, "waiting for all pods to respond")
			framework.Logf("found %d backend pods responding in namespace %s", len(pods.Items), ns.Name)

			err = waitForServiceResponding(c, ns.Name, backendSvcName)
			framework.ExpectNoError(err, "waiting for the service to respond")
		}

		// Now another tricky part:
		// It may happen that the service name is not yet in DNS.
		// So if we start our pod, it will fail. We must make sure
		// the name is already resolvable. So let's try to query DNS from
		// the pod we have, until we find our service name.
		// This complicated code may be removed if the pod itself retried after
		// dns error or timeout.
		// This code is probably unnecessary, but let's stay on the safe side.
		label := labels.SelectorFromSet(labels.Set(map[string]string{"name": backendPodName}))
		options := metav1.ListOptions{LabelSelector: label.String()}
		pods, err := c.CoreV1().Pods(namespaces[0].Name).List(options)

		if err != nil || pods == nil || len(pods.Items) == 0 {
			framework.Failf("no running pods found")
		}
		podName := pods.Items[0].Name

		queryDNS := fmt.Sprintf(queryDNSPythonTemplate, backendSvcName+"."+namespaces[0].Name)
		_, err = framework.LookForStringInPodExec(namespaces[0].Name, podName, []string{"python", "-c", queryDNS}, "ok", dnsReadyTimeout)
		framework.ExpectNoError(err, "waiting for output from pod exec")

		updatedPodYaml := prepareResourceWithReplacedString(frontendPodYaml, fmt.Sprintf("dns-backend.development.svc.%s", framework.TestContext.ClusterDNSDomain), fmt.Sprintf("dns-backend.%s.svc.%s", namespaces[0].Name, framework.TestContext.ClusterDNSDomain))

		// create a pod in each namespace
		for _, ns := range namespaces {
			framework.NewKubectlCommand(ns.Name, "create", "-f", "-", getNsCmdFlag(ns)).WithStdinData(updatedPodYaml).ExecOrDie(ns.Name)
		}

		// wait until the pods have been scheduler, i.e. are not Pending anymore. Remember
		// that we cannot wait for the pods to be running because our pods terminate by themselves.
		for _, ns := range namespaces {
			err := e2epod.WaitForPodNotPending(c, ns.Name, frontendPodName)
			framework.ExpectNoError(err)
		}

		// wait for pods to print their result
		for _, ns := range namespaces {
			_, err := framework.LookForStringInLog(ns.Name, frontendPodName, frontendPodContainerName, podOutput, framework.PodStartTimeout)
			framework.ExpectNoError(err, "pod %s failed to print result in logs", frontendPodName)
		}
	})
})

func getNsCmdFlag(ns *v1.Namespace) string {
	return fmt.Sprintf("--namespace=%v", ns.Name)
}

// pass enough context with the 'old' parameter so that it replaces what your really intended.
func prepareResourceWithReplacedString(inputFile, old, new string) string {
	f, err := os.Open(inputFile)
	framework.ExpectNoError(err, "failed to open file: %s", inputFile)
	defer f.Close()
	data, err := ioutil.ReadAll(f)
	framework.ExpectNoError(err, "failed to read from file: %s", inputFile)
	podYaml := strings.Replace(string(data), old, new, 1)
	return podYaml
}

// waitForServiceResponding waits for the service to be responding.
func waitForServiceResponding(c clientset.Interface, ns, name string) error {
	ginkgo.By(fmt.Sprintf("trying to dial the service %s.%s via the proxy", ns, name))

	return wait.PollImmediate(framework.Poll, RespondingTimeout, func() (done bool, err error) {
		proxyRequest, errProxy := e2eservice.GetServicesProxyRequest(c, c.CoreV1().RESTClient().Get())
		if errProxy != nil {
			framework.Logf("Failed to get services proxy request: %v:", errProxy)
			return false, nil
		}

		ctx, cancel := context.WithTimeout(context.Background(), framework.SingleCallTimeout)
		defer cancel()

		body, err := proxyRequest.Namespace(ns).
			Name(name).
			Do(ctx).
			Raw()
		if err != nil {
			if ctx.Err() != nil {
				framework.Failf("Failed to GET from service %s: %v", name, err)
				return true, err
			}
			framework.Logf("Failed to GET from service %s: %v:", name, err)
			return false, nil
		}
		got := string(body)
		if len(got) == 0 {
			framework.Logf("Service %s: expected non-empty response", name)
			return false, err // stop polling
		}
		framework.Logf("Service %s: found nonempty answer: %s", name, got)
		return true, nil
	})
}
