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

package e2e

import (
	"fmt"
	"path/filepath"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	dnsReadyTimeout = time.Minute
)

const queryDnsPythonTemplate string = `
import socket
try:
	socket.gethostbyname('%s')
	print 'ok'
except:
	print 'err'`

var _ = framework.KubeDescribe("ClusterDns [Feature:Example]", func() {
	f := framework.NewDefaultFramework("cluster-dns")

	var c clientset.Interface
	BeforeEach(func() {
		c = f.ClientSet
	})

	It("should create pod that uses dns", func() {
		mkpath := func(file string) string {
			return filepath.Join(framework.TestContext.RepoRoot, "examples/cluster-dns", file)
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
			namespaces[i], err = f.CreateNamespace(fmt.Sprintf("dnsexample%d", i), nil)
			Expect(err).NotTo(HaveOccurred())
		}

		for _, ns := range namespaces {
			framework.RunKubectlOrDie("create", "-f", backendRcYaml, getNsCmdFlag(ns))
		}

		for _, ns := range namespaces {
			framework.RunKubectlOrDie("create", "-f", backendSvcYaml, getNsCmdFlag(ns))
		}

		// wait for objects
		for _, ns := range namespaces {
			framework.WaitForControlledPodsRunning(c, ns.Name, backendRcName, api.Kind("ReplicationController"))
			framework.WaitForService(c, ns.Name, backendSvcName, true, framework.Poll, framework.ServiceStartTimeout)
		}
		// it is not enough that pods are running because they may be set to running, but
		// the application itself may have not been initialized. Just query the application.
		for _, ns := range namespaces {
			label := labels.SelectorFromSet(labels.Set(map[string]string{"name": backendRcName}))
			options := v1.ListOptions{LabelSelector: label.String()}
			pods, err := c.Core().Pods(ns.Name).List(options)
			Expect(err).NotTo(HaveOccurred())
			err = framework.PodsResponding(c, ns.Name, backendPodName, false, pods)
			Expect(err).NotTo(HaveOccurred(), "waiting for all pods to respond")
			framework.Logf("found %d backend pods responding in namespace %s", len(pods.Items), ns.Name)

			err = framework.ServiceResponding(c, ns.Name, backendSvcName)
			Expect(err).NotTo(HaveOccurred(), "waiting for the service to respond")
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
		options := v1.ListOptions{LabelSelector: label.String()}
		pods, err := c.Core().Pods(namespaces[0].Name).List(options)

		if err != nil || pods == nil || len(pods.Items) == 0 {
			framework.Failf("no running pods found")
		}
		podName := pods.Items[0].Name

		queryDns := fmt.Sprintf(queryDnsPythonTemplate, backendSvcName+"."+namespaces[0].Name)
		_, err = framework.LookForStringInPodExec(namespaces[0].Name, podName, []string{"python", "-c", queryDns}, "ok", dnsReadyTimeout)
		Expect(err).NotTo(HaveOccurred(), "waiting for output from pod exec")

		updatedPodYaml := prepareResourceWithReplacedString(frontendPodYaml, "dns-backend.development.cluster.local", fmt.Sprintf("dns-backend.%s.svc.cluster.local", namespaces[0].Name))

		// create a pod in each namespace
		for _, ns := range namespaces {
			framework.NewKubectlCommand("create", "-f", "-", getNsCmdFlag(ns)).WithStdinData(updatedPodYaml).ExecOrDie()
		}

		// wait until the pods have been scheduler, i.e. are not Pending anymore. Remember
		// that we cannot wait for the pods to be running because our pods terminate by themselves.
		for _, ns := range namespaces {
			err := framework.WaitForPodNotPending(c, ns.Name, frontendPodName, "")
			framework.ExpectNoError(err)
		}

		// wait for pods to print their result
		for _, ns := range namespaces {
			_, err := framework.LookForStringInLog(ns.Name, frontendPodName, frontendPodContainerName, podOutput, framework.PodStartTimeout)
			Expect(err).NotTo(HaveOccurred())
		}
	})
})

func getNsCmdFlag(ns *v1.Namespace) string {
	return fmt.Sprintf("--namespace=%v", ns.Name)
}
