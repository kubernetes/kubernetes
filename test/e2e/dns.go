/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/unversioned"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/wait"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var dnsServiceLableSelector = labels.Set{
	"k8s-app":                       "kube-dns",
	"kubernetes.io/cluster-service": "true",
}.AsSelector()

func createDNSPod(namespace, wheezyProbeCmd, jessieProbeCmd string) *api.Pod {
	pod := &api.Pod{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Pod",
			APIVersion: latest.GroupOrDie("").Version,
		},
		ObjectMeta: api.ObjectMeta{
			Name:      "dns-test-" + string(util.NewUUID()),
			Namespace: namespace,
		},
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					Name: "results",
					VolumeSource: api.VolumeSource{
						EmptyDir: &api.EmptyDirVolumeSource{},
					},
				},
			},
			Containers: []api.Container{
				// TODO: Consider scraping logs instead of running a webserver.
				{
					Name:  "webserver",
					Image: "gcr.io/google_containers/test-webserver",
					Ports: []api.ContainerPort{
						{
							Name:          "http",
							ContainerPort: 80,
						},
					},
					VolumeMounts: []api.VolumeMount{
						{
							Name:      "results",
							MountPath: "/results",
						},
					},
				},
				{
					Name:    "querier",
					Image:   "gcr.io/google_containers/dnsutils",
					Command: []string{"sh", "-c", wheezyProbeCmd},
					VolumeMounts: []api.VolumeMount{
						{
							Name:      "results",
							MountPath: "/results",
						},
					},
				},
				{
					Name:    "jessie-querier",
					Image:   "gcr.io/google_containers/jessie-dnsutils",
					Command: []string{"sh", "-c", jessieProbeCmd},
					VolumeMounts: []api.VolumeMount{
						{
							Name:      "results",
							MountPath: "/results",
						},
					},
				},
			},
		},
	}
	return pod
}

func createProbeCommand(namesToResolve []string, fileNamePrefix string) (string, []string) {
	fileNames := make([]string, 0, len(namesToResolve)*2)
	probeCmd := "for i in `seq 1 600`; do "
	for _, name := range namesToResolve {
		// Resolve by TCP and UDP DNS.  Use $$(...) because $(...) is
		// expanded by kubernetes (though this won't expand so should
		// remain a literal, safe > sorry).
		lookup := "A"
		if strings.HasPrefix(name, "_") {
			lookup = "SRV"
		}
		fileName := fmt.Sprintf("%s_udp@%s", fileNamePrefix, name)
		fileNames = append(fileNames, fileName)
		probeCmd += fmt.Sprintf(`test -n "$$(dig +notcp +noall +answer +search %s %s)" && echo OK > /results/%s;`, name, lookup, fileName)
		fileName = fmt.Sprintf("%s_tcp@%s", fileNamePrefix, name)
		fileNames = append(fileNames, fileName)
		probeCmd += fmt.Sprintf(`test -n "$$(dig +tcp +noall +answer +search %s %s)" && echo OK > /results/%s;`, name, lookup, fileName)
	}
	probeCmd += "sleep 1; done"
	return probeCmd, fileNames
}

func assertFilesExist(fileNames []string, fileDir string, pod *api.Pod, client *client.Client) {
	var failed []string

	expectNoError(wait.Poll(time.Second*2, time.Second*60, func() (bool, error) {
		failed = []string{}
		for _, fileName := range fileNames {
			if _, err := client.Get().
				Prefix("proxy").
				Resource("pods").
				Namespace(pod.Namespace).
				Name(pod.Name).
				Suffix(fileDir, fileName).
				Do().Raw(); err != nil {
				Logf("Unable to read %s from pod %s: %v", fileName, pod.Name, err)
				failed = append(failed, fileName)
			}
		}
		if len(failed) == 0 {
			return true, nil
		}
		Logf("Lookups using %s failed for: %v\n", pod.Name, failed)
		return false, nil
	}))
	Expect(len(failed)).To(Equal(0))
}

func validateDNSResults(f *Framework, pod *api.Pod, fileNames []string) {

	By("submitting the pod to kubernetes")
	podClient := f.Client.Pods(f.Namespace.Name)
	defer func() {
		By("deleting the pod")
		defer GinkgoRecover()
		podClient.Delete(pod.Name, api.NewDeleteOptions(0))
	}()
	if _, err := podClient.Create(pod); err != nil {
		Failf("Failed to create %s pod: %v", pod.Name, err)
	}

	expectNoError(f.WaitForPodRunning(pod.Name))

	By("retrieving the pod")
	pod, err := podClient.Get(pod.Name)
	if err != nil {
		Failf("Failed to get pod %s: %v", pod.Name, err)
	}

	// Try to find results for each expected name.
	By("looking for the results for each expected name from probiers")
	assertFilesExist(fileNames, "results", pod, f.Client)

	// TODO: probe from the host, too.

	Logf("DNS probes using %s succeeded\n", pod.Name)
}

var _ = Describe("DNS", func() {
	f := NewFramework("dns")

	It("should provide DNS for the cluster", func() {
		// TODO: support DNS on vagrant #3580
		SkipIfProviderIs("vagrant")

		systemClient := f.Client.Pods(api.NamespaceSystem)
		By("Waiting for DNS Service to be Running")
		dnsPods, err := systemClient.List(dnsServiceLableSelector, fields.Everything())
		if err != nil {
			Failf("Failed to list all dns service pods")
		}
		if len(dnsPods.Items) != 1 {
			Failf("Unexpected number of pods (%d) matches the label selector %v", len(dnsPods.Items), dnsServiceLableSelector.String())
		}
		expectNoError(waitForPodRunningInNamespace(f.Client, dnsPods.Items[0].Name, api.NamespaceSystem))

		// All the names we need to be able to resolve.
		// TODO: Spin up a separate test service and test that dns works for that service.
		namesToResolve := []string{
			"kubernetes.default",
			"kubernetes.default.svc",
			"kubernetes.default.svc.cluster.local",
			"google.com",
		}
		// Added due to #8512. This is critical for GCE and GKE deployments.
		if providerIs("gce", "gke") {
			namesToResolve = append(namesToResolve, "metadata")
		}

		wheezyProbeCmd, wheezyFileNames := createProbeCommand(namesToResolve, "wheezy")
		jessieProbeCmd, jessieFileNames := createProbeCommand(namesToResolve, "jessie")

		// Run a pod which probes DNS and exposes the results by HTTP.
		By("creating a pod to probe DNS")
		pod := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd)
		validateDNSResults(f, pod, append(wheezyFileNames, jessieFileNames...))
	})

	It("should provide DNS for services", func() {
		// TODO: support DNS on vagrant #3580
		SkipIfProviderIs("vagrant")

		systemClient := f.Client.Pods(api.NamespaceSystem)

		By("Waiting for DNS Service to be Running")
		dnsPods, err := systemClient.List(dnsServiceLableSelector, fields.Everything())
		if err != nil {
			Failf("Failed to list all dns service pods")
		}
		if len(dnsPods.Items) != 1 {
			Failf("Unexpected number of pods (%d) matches the label selector %v", len(dnsPods.Items), dnsServiceLableSelector.String())
		}
		expectNoError(waitForPodRunningInNamespace(f.Client, dnsPods.Items[0].Name, api.NamespaceSystem))

		// Create a test headless service.
		By("Creating a test headless service")
		testServiceSelector := map[string]string{
			"dns-test": "true",
		}
		headlessService := &api.Service{
			ObjectMeta: api.ObjectMeta{
				Name: "test-service",
			},
			Spec: api.ServiceSpec{
				ClusterIP: "None",
				Ports: []api.ServicePort{
					{Port: 80, Name: "http", Protocol: "TCP"},
				},
				Selector: testServiceSelector,
			},
		}

		_, err = f.Client.Services(f.Namespace.Name).Create(headlessService)
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			By("deleting the test headless service")
			defer GinkgoRecover()
			f.Client.Services(f.Namespace.Name).Delete(headlessService.Name)
		}()

		regularService := &api.Service{
			ObjectMeta: api.ObjectMeta{
				Name: "test-service-2",
			},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{
					{Port: 80, Name: "http", Protocol: "TCP"},
				},
				Selector: testServiceSelector,
			},
		}

		_, err = f.Client.Services(f.Namespace.Name).Create(regularService)
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			By("deleting the test service")
			defer GinkgoRecover()
			f.Client.Services(f.Namespace.Name).Delete(regularService.Name)
		}()

		// All the names we need to be able to resolve.
		// TODO: Create more endpoints and ensure that multiple A records are returned
		// for headless service.
		namesToResolve := []string{
			fmt.Sprintf("%s", headlessService.Name),
			fmt.Sprintf("%s.%s", headlessService.Name, f.Namespace.Name),
			fmt.Sprintf("%s.%s.svc", headlessService.Name, f.Namespace.Name),
			fmt.Sprintf("_http._tcp.%s.%s.svc", headlessService.Name, f.Namespace.Name),
			fmt.Sprintf("_http._tcp.%s.%s.svc", regularService.Name, f.Namespace.Name),
		}

		wheezyProbeCmd, wheezyFileNames := createProbeCommand(namesToResolve, "wheezy")
		jessieProbeCmd, jessieFileNames := createProbeCommand(namesToResolve, "jessie")

		// Run a pod which probes DNS and exposes the results by HTTP.
		By("creating a pod to probe DNS")
		pod := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd)
		pod.ObjectMeta.Labels = testServiceSelector

		validateDNSResults(f, pod, append(wheezyFileNames, jessieFileNames...))
	})

})
