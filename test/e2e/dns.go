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
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/pod"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
)

const dnsTestPodHostName = "dns-querier-1"
const dnsTestServiceName = "dns-test-service"

var dnsServiceLabelSelector = labels.Set{
	"k8s-app":                       "kube-dns",
	"kubernetes.io/cluster-service": "true",
}.AsSelector()

func createDNSPod(namespace, wheezyProbeCmd, jessieProbeCmd string, useAnnotation bool) *api.Pod {
	dnsPod := &api.Pod{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Pod",
			APIVersion: registered.GroupOrDie(api.GroupName).GroupVersion.String(),
		},
		ObjectMeta: api.ObjectMeta{
			Name:      "dns-test-" + string(uuid.NewUUID()),
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
					Image: "gcr.io/google_containers/test-webserver:e2e",
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
					Image:   "gcr.io/google_containers/dnsutils:e2e",
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
					Image:   "gcr.io/google_containers/jessie-dnsutils:e2e",
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

	if useAnnotation {
		dnsPod.ObjectMeta.Annotations = map[string]string{
			pod.PodHostnameAnnotation:  dnsTestPodHostName,
			pod.PodSubdomainAnnotation: dnsTestServiceName,
		}
	} else {
		dnsPod.Spec.Hostname = dnsTestPodHostName
		dnsPod.Spec.Subdomain = dnsTestServiceName
	}
	return dnsPod
}

func createProbeCommand(namesToResolve []string, hostEntries []string, ptrLookupIP string, fileNamePrefix, namespace string) (string, []string) {
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

	for _, name := range hostEntries {
		fileName := fmt.Sprintf("%s_hosts@%s", fileNamePrefix, name)
		fileNames = append(fileNames, fileName)
		probeCmd += fmt.Sprintf(`test -n "$$(getent hosts %s)" && echo OK > /results/%s;`, name, fileName)
	}

	podARecByUDPFileName := fmt.Sprintf("%s_udp@PodARecord", fileNamePrefix)
	podARecByTCPFileName := fmt.Sprintf("%s_tcp@PodARecord", fileNamePrefix)
	probeCmd += fmt.Sprintf(`podARec=$$(hostname -i| awk -F. '{print $$1"-"$$2"-"$$3"-"$$4".%s.pod.cluster.local"}');`, namespace)
	probeCmd += fmt.Sprintf(`test -n "$$(dig +notcp +noall +answer +search $${podARec} A)" && echo OK > /results/%s;`, podARecByUDPFileName)
	probeCmd += fmt.Sprintf(`test -n "$$(dig +tcp +noall +answer +search $${podARec} A)" && echo OK > /results/%s;`, podARecByTCPFileName)
	fileNames = append(fileNames, podARecByUDPFileName)
	fileNames = append(fileNames, podARecByTCPFileName)

	if len(ptrLookupIP) > 0 {
		ptrLookup := fmt.Sprintf("%s.in-addr.arpa.", strings.Join(reverseArray(strings.Split(ptrLookupIP, ".")), "."))
		ptrRecByUDPFileName := fmt.Sprintf("%s_udp@PTR", ptrLookupIP)
		ptrRecByTCPFileName := fmt.Sprintf("%s_tcp@PTR", ptrLookupIP)
		probeCmd += fmt.Sprintf(`test -n "$$(dig +notcp +noall +answer +search %s PTR)" && echo OK > /results/%s;`, ptrLookup, ptrRecByUDPFileName)
		probeCmd += fmt.Sprintf(`test -n "$$(dig +tcp +noall +answer +search %s PTR)" && echo OK > /results/%s;`, ptrLookup, ptrRecByTCPFileName)
		fileNames = append(fileNames, ptrRecByUDPFileName)
		fileNames = append(fileNames, ptrRecByTCPFileName)
	}

	probeCmd += "sleep 1; done"
	return probeCmd, fileNames
}

func assertFilesExist(fileNames []string, fileDir string, pod *api.Pod, client *client.Client) {
	var failed []string

	framework.ExpectNoError(wait.Poll(time.Second*2, time.Second*60, func() (bool, error) {
		failed = []string{}
		subResourceProxyAvailable, err := framework.ServerVersionGTE(framework.SubResourcePodProxyVersion, client)
		if err != nil {
			return false, err
		}
		for _, fileName := range fileNames {
			if subResourceProxyAvailable {
				_, err = client.Get().
					Namespace(pod.Namespace).
					Resource("pods").
					SubResource("proxy").
					Name(pod.Name).
					Suffix(fileDir, fileName).
					Do().Raw()
			} else {
				_, err = client.Get().
					Prefix("proxy").
					Resource("pods").
					Namespace(pod.Namespace).
					Name(pod.Name).
					Suffix(fileDir, fileName).
					Do().Raw()
			}
			if err != nil {
				framework.Logf("Unable to read %s from pod %s: %v", fileName, pod.Name, err)
				failed = append(failed, fileName)
			}
		}
		if len(failed) == 0 {
			return true, nil
		}
		framework.Logf("Lookups using %s failed for: %v\n", pod.Name, failed)
		return false, nil
	}))
	Expect(len(failed)).To(Equal(0))
}

func validateDNSResults(f *framework.Framework, pod *api.Pod, fileNames []string) {

	By("submitting the pod to kubernetes")
	podClient := f.Client.Pods(f.Namespace.Name)
	defer func() {
		By("deleting the pod")
		defer GinkgoRecover()
		podClient.Delete(pod.Name, api.NewDeleteOptions(0))
	}()
	if _, err := podClient.Create(pod); err != nil {
		framework.Failf("Failed to create %s pod: %v", pod.Name, err)
	}

	framework.ExpectNoError(f.WaitForPodRunning(pod.Name))

	By("retrieving the pod")
	pod, err := podClient.Get(pod.Name)
	if err != nil {
		framework.Failf("Failed to get pod %s: %v", pod.Name, err)
	}
	// Try to find results for each expected name.
	By("looking for the results for each expected name from probiers")
	assertFilesExist(fileNames, "results", pod, f.Client)

	// TODO: probe from the host, too.

	framework.Logf("DNS probes using %s succeeded\n", pod.Name)
}

func verifyDNSPodIsRunning(f *framework.Framework) {
	systemClient := f.Client.Pods(api.NamespaceSystem)
	By("Waiting for DNS Service to be Running")
	options := api.ListOptions{LabelSelector: dnsServiceLabelSelector}
	dnsPods, err := systemClient.List(options)
	if err != nil {
		framework.Failf("Failed to list all dns service pods")
	}
	if len(dnsPods.Items) < 1 {
		framework.Failf("No pods match the label selector %v", dnsServiceLabelSelector.String())
	}
	pod := dnsPods.Items[0]
	framework.ExpectNoError(framework.WaitForPodRunningInNamespace(f.Client, &pod))
}

func createServiceSpec(serviceName string, isHeadless bool, selector map[string]string) *api.Service {
	headlessService := &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: serviceName,
		},
		Spec: api.ServiceSpec{
			Ports: []api.ServicePort{
				{Port: 80, Name: "http", Protocol: "TCP"},
			},
			Selector: selector,
		},
	}
	if isHeadless {
		headlessService.Spec.ClusterIP = "None"
	}
	return headlessService
}

func reverseArray(arr []string) []string {
	for i := 0; i < len(arr)/2; i++ {
		j := len(arr) - i - 1
		arr[i], arr[j] = arr[j], arr[i]
	}
	return arr
}

var _ = framework.KubeDescribe("DNS", func() {
	f := framework.NewDefaultFramework("dns")

	It("should provide DNS for the cluster [Conformance]", func() {
		// All the names we need to be able to resolve.
		// TODO: Spin up a separate test service and test that dns works for that service.
		namesToResolve := []string{
			"kubernetes.default",
			"kubernetes.default.svc",
			"kubernetes.default.svc.cluster.local",
			"google.com",
		}
		// Added due to #8512. This is critical for GCE and GKE deployments.
		if framework.ProviderIs("gce", "gke") {
			namesToResolve = append(namesToResolve, "metadata")
		}
		hostFQDN := fmt.Sprintf("%s.%s.%s.svc.cluster.local", dnsTestPodHostName, dnsTestServiceName, f.Namespace.Name)
		hostEntries := []string{hostFQDN, dnsTestPodHostName}
		wheezyProbeCmd, wheezyFileNames := createProbeCommand(namesToResolve, hostEntries, "", "wheezy", f.Namespace.Name)
		jessieProbeCmd, jessieFileNames := createProbeCommand(namesToResolve, hostEntries, "", "jessie", f.Namespace.Name)
		By("Running these commands on wheezy:" + wheezyProbeCmd + "\n")
		By("Running these commands on jessie:" + jessieProbeCmd + "\n")

		// Run a pod which probes DNS and exposes the results by HTTP.
		By("creating a pod to probe DNS")
		pod := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd, true)
		validateDNSResults(f, pod, append(wheezyFileNames, jessieFileNames...))
	})

	It("should provide DNS for services [Conformance]", func() {
		// Create a test headless service.
		By("Creating a test headless service")
		testServiceSelector := map[string]string{
			"dns-test": "true",
		}
		headlessService := createServiceSpec(dnsTestServiceName, true, testServiceSelector)
		_, err := f.Client.Services(f.Namespace.Name).Create(headlessService)
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			By("deleting the test headless service")
			defer GinkgoRecover()
			f.Client.Services(f.Namespace.Name).Delete(headlessService.Name)
		}()

		regularService := createServiceSpec("test-service-2", false, testServiceSelector)
		regularService, err = f.Client.Services(f.Namespace.Name).Create(regularService)
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

		wheezyProbeCmd, wheezyFileNames := createProbeCommand(namesToResolve, nil, regularService.Spec.ClusterIP, "wheezy", f.Namespace.Name)
		jessieProbeCmd, jessieFileNames := createProbeCommand(namesToResolve, nil, regularService.Spec.ClusterIP, "jessie", f.Namespace.Name)
		By("Running these commands on wheezy:" + wheezyProbeCmd + "\n")
		By("Running these commands on jessie:" + jessieProbeCmd + "\n")

		// Run a pod which probes DNS and exposes the results by HTTP.
		By("creating a pod to probe DNS")
		pod := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd, false)
		pod.ObjectMeta.Labels = testServiceSelector

		validateDNSResults(f, pod, append(wheezyFileNames, jessieFileNames...))
	})

	It("should provide DNS for pods for Hostname and Subdomain Annotation", func() {
		// Create a test headless service.
		By("Creating a test headless service")
		testServiceSelector := map[string]string{
			"dns-test-hostname-attribute": "true",
		}
		serviceName := "dns-test-service-2"
		podHostname := "dns-querier-2"
		headlessService := createServiceSpec(serviceName, true, testServiceSelector)
		_, err := f.Client.Services(f.Namespace.Name).Create(headlessService)
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			By("deleting the test headless service")
			defer GinkgoRecover()
			f.Client.Services(f.Namespace.Name).Delete(headlessService.Name)
		}()

		hostFQDN := fmt.Sprintf("%s.%s.%s.svc.cluster.local", podHostname, serviceName, f.Namespace.Name)
		hostNames := []string{hostFQDN, podHostname}
		namesToResolve := []string{hostFQDN}
		wheezyProbeCmd, wheezyFileNames := createProbeCommand(namesToResolve, hostNames, "", "wheezy", f.Namespace.Name)
		jessieProbeCmd, jessieFileNames := createProbeCommand(namesToResolve, hostNames, "", "jessie", f.Namespace.Name)
		By("Running these commands on wheezy:" + wheezyProbeCmd + "\n")
		By("Running these commands on jessie:" + jessieProbeCmd + "\n")

		// Run a pod which probes DNS and exposes the results by HTTP.
		By("creating a pod to probe DNS")
		pod1 := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd, true)
		pod1.ObjectMeta.Labels = testServiceSelector
		pod1.ObjectMeta.Annotations = map[string]string{
			pod.PodHostnameAnnotation:  podHostname,
			pod.PodSubdomainAnnotation: serviceName,
		}

		validateDNSResults(f, pod1, append(wheezyFileNames, jessieFileNames...))
	})
})
