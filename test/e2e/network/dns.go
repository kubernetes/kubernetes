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
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const dnsTestPodHostName = "dns-querier-1"
const dnsTestServiceName = "dns-test-service"

func createDNSPod(namespace, wheezyProbeCmd, jessieProbeCmd string) *v1.Pod {
	dnsPod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: testapi.Groups[v1.GroupName].GroupVersion().String(),
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "dns-test-" + string(uuid.NewUUID()),
			Namespace: namespace,
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "results",
					VolumeSource: v1.VolumeSource{
						EmptyDir: &v1.EmptyDirVolumeSource{},
					},
				},
			},
			Containers: []v1.Container{
				// TODO: Consider scraping logs instead of running a webserver.
				{
					Name:  "webserver",
					Image: imageutils.GetE2EImage(imageutils.TestWebserver),
					Ports: []v1.ContainerPort{
						{
							Name:          "http",
							ContainerPort: 80,
						},
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "results",
							MountPath: "/results",
						},
					},
				},
				{
					Name:    "querier",
					Image:   imageutils.GetE2EImage(imageutils.Dnsutils),
					Command: []string{"sh", "-c", wheezyProbeCmd},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "results",
							MountPath: "/results",
						},
					},
				},
				{
					Name:    "jessie-querier",
					Image:   imageutils.GetE2EImage(imageutils.JessieDnsutils),
					Command: []string{"sh", "-c", jessieProbeCmd},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "results",
							MountPath: "/results",
						},
					},
				},
			},
		},
	}

	dnsPod.Spec.Hostname = dnsTestPodHostName
	dnsPod.Spec.Subdomain = dnsTestServiceName

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

// createTargetedProbeCommand returns a command line that performs a DNS lookup for a specific record type
func createTargetedProbeCommand(nameToResolve string, lookup string, fileNamePrefix string) (string, string) {
	fileName := fmt.Sprintf("%s_udp@%s", fileNamePrefix, nameToResolve)
	probeCmd := fmt.Sprintf("dig +short +tries=12 +norecurse %s %s > /results/%s", nameToResolve, lookup, fileName)
	return probeCmd, fileName
}

func assertFilesExist(fileNames []string, fileDir string, pod *v1.Pod, client clientset.Interface) {
	assertFilesContain(fileNames, fileDir, pod, client, false, "")
}

func assertFilesContain(fileNames []string, fileDir string, pod *v1.Pod, client clientset.Interface, check bool, expected string) {
	var failed []string

	framework.ExpectNoError(wait.Poll(time.Second*10, time.Second*600, func() (bool, error) {
		failed = []string{}
		subResourceProxyAvailable, err := framework.ServerVersionGTE(framework.SubResourcePodProxyVersion, client.Discovery())
		if err != nil {
			return false, err
		}

		ctx, cancel := context.WithTimeout(context.Background(), framework.SingleCallTimeout)
		defer cancel()

		var contents []byte
		for _, fileName := range fileNames {
			if subResourceProxyAvailable {
				contents, err = client.Core().RESTClient().Get().
					Context(ctx).
					Namespace(pod.Namespace).
					Resource("pods").
					SubResource("proxy").
					Name(pod.Name).
					Suffix(fileDir, fileName).
					Do().Raw()
			} else {
				contents, err = client.Core().RESTClient().Get().
					Context(ctx).
					Prefix("proxy").
					Resource("pods").
					Namespace(pod.Namespace).
					Name(pod.Name).
					Suffix(fileDir, fileName).
					Do().Raw()
			}
			if err != nil {
				if ctx.Err() != nil {
					framework.Failf("Unable to read %s from pod %s: %v", fileName, pod.Name, err)
				} else {
					framework.Logf("Unable to read %s from pod %s: %v", fileName, pod.Name, err)
				}
				failed = append(failed, fileName)
			} else if check && strings.TrimSpace(string(contents)) != expected {
				framework.Logf("File %s from pod %s contains '%s' instead of '%s'", fileName, pod.Name, string(contents), expected)
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

func validateDNSResults(f *framework.Framework, pod *v1.Pod, fileNames []string) {
	By("submitting the pod to kubernetes")
	podClient := f.ClientSet.Core().Pods(f.Namespace.Name)
	defer func() {
		By("deleting the pod")
		defer GinkgoRecover()
		podClient.Delete(pod.Name, metav1.NewDeleteOptions(0))
	}()
	if _, err := podClient.Create(pod); err != nil {
		framework.Failf("Failed to create %s pod: %v", pod.Name, err)
	}

	framework.ExpectNoError(f.WaitForPodRunning(pod.Name))

	By("retrieving the pod")
	pod, err := podClient.Get(pod.Name, metav1.GetOptions{})
	if err != nil {
		framework.Failf("Failed to get pod %s: %v", pod.Name, err)
	}
	// Try to find results for each expected name.
	By("looking for the results for each expected name from probers")
	assertFilesExist(fileNames, "results", pod, f.ClientSet)

	// TODO: probe from the host, too.

	framework.Logf("DNS probes using %s succeeded\n", pod.Name)
}

func validateTargetedProbeOutput(f *framework.Framework, pod *v1.Pod, fileNames []string, value string) {
	By("submitting the pod to kubernetes")
	podClient := f.ClientSet.Core().Pods(f.Namespace.Name)
	defer func() {
		By("deleting the pod")
		defer GinkgoRecover()
		podClient.Delete(pod.Name, metav1.NewDeleteOptions(0))
	}()
	if _, err := podClient.Create(pod); err != nil {
		framework.Failf("Failed to create %s pod: %v", pod.Name, err)
	}

	framework.ExpectNoError(f.WaitForPodRunning(pod.Name))

	By("retrieving the pod")
	pod, err := podClient.Get(pod.Name, metav1.GetOptions{})
	if err != nil {
		framework.Failf("Failed to get pod %s: %v", pod.Name, err)
	}
	// Try to find the expected value for each expected name.
	By("looking for the results for each expected name from probers")
	assertFilesContain(fileNames, "results", pod, f.ClientSet, true, value)

	framework.Logf("DNS probes using %s succeeded\n", pod.Name)
}

func reverseArray(arr []string) []string {
	for i := 0; i < len(arr)/2; i++ {
		j := len(arr) - i - 1
		arr[i], arr[j] = arr[j], arr[i]
	}
	return arr
}

var _ = SIGDescribe("DNS", func() {
	f := framework.NewDefaultFramework("dns")

	It("should provide DNS for the cluster [Conformance]", func() {
		// All the names we need to be able to resolve.
		// TODO: Spin up a separate test service and test that dns works for that service.
		namesToResolve := []string{
			"kubernetes.default",
			"kubernetes.default.svc",
			"kubernetes.default.svc.cluster.local",
		}
		// Added due to #8512. This is critical for GCE and GKE deployments.
		if framework.ProviderIs("gce", "gke") {
			namesToResolve = append(namesToResolve, "google.com")
			namesToResolve = append(namesToResolve, "metadata")
		}
		hostFQDN := fmt.Sprintf("%s.%s.%s.svc.cluster.local", dnsTestPodHostName, dnsTestServiceName, f.Namespace.Name)
		hostEntries := []string{hostFQDN, dnsTestPodHostName}
		wheezyProbeCmd, wheezyFileNames := createProbeCommand(namesToResolve, hostEntries, "", "wheezy", f.Namespace.Name)
		jessieProbeCmd, jessieFileNames := createProbeCommand(namesToResolve, hostEntries, "", "jessie", f.Namespace.Name)
		By("Running these commands on wheezy: " + wheezyProbeCmd + "\n")
		By("Running these commands on jessie: " + jessieProbeCmd + "\n")

		// Run a pod which probes DNS and exposes the results by HTTP.
		By("creating a pod to probe DNS")
		pod := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd)
		validateDNSResults(f, pod, append(wheezyFileNames, jessieFileNames...))
	})

	It("should provide DNS for services [Conformance]", func() {
		// Create a test headless service.
		By("Creating a test headless service")
		testServiceSelector := map[string]string{
			"dns-test": "true",
		}
		headlessService := framework.CreateServiceSpec(dnsTestServiceName, "", true, testServiceSelector)
		_, err := f.ClientSet.Core().Services(f.Namespace.Name).Create(headlessService)
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			By("deleting the test headless service")
			defer GinkgoRecover()
			f.ClientSet.Core().Services(f.Namespace.Name).Delete(headlessService.Name, nil)
		}()

		regularService := framework.CreateServiceSpec("test-service-2", "", false, testServiceSelector)
		regularService, err = f.ClientSet.Core().Services(f.Namespace.Name).Create(regularService)
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			By("deleting the test service")
			defer GinkgoRecover()
			f.ClientSet.Core().Services(f.Namespace.Name).Delete(regularService.Name, nil)
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
		By("Running these commands on wheezy: " + wheezyProbeCmd + "\n")
		By("Running these commands on jessie: " + jessieProbeCmd + "\n")

		// Run a pod which probes DNS and exposes the results by HTTP.
		By("creating a pod to probe DNS")
		pod := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd)
		pod.ObjectMeta.Labels = testServiceSelector

		validateDNSResults(f, pod, append(wheezyFileNames, jessieFileNames...))
	})

	It("should provide DNS for pods for Hostname and Subdomain", func() {
		// Create a test headless service.
		By("Creating a test headless service")
		testServiceSelector := map[string]string{
			"dns-test-hostname-attribute": "true",
		}
		serviceName := "dns-test-service-2"
		podHostname := "dns-querier-2"
		headlessService := framework.CreateServiceSpec(serviceName, "", true, testServiceSelector)
		_, err := f.ClientSet.Core().Services(f.Namespace.Name).Create(headlessService)
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			By("deleting the test headless service")
			defer GinkgoRecover()
			f.ClientSet.Core().Services(f.Namespace.Name).Delete(headlessService.Name, nil)
		}()

		hostFQDN := fmt.Sprintf("%s.%s.%s.svc.cluster.local", podHostname, serviceName, f.Namespace.Name)
		hostNames := []string{hostFQDN, podHostname}
		namesToResolve := []string{hostFQDN}
		wheezyProbeCmd, wheezyFileNames := createProbeCommand(namesToResolve, hostNames, "", "wheezy", f.Namespace.Name)
		jessieProbeCmd, jessieFileNames := createProbeCommand(namesToResolve, hostNames, "", "jessie", f.Namespace.Name)
		By("Running these commands on wheezy: " + wheezyProbeCmd + "\n")
		By("Running these commands on jessie: " + jessieProbeCmd + "\n")

		// Run a pod which probes DNS and exposes the results by HTTP.
		By("creating a pod to probe DNS")
		pod1 := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd)
		pod1.ObjectMeta.Labels = testServiceSelector
		pod1.Spec.Hostname = podHostname
		pod1.Spec.Subdomain = serviceName

		validateDNSResults(f, pod1, append(wheezyFileNames, jessieFileNames...))
	})

	It("should provide DNS for ExternalName services", func() {
		// Create a test ExternalName service.
		By("Creating a test externalName service")
		serviceName := "dns-test-service-3"
		externalNameService := framework.CreateServiceSpec(serviceName, "foo.example.com", false, nil)
		_, err := f.ClientSet.Core().Services(f.Namespace.Name).Create(externalNameService)
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			By("deleting the test externalName service")
			defer GinkgoRecover()
			f.ClientSet.Core().Services(f.Namespace.Name).Delete(externalNameService.Name, nil)
		}()

		hostFQDN := fmt.Sprintf("%s.%s.svc.cluster.local", serviceName, f.Namespace.Name)
		wheezyProbeCmd, wheezyFileName := createTargetedProbeCommand(hostFQDN, "CNAME", "wheezy")
		jessieProbeCmd, jessieFileName := createTargetedProbeCommand(hostFQDN, "CNAME", "jessie")
		By("Running these commands on wheezy: " + wheezyProbeCmd + "\n")
		By("Running these commands on jessie: " + jessieProbeCmd + "\n")

		// Run a pod which probes DNS and exposes the results by HTTP.
		By("creating a pod to probe DNS")
		pod1 := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd)

		validateTargetedProbeOutput(f, pod1, []string{wheezyFileName, jessieFileName}, "foo.example.com.")

		// Test changing the externalName field
		By("changing the externalName to bar.example.com")
		_, err = framework.UpdateService(f.ClientSet, f.Namespace.Name, serviceName, func(s *v1.Service) {
			s.Spec.ExternalName = "bar.example.com"
		})
		Expect(err).NotTo(HaveOccurred())
		wheezyProbeCmd, wheezyFileName = createTargetedProbeCommand(hostFQDN, "CNAME", "wheezy")
		jessieProbeCmd, jessieFileName = createTargetedProbeCommand(hostFQDN, "CNAME", "jessie")
		By("Running these commands on wheezy: " + wheezyProbeCmd + "\n")
		By("Running these commands on jessie: " + jessieProbeCmd + "\n")

		// Run a pod which probes DNS and exposes the results by HTTP.
		By("creating a second pod to probe DNS")
		pod2 := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd)

		validateTargetedProbeOutput(f, pod2, []string{wheezyFileName, jessieFileName}, "bar.example.com.")

		// Test changing type from ExternalName to ClusterIP
		By("changing the service to type=ClusterIP")
		_, err = framework.UpdateService(f.ClientSet, f.Namespace.Name, serviceName, func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeClusterIP
			s.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "http", Protocol: "TCP"},
			}
		})
		Expect(err).NotTo(HaveOccurred())
		wheezyProbeCmd, wheezyFileName = createTargetedProbeCommand(hostFQDN, "A", "wheezy")
		jessieProbeCmd, jessieFileName = createTargetedProbeCommand(hostFQDN, "A", "jessie")
		By("Running these commands on wheezy: " + wheezyProbeCmd + "\n")
		By("Running these commands on jessie: " + jessieProbeCmd + "\n")

		// Run a pod which probes DNS and exposes the results by HTTP.
		By("creating a third pod to probe DNS")
		pod3 := createDNSPod(f.Namespace.Name, wheezyProbeCmd, jessieProbeCmd)

		svc, err := f.ClientSet.Core().Services(f.Namespace.Name).Get(externalNameService.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())

		validateTargetedProbeOutput(f, pod3, []string{wheezyFileName, jessieFileName}, svc.Spec.ClusterIP)
	})
})
