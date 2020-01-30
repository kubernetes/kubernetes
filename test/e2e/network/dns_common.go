/*
Copyright 2017 The Kubernetes Authors.

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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	dnsutil "github.com/miekg/dns"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

type dnsTestCommon struct {
	f      *framework.Framework
	c      clientset.Interface
	ns     string
	name   string
	labels []string

	dnsPod       *v1.Pod
	utilPod      *v1.Pod
	utilService  *v1.Service
	dnsServerPod *v1.Pod

	cm *v1.ConfigMap
}

func newDNSTestCommon() dnsTestCommon {
	return dnsTestCommon{
		f:  framework.NewDefaultFramework("dns-config-map"),
		ns: "kube-system",
	}
}

func (t *dnsTestCommon) init() {
	ginkgo.By("Finding a DNS pod")
	label := labels.SelectorFromSet(labels.Set(map[string]string{"k8s-app": "kube-dns"}))
	options := metav1.ListOptions{LabelSelector: label.String()}

	namespace := "kube-system"
	pods, err := t.f.ClientSet.CoreV1().Pods(namespace).List(options)
	framework.ExpectNoError(err, "failed to list pods in namespace: %s", namespace)
	gomega.Expect(len(pods.Items)).Should(gomega.BeNumerically(">=", 1))

	t.dnsPod = &pods.Items[0]
	framework.Logf("Using DNS pod: %v", t.dnsPod.Name)

	if strings.Contains(t.dnsPod.Name, "coredns") {
		t.name = "coredns"
	} else {
		t.name = "kube-dns"
	}
}

func (t *dnsTestCommon) checkDNSRecordFrom(name string, predicate func([]string) bool, target string, timeout time.Duration) {
	var actual []string

	err := wait.PollImmediate(
		time.Duration(1)*time.Second,
		timeout,
		func() (bool, error) {
			actual = t.runDig(name, target)
			if predicate(actual) {
				return true, nil
			}
			return false, nil
		})

	if err != nil {
		framework.Failf("dig result did not match: %#v after %v",
			actual, timeout)
	}
}

// runDig queries for `dnsName`. Returns a list of responses.
func (t *dnsTestCommon) runDig(dnsName, target string) []string {
	cmd := []string{"/usr/bin/dig", "+short"}
	switch target {
	case "coredns":
		cmd = append(cmd, "@"+t.dnsPod.Status.PodIP)
	case "kube-dns":
		cmd = append(cmd, "@"+t.dnsPod.Status.PodIP, "-p", "10053")
	case "ptr-record":
		cmd = append(cmd, "-x")
	case "cluster-dns":
	case "cluster-dns-ipv6":
		cmd = append(cmd, "AAAA")
	default:
		panic(fmt.Errorf("invalid target: " + target))
	}
	cmd = append(cmd, dnsName)

	stdout, stderr, err := t.f.ExecWithOptions(framework.ExecOptions{
		Command:       cmd,
		Namespace:     t.f.Namespace.Name,
		PodName:       t.utilPod.Name,
		ContainerName: "util",
		CaptureStdout: true,
		CaptureStderr: true,
	})

	framework.Logf("Running dig: %v, stdout: %q, stderr: %q, err: %v",
		cmd, stdout, stderr, err)

	if stdout == "" {
		return []string{}
	}
	return strings.Split(stdout, "\n")
}

func (t *dnsTestCommon) setConfigMap(cm *v1.ConfigMap) {
	if t.cm != nil {
		t.cm = cm
	}

	cm.ObjectMeta.Namespace = t.ns
	cm.ObjectMeta.Name = t.name

	options := metav1.ListOptions{
		FieldSelector: fields.Set{
			"metadata.namespace": t.ns,
			"metadata.name":      t.name,
		}.AsSelector().String(),
	}
	cmList, err := t.c.CoreV1().ConfigMaps(t.ns).List(options)
	framework.ExpectNoError(err, "failed to list ConfigMaps in namespace: %s", t.ns)

	if len(cmList.Items) == 0 {
		ginkgo.By(fmt.Sprintf("Creating the ConfigMap (%s:%s) %+v", t.ns, t.name, *cm))
		_, err := t.c.CoreV1().ConfigMaps(t.ns).Create(cm)
		framework.ExpectNoError(err, "failed to create ConfigMap (%s:%s) %+v", t.ns, t.name, *cm)
	} else {
		ginkgo.By(fmt.Sprintf("Updating the ConfigMap (%s:%s) to %+v", t.ns, t.name, *cm))
		_, err := t.c.CoreV1().ConfigMaps(t.ns).Update(cm)
		framework.ExpectNoError(err, "failed to update ConfigMap (%s:%s) to %+v", t.ns, t.name, *cm)
	}
}

func (t *dnsTestCommon) fetchDNSConfigMapData() map[string]string {
	if t.name == "coredns" {
		pcm, err := t.c.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(t.name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get DNS ConfigMap: %s", t.name)
		return pcm.Data
	}
	return nil
}

func (t *dnsTestCommon) restoreDNSConfigMap(configMapData map[string]string) {
	if t.name == "coredns" {
		t.setConfigMap(&v1.ConfigMap{Data: configMapData})
		t.deleteCoreDNSPods()
	} else {
		t.c.CoreV1().ConfigMaps(t.ns).Delete(t.name, nil)
	}
}

func (t *dnsTestCommon) deleteConfigMap() {
	ginkgo.By(fmt.Sprintf("Deleting the ConfigMap (%s:%s)", t.ns, t.name))
	t.cm = nil
	err := t.c.CoreV1().ConfigMaps(t.ns).Delete(t.name, nil)
	framework.ExpectNoError(err, "failed to delete config map: %s", t.name)
}

func (t *dnsTestCommon) createUtilPodLabel(baseName string) {
	// Actual port # doesn't matter, just needs to exist.
	const servicePort = 10101

	t.utilPod = &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind: "Pod",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace:    t.f.Namespace.Name,
			Labels:       map[string]string{"app": baseName},
			GenerateName: baseName + "-",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    "util",
					Image:   imageutils.GetE2EImage(imageutils.Dnsutils),
					Command: []string{"sleep", "10000"},
					Ports: []v1.ContainerPort{
						{ContainerPort: servicePort, Protocol: v1.ProtocolTCP},
					},
				},
			},
		},
	}

	var err error
	t.utilPod, err = t.c.CoreV1().Pods(t.f.Namespace.Name).Create(t.utilPod)
	framework.ExpectNoError(err, "failed to create pod: %v", t.utilPod)
	framework.Logf("Created pod %v", t.utilPod)
	err = t.f.WaitForPodRunning(t.utilPod.Name)
	framework.ExpectNoError(err, "pod failed to start running: %v", t.utilPod)

	t.utilService = &v1.Service{
		TypeMeta: metav1.TypeMeta{
			Kind: "Service",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: t.f.Namespace.Name,
			Name:      baseName,
		},
		Spec: v1.ServiceSpec{
			Selector: map[string]string{"app": baseName},
			Ports: []v1.ServicePort{
				{
					Protocol:   v1.ProtocolTCP,
					Port:       servicePort,
					TargetPort: intstr.FromInt(servicePort),
				},
			},
		},
	}

	t.utilService, err = t.c.CoreV1().Services(t.f.Namespace.Name).Create(t.utilService)
	framework.ExpectNoError(err, "failed to create service: %s/%s", t.f.Namespace.Name, t.utilService.ObjectMeta.Name)
	framework.Logf("Created service %v", t.utilService)
}

func (t *dnsTestCommon) deleteUtilPod() {
	podClient := t.c.CoreV1().Pods(t.f.Namespace.Name)
	if err := podClient.Delete(t.utilPod.Name, metav1.NewDeleteOptions(0)); err != nil {
		framework.Logf("Delete of pod %v/%v failed: %v",
			t.utilPod.Namespace, t.utilPod.Name, err)
	}
}

// deleteCoreDNSPods manually deletes the CoreDNS pods to apply the changes to the ConfigMap.
func (t *dnsTestCommon) deleteCoreDNSPods() {

	label := labels.SelectorFromSet(labels.Set(map[string]string{"k8s-app": "kube-dns"}))
	options := metav1.ListOptions{LabelSelector: label.String()}

	pods, err := t.f.ClientSet.CoreV1().Pods("kube-system").List(options)
	framework.ExpectNoError(err, "failed to list pods of kube-system with label %q", label.String())
	podClient := t.c.CoreV1().Pods(metav1.NamespaceSystem)

	for _, pod := range pods.Items {
		err = podClient.Delete(pod.Name, metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err, "failed to delete pod: %s", pod.Name)
	}
}

func generateDNSServerPod(aRecords map[string]string) *v1.Pod {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind: "Pod",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "e2e-dns-configmap-dns-server-",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "dns",
					Image: imageutils.GetE2EImage(imageutils.Dnsutils),
					Command: []string{
						"/usr/sbin/dnsmasq",
						"-u", "root",
						"-k",
						"--log-facility", "-",
						"-q",
					},
				},
			},
			DNSPolicy: "Default",
		},
	}

	for name, ip := range aRecords {
		pod.Spec.Containers[0].Command = append(
			pod.Spec.Containers[0].Command,
			fmt.Sprintf("-A/%v/%v", name, ip))
	}
	return pod
}

func (t *dnsTestCommon) createDNSPodFromObj(pod *v1.Pod) {
	t.dnsServerPod = pod

	var err error
	t.dnsServerPod, err = t.c.CoreV1().Pods(t.f.Namespace.Name).Create(t.dnsServerPod)
	framework.ExpectNoError(err, "failed to create pod: %v", t.dnsServerPod)
	framework.Logf("Created pod %v", t.dnsServerPod)
	err = t.f.WaitForPodRunning(t.dnsServerPod.Name)
	framework.ExpectNoError(err, "pod failed to start running: %v", t.dnsServerPod)

	t.dnsServerPod, err = t.c.CoreV1().Pods(t.f.Namespace.Name).Get(
		t.dnsServerPod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "failed to get pod: %s", t.dnsServerPod.Name)
}

func (t *dnsTestCommon) createDNSServer(aRecords map[string]string) {
	t.createDNSPodFromObj(generateDNSServerPod(aRecords))
}

func (t *dnsTestCommon) createDNSServerWithPtrRecord(isIPv6 bool) {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind: "Pod",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "e2e-dns-configmap-dns-server-",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "dns",
					Image: imageutils.GetE2EImage(imageutils.Dnsutils),
					Command: []string{
						"/usr/sbin/dnsmasq",
						"-u", "root",
						"-k",
						"--log-facility", "-",
						"-q",
					},
				},
			},
			DNSPolicy: "Default",
		},
	}

	if isIPv6 {
		pod.Spec.Containers[0].Command = append(
			pod.Spec.Containers[0].Command,
			fmt.Sprintf("--host-record=my.test,2001:db8::29"))
	} else {
		pod.Spec.Containers[0].Command = append(
			pod.Spec.Containers[0].Command,
			fmt.Sprintf("--host-record=my.test,192.0.2.123"))
	}

	t.createDNSPodFromObj(pod)
}

func (t *dnsTestCommon) deleteDNSServerPod() {
	podClient := t.c.CoreV1().Pods(t.f.Namespace.Name)
	if err := podClient.Delete(t.dnsServerPod.Name, metav1.NewDeleteOptions(0)); err != nil {
		framework.Logf("Delete of pod %v/%v failed: %v",
			t.utilPod.Namespace, t.dnsServerPod.Name, err)
	}
}

func createDNSPod(namespace, wheezyProbeCmd, jessieProbeCmd, podHostName, serviceName string) *v1.Pod {
	dnsPod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
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

	dnsPod.Spec.Hostname = podHostName
	dnsPod.Spec.Subdomain = serviceName

	return dnsPod
}

func createProbeCommand(namesToResolve []string, hostEntries []string, ptrLookupIP string, fileNamePrefix, namespace, dnsDomain string, isIPv6 bool) (string, []string) {
	fileNames := make([]string, 0, len(namesToResolve)*2)
	probeCmd := "for i in `seq 1 600`; do "
	dnsRecord := "A"
	if isIPv6 {
		dnsRecord = "AAAA"
	}
	for _, name := range namesToResolve {
		// Resolve by TCP and UDP DNS.  Use $$(...) because $(...) is
		// expanded by kubernetes (though this won't expand so should
		// remain a literal, safe > sorry).
		lookup := fmt.Sprintf("%s %s", name, dnsRecord)
		if strings.HasPrefix(name, "_") {
			lookup = fmt.Sprintf("%s SRV", name)
		}
		fileName := fmt.Sprintf("%s_udp@%s", fileNamePrefix, name)
		fileNames = append(fileNames, fileName)
		probeCmd += fmt.Sprintf(`check="$$(dig +notcp +noall +answer +search %s)" && test -n "$$check" && echo OK > /results/%s;`, lookup, fileName)
		fileName = fmt.Sprintf("%s_tcp@%s", fileNamePrefix, name)
		fileNames = append(fileNames, fileName)
		probeCmd += fmt.Sprintf(`check="$$(dig +tcp +noall +answer +search %s)" && test -n "$$check" && echo OK > /results/%s;`, lookup, fileName)
	}

	for _, name := range hostEntries {
		fileName := fmt.Sprintf("%s_hosts@%s", fileNamePrefix, name)
		fileNames = append(fileNames, fileName)
		probeCmd += fmt.Sprintf(`test -n "$$(getent hosts %s)" && echo OK > /results/%s;`, name, fileName)
	}

	podARecByUDPFileName := fmt.Sprintf("%s_udp@PodARecord", fileNamePrefix)
	podARecByTCPFileName := fmt.Sprintf("%s_tcp@PodARecord", fileNamePrefix)

	// getent doesn't work properly on Windows hosts and hostname -i doesn't return an IPv6 address
	// so we  have to use a different command per IP family
	if isIPv6 {
		probeCmd += fmt.Sprintf(`podARec=$$(getent hosts $$(hostname -s) | tr ":." "-" | awk '{print $$1".%s.pod.%s"}');`, namespace, dnsDomain)
	} else {
		probeCmd += fmt.Sprintf(`podARec=$$(hostname -i| awk -F. '{print $$1"-"$$2"-"$$3"-"$$4".%s.pod.%s"}');`, namespace, dnsDomain)
	}

	probeCmd += fmt.Sprintf(`check="$$(dig +notcp +noall +answer +search $${podARec} %s)" && test -n "$$check" && echo OK > /results/%s;`, dnsRecord, podARecByUDPFileName)
	probeCmd += fmt.Sprintf(`check="$$(dig +tcp +noall +answer +search $${podARec} %s)" && test -n "$$check" && echo OK > /results/%s;`, dnsRecord, podARecByTCPFileName)
	fileNames = append(fileNames, podARecByUDPFileName)
	fileNames = append(fileNames, podARecByTCPFileName)

	if len(ptrLookupIP) > 0 {
		ptrLookup, err := dnsutil.ReverseAddr(ptrLookupIP)
		if err != nil {
			framework.Failf("Unable to obtain reverse IP address record from IP %s: %v", ptrLookupIP, err)
		}
		ptrRecByUDPFileName := fmt.Sprintf("%s_udp@PTR", ptrLookupIP)
		ptrRecByTCPFileName := fmt.Sprintf("%s_tcp@PTR", ptrLookupIP)
		probeCmd += fmt.Sprintf(`check="$$(dig +notcp +noall +answer +search %s PTR)" && test -n "$$check" && echo OK > /results/%s;`, ptrLookup, ptrRecByUDPFileName)
		probeCmd += fmt.Sprintf(`check="$$(dig +tcp +noall +answer +search %s PTR)" && test -n "$$check" && echo OK > /results/%s;`, ptrLookup, ptrRecByTCPFileName)
		fileNames = append(fileNames, ptrRecByUDPFileName)
		fileNames = append(fileNames, ptrRecByTCPFileName)
	}

	probeCmd += "sleep 1; done"
	return probeCmd, fileNames
}

// createTargetedProbeCommand returns a command line that performs a DNS lookup for a specific record type
func createTargetedProbeCommand(nameToResolve string, lookup string, fileNamePrefix string) (string, string) {
	fileName := fmt.Sprintf("%s_udp@%s", fileNamePrefix, nameToResolve)
	nameLookup := fmt.Sprintf("%s %s", nameToResolve, lookup)
	probeCmd := fmt.Sprintf("for i in `seq 1 30`; do dig +short %s > /results/%s; sleep 1; done", nameLookup, fileName)
	return probeCmd, fileName
}

func assertFilesExist(fileNames []string, fileDir string, pod *v1.Pod, client clientset.Interface) {
	assertFilesContain(fileNames, fileDir, pod, client, false, "")
}

func assertFilesContain(fileNames []string, fileDir string, pod *v1.Pod, client clientset.Interface, check bool, expected string) {
	var failed []string

	framework.ExpectNoError(wait.PollImmediate(time.Second*5, time.Second*600, func() (bool, error) {
		failed = []string{}

		ctx, cancel := context.WithTimeout(context.Background(), framework.SingleCallTimeout)
		defer cancel()

		for _, fileName := range fileNames {
			contents, err := client.CoreV1().RESTClient().Get().
				Namespace(pod.Namespace).
				Resource("pods").
				SubResource("proxy").
				Name(pod.Name).
				Suffix(fileDir, fileName).
				Do(ctx).Raw()

			if err != nil {
				if ctx.Err() != nil {
					framework.Failf("Unable to read %s from pod %s/%s: %v", fileName, pod.Namespace, pod.Name, err)
				} else {
					framework.Logf("Unable to read %s from pod %s/%s: %v", fileName, pod.Namespace, pod.Name, err)
				}
				failed = append(failed, fileName)
			} else if check && strings.TrimSpace(string(contents)) != expected {
				framework.Logf("File %s from pod  %s/%s contains '%s' instead of '%s'", fileName, pod.Namespace, pod.Name, string(contents), expected)
				failed = append(failed, fileName)
			}
		}
		if len(failed) == 0 {
			return true, nil
		}
		framework.Logf("Lookups using %s/%s failed for: %v\n", pod.Namespace, pod.Name, failed)
		return false, nil
	}))
	framework.ExpectEqual(len(failed), 0)
}

func validateDNSResults(f *framework.Framework, pod *v1.Pod, fileNames []string) {
	ginkgo.By("submitting the pod to kubernetes")
	podClient := f.ClientSet.CoreV1().Pods(f.Namespace.Name)
	defer func() {
		ginkgo.By("deleting the pod")
		defer ginkgo.GinkgoRecover()
		podClient.Delete(pod.Name, metav1.NewDeleteOptions(0))
	}()
	if _, err := podClient.Create(pod); err != nil {
		framework.Failf("ginkgo.Failed to create pod %s/%s: %v", pod.Namespace, pod.Name, err)
	}

	framework.ExpectNoError(f.WaitForPodRunning(pod.Name))

	ginkgo.By("retrieving the pod")
	pod, err := podClient.Get(pod.Name, metav1.GetOptions{})
	if err != nil {
		framework.Failf("ginkgo.Failed to get pod %s/%s: %v", pod.Namespace, pod.Name, err)
	}
	// Try to find results for each expected name.
	ginkgo.By("looking for the results for each expected name from probers")
	assertFilesExist(fileNames, "results", pod, f.ClientSet)

	// TODO: probe from the host, too.

	framework.Logf("DNS probes using %s/%s succeeded\n", pod.Namespace, pod.Name)
}

func validateTargetedProbeOutput(f *framework.Framework, pod *v1.Pod, fileNames []string, value string) {
	ginkgo.By("submitting the pod to kubernetes")
	podClient := f.ClientSet.CoreV1().Pods(f.Namespace.Name)
	defer func() {
		ginkgo.By("deleting the pod")
		defer ginkgo.GinkgoRecover()
		podClient.Delete(pod.Name, metav1.NewDeleteOptions(0))
	}()
	if _, err := podClient.Create(pod); err != nil {
		framework.Failf("ginkgo.Failed to create pod %s/%s: %v", pod.Namespace, pod.Name, err)
	}

	framework.ExpectNoError(f.WaitForPodRunning(pod.Name))

	ginkgo.By("retrieving the pod")
	pod, err := podClient.Get(pod.Name, metav1.GetOptions{})
	if err != nil {
		framework.Failf("ginkgo.Failed to get pod %s/%s: %v", pod.Namespace, pod.Name, err)
	}
	// Try to find the expected value for each expected name.
	ginkgo.By("looking for the results for each expected name from probers")
	assertFilesContain(fileNames, "results", pod, f.ClientSet, true, value)

	framework.Logf("DNS probes using %s succeeded\n", pod.Name)
}

func generateDNSUtilsPod() *v1.Pod {
	return &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind: "Pod",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "e2e-dns-utils-",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    "util",
					Image:   imageutils.GetE2EImage(imageutils.Dnsutils),
					Command: []string{"sleep", "10000"},
				},
			},
		},
	}
}
