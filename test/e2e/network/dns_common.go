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
	"errors"
	"fmt"
	"regexp"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	dnsclient "k8s.io/kubernetes/third_party/forked/golang/net"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

// Windows output can contain additional \r
var newLineRegexp = regexp.MustCompile("\r?\n")

type dnsTestCommon struct {
	f    *framework.Framework
	c    clientset.Interface
	ns   string
	name string

	dnsPod       *v1.Pod
	utilPod      *v1.Pod
	utilService  *v1.Service
	dnsServerPod *v1.Pod

	cm *v1.ConfigMap
}

func newDNSTestCommon() dnsTestCommon {
	framework := framework.NewDefaultFramework("dns-config-map")
	framework.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	return dnsTestCommon{
		f:  framework,
		ns: "kube-system",
	}
}

func (t *dnsTestCommon) init(ctx context.Context) {
	ginkgo.By("Finding a DNS pod")
	label := labels.SelectorFromSet(labels.Set(map[string]string{"k8s-app": "kube-dns"}))
	options := metav1.ListOptions{LabelSelector: label.String()}

	namespace := "kube-system"
	pods, err := t.f.ClientSet.CoreV1().Pods(namespace).List(ctx, options)
	framework.ExpectNoError(err, "failed to list pods in namespace: %s", namespace)
	gomega.Expect(pods.Items).ToNot(gomega.BeEmpty())

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
	cmd := []string{"dig", "+short"}
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
		panic(errors.New("invalid target: " + target))
	}
	cmd = append(cmd, dnsName)

	stdout, stderr, err := e2epod.ExecWithOptions(t.f, e2epod.ExecOptions{
		Command:       cmd,
		Namespace:     t.f.Namespace.Name,
		PodName:       t.utilPod.Name,
		ContainerName: t.utilPod.Spec.Containers[0].Name,
		CaptureStdout: true,
		CaptureStderr: true,
	})

	framework.Logf("Running dig: %v, stdout: %q, stderr: %q, err: %v",
		cmd, stdout, stderr, err)

	if stdout == "" {
		return []string{}
	}
	return newLineRegexp.Split(stdout, -1)
}

func (t *dnsTestCommon) setConfigMap(ctx context.Context, cm *v1.ConfigMap) {
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
	cmList, err := t.c.CoreV1().ConfigMaps(t.ns).List(ctx, options)
	framework.ExpectNoError(err, "failed to list ConfigMaps in namespace: %s", t.ns)

	if len(cmList.Items) == 0 {
		ginkgo.By(fmt.Sprintf("Creating the ConfigMap (%s:%s) %+v", t.ns, t.name, *cm))
		_, err := t.c.CoreV1().ConfigMaps(t.ns).Create(ctx, cm, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create ConfigMap (%s:%s) %+v", t.ns, t.name, *cm)
	} else {
		ginkgo.By(fmt.Sprintf("Updating the ConfigMap (%s:%s) to %+v", t.ns, t.name, *cm))
		_, err := t.c.CoreV1().ConfigMaps(t.ns).Update(ctx, cm, metav1.UpdateOptions{})
		framework.ExpectNoError(err, "failed to update ConfigMap (%s:%s) to %+v", t.ns, t.name, *cm)
	}
}

func (t *dnsTestCommon) fetchDNSConfigMapData(ctx context.Context) map[string]string {
	if t.name == "coredns" {
		pcm, err := t.c.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(ctx, t.name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get DNS ConfigMap: %s", t.name)
		return pcm.Data
	}
	return nil
}

func (t *dnsTestCommon) restoreDNSConfigMap(ctx context.Context, configMapData map[string]string) {
	if t.name == "coredns" {
		t.setConfigMap(ctx, &v1.ConfigMap{Data: configMapData})
		t.deleteCoreDNSPods(ctx)
	} else {
		err := t.c.CoreV1().ConfigMaps(t.ns).Delete(ctx, t.name, metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			framework.Failf("Unexpected error deleting configmap %s/%s", t.ns, t.name)
		}
	}
}

func (t *dnsTestCommon) createUtilPodLabel(ctx context.Context, baseName string) {
	// Actual port # doesn't matter, just needs to exist.
	const servicePort = 10101
	podName := fmt.Sprintf("%s-%s", baseName, string(uuid.NewUUID()))
	ports := []v1.ContainerPort{{ContainerPort: servicePort, Protocol: v1.ProtocolTCP}}
	t.utilPod = e2epod.NewAgnhostPod(t.f.Namespace.Name, podName, nil, nil, ports)

	var err error
	t.utilPod, err = t.c.CoreV1().Pods(t.f.Namespace.Name).Create(ctx, t.utilPod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "failed to create pod: %v", t.utilPod)
	framework.Logf("Created pod %v", t.utilPod)
	err = e2epod.WaitForPodNameRunningInNamespace(ctx, t.f.ClientSet, t.utilPod.Name, t.f.Namespace.Name)
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
					TargetPort: intstr.FromInt32(servicePort),
				},
			},
		},
	}

	t.utilService, err = t.c.CoreV1().Services(t.f.Namespace.Name).Create(ctx, t.utilService, metav1.CreateOptions{})
	framework.ExpectNoError(err, "failed to create service: %s/%s", t.f.Namespace.Name, t.utilService.ObjectMeta.Name)
	framework.Logf("Created service %v", t.utilService)
}

func (t *dnsTestCommon) deleteUtilPod(ctx context.Context) {
	podClient := t.c.CoreV1().Pods(t.f.Namespace.Name)
	if err := podClient.Delete(ctx, t.utilPod.Name, *metav1.NewDeleteOptions(0)); err != nil {
		framework.Logf("Delete of pod %v/%v failed: %v",
			t.utilPod.Namespace, t.utilPod.Name, err)
	}
}

// deleteCoreDNSPods manually deletes the CoreDNS pods to apply the changes to the ConfigMap.
func (t *dnsTestCommon) deleteCoreDNSPods(ctx context.Context) {

	label := labels.SelectorFromSet(labels.Set(map[string]string{"k8s-app": "kube-dns"}))
	options := metav1.ListOptions{LabelSelector: label.String()}

	pods, err := t.f.ClientSet.CoreV1().Pods("kube-system").List(ctx, options)
	framework.ExpectNoError(err, "failed to list pods of kube-system with label %q", label.String())
	podClient := t.c.CoreV1().Pods(metav1.NamespaceSystem)

	for _, pod := range pods.Items {
		err = podClient.Delete(ctx, pod.Name, *metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err, "failed to delete pod: %s", pod.Name)
	}
}

func generateCoreDNSServerPod(corednsConfig *v1.ConfigMap) *v1.Pod {
	podName := fmt.Sprintf("e2e-configmap-dns-server-%s", string(uuid.NewUUID()))
	volumes := []v1.Volume{
		{
			Name: "coredns-config",
			VolumeSource: v1.VolumeSource{
				ConfigMap: &v1.ConfigMapVolumeSource{
					LocalObjectReference: v1.LocalObjectReference{
						Name: corednsConfig.Name,
					},
				},
			},
		},
	}
	mounts := []v1.VolumeMount{
		{
			Name:      "coredns-config",
			MountPath: "/etc/coredns",
			ReadOnly:  true,
		},
	}

	pod := e2epod.NewAgnhostPod("", podName, volumes, mounts, nil, "-conf", "/etc/coredns/Corefile")
	pod.Spec.Containers[0].Command = []string{"/coredns"}
	pod.Spec.DNSPolicy = "Default"
	return pod
}

func generateCoreDNSConfigmap(namespaceName string, aRecords map[string]string) *v1.ConfigMap {
	entries := ""
	for name, ip := range aRecords {
		entries += fmt.Sprintf("\n\t\t%v %v", ip, name)
	}

	corefileData := fmt.Sprintf(`. {
	hosts {%s
	}
	log
}`, entries)

	return &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:    namespaceName,
			GenerateName: "e2e-coredns-configmap-",
		},
		Data: map[string]string{
			"Corefile": corefileData,
		},
	}
}

func (t *dnsTestCommon) createDNSPodFromObj(ctx context.Context, pod *v1.Pod) {
	t.dnsServerPod = pod

	var err error
	t.dnsServerPod, err = t.c.CoreV1().Pods(t.f.Namespace.Name).Create(ctx, t.dnsServerPod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "failed to create pod: %v", t.dnsServerPod)
	framework.Logf("Created pod %v", t.dnsServerPod)
	err = e2epod.WaitForPodNameRunningInNamespace(ctx, t.f.ClientSet, t.dnsServerPod.Name, t.f.Namespace.Name)
	framework.ExpectNoError(err, "pod failed to start running: %v", t.dnsServerPod)

	t.dnsServerPod, err = t.c.CoreV1().Pods(t.f.Namespace.Name).Get(ctx, t.dnsServerPod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "failed to get pod: %s", t.dnsServerPod.Name)
}

func (t *dnsTestCommon) createDNSServer(ctx context.Context, namespace string, aRecords map[string]string) {
	corednsConfig := generateCoreDNSConfigmap(namespace, aRecords)
	corednsConfig, err := t.c.CoreV1().ConfigMaps(namespace).Create(ctx, corednsConfig, metav1.CreateOptions{})
	if err != nil {
		framework.Failf("unable to create test configMap %s: %v", corednsConfig.Name, err)
	}

	t.createDNSPodFromObj(ctx, generateCoreDNSServerPod(corednsConfig))
}

func (t *dnsTestCommon) createDNSServerWithPtrRecord(ctx context.Context, namespace string, isIPv6 bool) {
	// NOTE: PTR records are generated automatically by CoreDNS. So, if we're creating A records, we're
	// going to also have PTR records. See: https://coredns.io/plugins/hosts/
	var aRecords map[string]string
	if isIPv6 {
		aRecords = map[string]string{"my.test": "2001:db8::29"}
	} else {
		aRecords = map[string]string{"my.test": "192.0.2.123"}
	}
	t.createDNSServer(ctx, namespace, aRecords)
}

func (t *dnsTestCommon) deleteDNSServerPod(ctx context.Context) {
	podClient := t.c.CoreV1().Pods(t.f.Namespace.Name)
	if err := podClient.Delete(ctx, t.dnsServerPod.Name, *metav1.NewDeleteOptions(0)); err != nil {
		framework.Logf("Delete of pod %v/%v failed: %v",
			t.utilPod.Namespace, t.dnsServerPod.Name, err)
	}
}

type dnsQuerier struct {
	name  string             // container name
	image imageutils.ImageID // container image
	cmd   string             // a shell-script in a string
}

func createDNSPod(namespace string, probers []dnsQuerier, podHostName, serviceName string) *v1.Pod {
	podName := "dns-test-" + string(uuid.NewUUID())
	volumes := []v1.Volume{
		{
			Name: "results",
			VolumeSource: v1.VolumeSource{
				EmptyDir: &v1.EmptyDirVolumeSource{},
			},
		},
	}
	mounts := []v1.VolumeMount{
		{
			Name:      "results",
			MountPath: "/results",
		},
	}

	// This is an "agnhost pod" but we use the 0th container as a webserver.
	// TODO: Consider scraping logs instead of running a webserver.
	dnsPod := e2epod.NewAgnhostPod(namespace, podName, volumes, mounts, nil, "test-webserver")
	dnsPod.Spec.Containers[0].Name = "webserver"

	probeCtrs := []v1.Container{}
	for _, probe := range probers {
		name := probe.name + "-querier"
		if probe.image == imageutils.Agnhost {
			// agnhost is special cased, to keep all of its logic consistent.
			ctr := e2epod.NewAgnhostContainer(name, mounts, nil)
			ctr.Command = []string{"sh", "-c", probe.cmd}
			probeCtrs = append(probeCtrs, ctr)
		} else {
			ctr := v1.Container{
				Name:         name,
				Image:        imageutils.GetE2EImage(probe.image),
				Command:      []string{"sh", "-c", probe.cmd},
				VolumeMounts: mounts,
			}
			probeCtrs = append(probeCtrs, ctr)
		}
	}

	dnsPod.Spec.Containers = append(dnsPod.Spec.Containers, probeCtrs...)
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

	hostEntryCmd := `test -n "$$(getent hosts %s)" && echo OK > /results/%s;`
	if framework.NodeOSDistroIs("windows") {
		// We don't have getent on Windows, but we can still check the hosts file.
		hostEntryCmd = `test -n "$$(grep '%s' C:/Windows/System32/drivers/etc/hosts)" && echo OK > /results/%s;`
	}
	for _, name := range hostEntries {
		fileName := fmt.Sprintf("%s_hosts@%s", fileNamePrefix, name)
		fileNames = append(fileNames, fileName)
		probeCmd += fmt.Sprintf(hostEntryCmd, name, fileName)
	}

	if len(ptrLookupIP) > 0 {
		ptrLookup, err := dnsclient.Reverseaddr(ptrLookupIP)
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

func assertFilesExist(ctx context.Context, fileNames []string, fileDir string, pod *v1.Pod, client clientset.Interface) {
	assertFilesContain(ctx, fileNames, fileDir, pod, client, false, "")
}

func assertFilesContain(ctx context.Context, fileNames []string, fileDir string, pod *v1.Pod, client clientset.Interface, check bool, expected string) {
	var failed []string

	framework.ExpectNoError(wait.PollUntilContextTimeout(ctx, time.Second*5, time.Second*600, true, func(ctx context.Context) (bool, error) {
		failed = []string{}

		ctx, cancel := context.WithTimeout(ctx, framework.SingleCallTimeout)
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
					return false, fmt.Errorf("Unable to read %s from pod %s/%s: %v", fileName, pod.Namespace, pod.Name, err)
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

		// grab logs from all the containers
		for _, container := range pod.Spec.Containers {
			logs, err := e2epod.GetPodLogs(ctx, client, pod.Namespace, pod.Name, container.Name)
			if err != nil {
				return false, fmt.Errorf("unexpected error getting pod client logs for %s: %v", container.Name, err)
			}
			framework.Logf("Pod client logs for %s: %s", container.Name, logs)
		}

		return false, nil
	}))
	gomega.Expect(failed).To(gomega.BeEmpty())
}

func validateDNSResults(ctx context.Context, f *framework.Framework, pod *v1.Pod, fileNames []string) {
	ginkgo.By("submitting the pod to kubernetes")
	podClient := f.ClientSet.CoreV1().Pods(f.Namespace.Name)
	ginkgo.DeferCleanup(func(ctx context.Context) error {
		ginkgo.By("deleting the pod")
		return podClient.Delete(ctx, pod.Name, *metav1.NewDeleteOptions(0))
	})
	if _, err := podClient.Create(ctx, pod, metav1.CreateOptions{}); err != nil {
		framework.Failf("ginkgo.Failed to create pod %s/%s: %v", pod.Namespace, pod.Name, err)
	}

	framework.ExpectNoError(e2epod.WaitForPodRunningInNamespaceSlow(ctx, f.ClientSet, pod.Name, f.Namespace.Name))

	ginkgo.By("retrieving the pod")
	pod, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
	if err != nil {
		framework.Failf("ginkgo.Failed to get pod %s/%s: %v", pod.Namespace, pod.Name, err)
	}
	// Try to find results for each expected name.
	ginkgo.By("looking for the results for each expected name from probers")
	assertFilesExist(ctx, fileNames, "results", pod, f.ClientSet)

	// TODO: probe from the host, too.

	framework.Logf("DNS probes using %s/%s succeeded\n", pod.Namespace, pod.Name)
}

func validateTargetedProbeOutput(ctx context.Context, f *framework.Framework, pod *v1.Pod, fileNames []string, value string) {
	ginkgo.By("submitting the pod to kubernetes")
	podClient := f.ClientSet.CoreV1().Pods(f.Namespace.Name)
	ginkgo.DeferCleanup(func(ctx context.Context) error {
		ginkgo.By("deleting the pod")
		return podClient.Delete(ctx, pod.Name, *metav1.NewDeleteOptions(0))
	})
	if _, err := podClient.Create(ctx, pod, metav1.CreateOptions{}); err != nil {
		framework.Failf("ginkgo.Failed to create pod %s/%s: %v", pod.Namespace, pod.Name, err)
	}

	framework.ExpectNoError(e2epod.WaitForPodRunningInNamespaceSlow(ctx, f.ClientSet, pod.Name, f.Namespace.Name))

	ginkgo.By("retrieving the pod")
	pod, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
	if err != nil {
		framework.Failf("ginkgo.Failed to get pod %s/%s: %v", pod.Namespace, pod.Name, err)
	}
	// Try to find the expected value for each expected name.
	ginkgo.By("looking for the results for each expected name from probers")
	assertFilesContain(ctx, fileNames, "results", pod, f.ClientSet, true, value)

	framework.Logf("DNS probes using %s succeeded\n", pod.Name)
}
