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

package e2e

import (
	"fmt"
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
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

func newDnsTestCommon() dnsTestCommon {
	return dnsTestCommon{
		f:    framework.NewDefaultFramework("dns-config-map"),
		ns:   "kube-system",
		name: "kube-dns",
	}
}

func (t *dnsTestCommon) init() {
	By("Finding a DNS pod")
	label := labels.SelectorFromSet(labels.Set(map[string]string{"k8s-app": "kube-dns"}))
	options := metav1.ListOptions{LabelSelector: label.String()}

	pods, err := t.f.ClientSet.Core().Pods("kube-system").List(options)
	Expect(err).NotTo(HaveOccurred())
	Expect(len(pods.Items)).Should(BeNumerically(">=", 1))

	t.dnsPod = &pods.Items[0]
	framework.Logf("Using DNS pod: %v", t.dnsPod.Name)
}

func (t *dnsTestCommon) checkDNSRecord(name string, predicate func([]string) bool, timeout time.Duration) {
	t.checkDNSRecordFrom(name, predicate, "kube-dns", timeout)
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
	case "kube-dns":
		cmd = append(cmd, "@"+t.dnsPod.Status.PodIP, "-p", "10053")
	case "dnsmasq":
		break
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
	} else {
		return strings.Split(stdout, "\n")
	}
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
	cmList, err := t.c.Core().ConfigMaps(t.ns).List(options)
	Expect(err).NotTo(HaveOccurred())

	if len(cmList.Items) == 0 {
		By(fmt.Sprintf("Creating the ConfigMap (%s:%s) %+v", t.ns, t.name, *cm))
		_, err := t.c.Core().ConfigMaps(t.ns).Create(cm)
		Expect(err).NotTo(HaveOccurred())
	} else {
		By(fmt.Sprintf("Updating the ConfigMap (%s:%s) to %+v", t.ns, t.name, *cm))
		_, err := t.c.Core().ConfigMaps(t.ns).Update(cm)
		Expect(err).NotTo(HaveOccurred())
	}
}

func (t *dnsTestCommon) deleteConfigMap() {
	By(fmt.Sprintf("Deleting the ConfigMap (%s:%s)", t.ns, t.name))
	t.cm = nil
	err := t.c.Core().ConfigMaps(t.ns).Delete(t.name, nil)
	Expect(err).NotTo(HaveOccurred())
}

func (t *dnsTestCommon) createUtilPod() {
	// Actual port # doesn't matter, just needs to exist.
	const servicePort = 10101

	t.utilPod = &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind: "Pod",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace:    t.f.Namespace.Name,
			Labels:       map[string]string{"app": "e2e-dns-configmap"},
			GenerateName: "e2e-dns-configmap-",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    "util",
					Image:   "gcr.io/google_containers/dnsutils:e2e",
					Command: []string{"sleep", "10000"},
					Ports: []v1.ContainerPort{
						{ContainerPort: servicePort, Protocol: "TCP"},
					},
				},
			},
		},
	}

	var err error
	t.utilPod, err = t.c.Core().Pods(t.f.Namespace.Name).Create(t.utilPod)
	Expect(err).NotTo(HaveOccurred())
	framework.Logf("Created pod %v", t.utilPod)
	Expect(t.f.WaitForPodRunning(t.utilPod.Name)).NotTo(HaveOccurred())

	t.utilService = &v1.Service{
		TypeMeta: metav1.TypeMeta{
			Kind: "Service",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: t.f.Namespace.Name,
			Name:      "e2e-dns-configmap",
		},
		Spec: v1.ServiceSpec{
			Selector: map[string]string{"app": "e2e-dns-configmap"},
			Ports: []v1.ServicePort{
				{
					Protocol:   "TCP",
					Port:       servicePort,
					TargetPort: intstr.FromInt(servicePort),
				},
			},
		},
	}

	t.utilService, err = t.c.Core().Services(t.f.Namespace.Name).Create(t.utilService)
	Expect(err).NotTo(HaveOccurred())
	framework.Logf("Created service %v", t.utilService)
}

func (t *dnsTestCommon) deleteUtilPod() {
	podClient := t.c.Core().Pods(t.f.Namespace.Name)
	if err := podClient.Delete(t.utilPod.Name, metav1.NewDeleteOptions(0)); err != nil {
		framework.Logf("Delete of pod %v:%v failed: %v",
			t.utilPod.Namespace, t.utilPod.Name, err)
	}
}

func (t *dnsTestCommon) createDNSServer(aRecords map[string]string) {
	t.dnsServerPod = &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind: "Pod",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace:    t.f.Namespace.Name,
			GenerateName: "e2e-dns-configmap-dns-server-",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "dns",
					Image: "gcr.io/google_containers/k8s-dns-dnsmasq-amd64:1.13.0",
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
		t.dnsServerPod.Spec.Containers[0].Command = append(
			t.dnsServerPod.Spec.Containers[0].Command,
			fmt.Sprintf("-A/%v/%v", name, ip))
	}

	var err error
	t.dnsServerPod, err = t.c.Core().Pods(t.f.Namespace.Name).Create(t.dnsServerPod)
	Expect(err).NotTo(HaveOccurred())
	framework.Logf("Created pod %v", t.dnsServerPod)
	Expect(t.f.WaitForPodRunning(t.dnsServerPod.Name)).NotTo(HaveOccurred())

	t.dnsServerPod, err = t.c.Core().Pods(t.f.Namespace.Name).Get(
		t.dnsServerPod.Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
}

func (t *dnsTestCommon) deleteDNSServerPod() {
	podClient := t.c.Core().Pods(t.f.Namespace.Name)
	if err := podClient.Delete(t.dnsServerPod.Name, metav1.NewDeleteOptions(0)); err != nil {
		framework.Logf("Delete of pod %v:%v failed: %v",
			t.utilPod.Namespace, t.dnsServerPod.Name, err)
	}
}
