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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

type dnsConfigMapTest struct {
	f      *framework.Framework
	c      clientset.Interface
	ns     string
	name   string
	labels []string

	cm      *v1.ConfigMap
	fedMap  map[string]string
	isValid bool

	dnsPod      *v1.Pod
	utilPod     *v1.Pod
	utilService *v1.Service
}

var _ = framework.KubeDescribe("DNS config map", func() {
	test := &dnsConfigMapTest{
		f:    framework.NewDefaultFramework("dns-config-map"),
		ns:   "kube-system",
		name: "kube-dns",
	}

	BeforeEach(func() {
		test.c = test.f.ClientSet
	})

	It("should be able to change configuration", func() {
		test.run()
	})
})

func (t *dnsConfigMapTest) init() {
	By("Finding a DNS pod")
	label := labels.SelectorFromSet(labels.Set(map[string]string{"k8s-app": "kube-dns"}))
	options := v1.ListOptions{LabelSelector: label.String()}

	pods, err := t.f.ClientSet.Core().Pods("kube-system").List(options)
	Expect(err).NotTo(HaveOccurred())
	Expect(len(pods.Items)).Should(BeNumerically(">=", 1))

	t.dnsPod = &pods.Items[0]
	framework.Logf("Using DNS pod: %v", t.dnsPod.Name)
}

func (t *dnsConfigMapTest) run() {
	t.init()

	defer t.c.Core().ConfigMaps(t.ns).Delete(t.name, nil)
	t.createUtilPod()
	defer t.deleteUtilPod()

	t.validate()

	t.labels = []string{"abc", "ghi"}
	valid1 := map[string]string{"federations": t.labels[0] + "=def"}
	valid1m := map[string]string{t.labels[0]: "def"}
	valid2 := map[string]string{"federations": t.labels[1] + "=xyz"}
	valid2m := map[string]string{t.labels[1]: "xyz"}
	invalid := map[string]string{"federations": "invalid.map=xyz"}

	By("empty -> valid1")
	t.setConfigMap(&v1.ConfigMap{Data: valid1}, valid1m, true)
	t.validate()

	By("valid1 -> valid2")
	t.setConfigMap(&v1.ConfigMap{Data: valid2}, valid2m, true)
	t.validate()

	By("valid2 -> invalid")
	t.setConfigMap(&v1.ConfigMap{Data: invalid}, nil, false)
	t.validate()

	By("invalid -> valid1")
	t.setConfigMap(&v1.ConfigMap{Data: valid1}, valid1m, true)
	t.validate()

	By("valid1 -> deleted")
	t.deleteConfigMap()
	t.validate()

	By("deleted -> invalid")
	t.setConfigMap(&v1.ConfigMap{Data: invalid}, nil, false)
	t.validate()
}

func (t *dnsConfigMapTest) validate() {
	t.validateFederation()
}

func (t *dnsConfigMapTest) validateFederation() {
	federations := t.fedMap

	if len(federations) == 0 {
		By(fmt.Sprintf("Validating federation labels %v do not exist", t.labels))

		for _, label := range t.labels {
			var federationDNS = fmt.Sprintf("e2e-dns-configmap.%s.%s.svc.cluster.local.",
				t.f.Namespace.Name, label)
			predicate := func(actual []string) bool {
				return len(actual) == 0
			}
			t.checkDNSRecord(federationDNS, predicate, wait.ForeverTestTimeout)
		}
	} else {
		for label := range federations {
			var federationDNS = fmt.Sprintf("%s.%s.%s.svc.cluster.local.",
				t.utilService.ObjectMeta.Name, t.f.Namespace.Name, label)
			var localDNS = fmt.Sprintf("%s.%s.svc.cluster.local.",
				t.utilService.ObjectMeta.Name, t.f.Namespace.Name)
			// Check local mapping. Checking a remote mapping requires
			// creating an arbitrary DNS record which is not possible at the
			// moment.
			By(fmt.Sprintf("Validating federation record %v", label))
			predicate := func(actual []string) bool {
				for _, v := range actual {
					if v == localDNS {
						return true
					}
				}
				return false
			}
			t.checkDNSRecord(federationDNS, predicate, wait.ForeverTestTimeout)
		}
	}
}

func (t *dnsConfigMapTest) checkDNSRecord(name string, predicate func([]string) bool, timeout time.Duration) {
	var actual []string

	err := wait.PollImmediate(
		time.Duration(1)*time.Second,
		timeout,
		func() (bool, error) {
			actual = t.runDig(name)
			if predicate(actual) {
				return true, nil
			}
			return false, nil
		})

	if err != nil {
		framework.Logf("dig result did not match: %#v after %v",
			actual, timeout)
	}
}

// runDig querying for `dnsName`. Returns a list of responses.
func (t *dnsConfigMapTest) runDig(dnsName string) []string {
	cmd := []string{
		"/usr/bin/dig",
		"+short",
		"@" + t.dnsPod.Status.PodIP,
		"-p", "10053", dnsName,
	}
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

func (t *dnsConfigMapTest) setConfigMap(cm *v1.ConfigMap, fedMap map[string]string, isValid bool) {
	if isValid {
		t.cm = cm
		t.fedMap = fedMap
	}
	t.isValid = isValid

	cm.ObjectMeta.Namespace = t.ns
	cm.ObjectMeta.Name = t.name

	options := v1.ListOptions{
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

func (t *dnsConfigMapTest) deleteConfigMap() {
	By(fmt.Sprintf("Deleting the ConfigMap (%s:%s)", t.ns, t.name))

	t.cm = nil
	t.isValid = false

	err := t.c.Core().ConfigMaps(t.ns).Delete(t.name, nil)
	Expect(err).NotTo(HaveOccurred())
}

func (t *dnsConfigMapTest) createUtilPod() {
	// Actual port # doesn't matter, just need to exist.
	const servicePort = 10101

	t.utilPod = &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind: "Pod",
		},
		ObjectMeta: v1.ObjectMeta{
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
		ObjectMeta: v1.ObjectMeta{
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

func (t *dnsConfigMapTest) deleteUtilPod() {
	podClient := t.c.Core().Pods(t.f.Namespace.Name)
	if err := podClient.Delete(t.utilPod.Name, v1.NewDeleteOptions(0)); err != nil {
		framework.Logf("Delete of pod %v:%v failed: %v",
			t.utilPod.Namespace, t.utilPod.Name, err)
	}
}
