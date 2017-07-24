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
	"fmt"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"

	. "github.com/onsi/ginkgo"
)

type dnsFederationsConfigMapTest struct {
	dnsTestCommon

	fedMap  map[string]string
	isValid bool
}

var _ = SIGDescribe("DNS configMap federations", func() {
	t := &dnsNameserverTest{dnsTestCommon: newDnsTestCommon()}
	BeforeEach(func() { t.c = t.f.ClientSet })

	It("should be able to change federation configuration [Slow][Serial]", func() {
		t.run()
	})
})

func (t *dnsFederationsConfigMapTest) run() {
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

func (t *dnsFederationsConfigMapTest) validate() {
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

func (t *dnsFederationsConfigMapTest) setConfigMap(cm *v1.ConfigMap, fedMap map[string]string, isValid bool) {
	if isValid {
		t.fedMap = fedMap
	}
	t.isValid = isValid
	t.dnsTestCommon.setConfigMap(cm)
}

func (t *dnsFederationsConfigMapTest) deleteConfigMap() {
	t.isValid = false
	t.dnsTestCommon.deleteConfigMap()
}

type dnsNameserverTest struct {
	dnsTestCommon
}

func (t *dnsNameserverTest) run() {
	t.init()

	t.createUtilPod()
	defer t.deleteUtilPod()

	t.createDNSServer(map[string]string{
		"abc.acme.local": "1.1.1.1",
		"def.acme.local": "2.2.2.2",
		"widget.local":   "3.3.3.3",
	})
	defer t.deleteDNSServerPod()

	t.setConfigMap(&v1.ConfigMap{Data: map[string]string{
		"stubDomains":         fmt.Sprintf(`{"acme.local":["%v"]}`, t.dnsServerPod.Status.PodIP),
		"upstreamNameservers": fmt.Sprintf(`["%v"]`, t.dnsServerPod.Status.PodIP),
	}})

	// The ConfigMap update mechanism takes longer than the standard
	// wait.ForeverTestTimeout.
	moreForeverTestTimeout := 2 * 60 * time.Second

	t.checkDNSRecordFrom(
		"abc.acme.local",
		func(actual []string) bool { return len(actual) == 1 && actual[0] == "1.1.1.1" },
		"dnsmasq",
		moreForeverTestTimeout)
	t.checkDNSRecordFrom(
		"def.acme.local",
		func(actual []string) bool { return len(actual) == 1 && actual[0] == "2.2.2.2" },
		"dnsmasq",
		moreForeverTestTimeout)
	t.checkDNSRecordFrom(
		"widget.local",
		func(actual []string) bool { return len(actual) == 1 && actual[0] == "3.3.3.3" },
		"dnsmasq",
		moreForeverTestTimeout)

	t.c.Core().ConfigMaps(t.ns).Delete(t.name, nil)
	// Wait for the deleted ConfigMap to take effect, otherwise the
	// configuration can bleed into other tests.
	t.checkDNSRecordFrom(
		"abc.acme.local",
		func(actual []string) bool { return len(actual) == 0 },
		"dnsmasq",
		moreForeverTestTimeout)
}

var _ = SIGDescribe("DNS configMap nameserver", func() {
	t := &dnsNameserverTest{dnsTestCommon: newDnsTestCommon()}
	BeforeEach(func() { t.c = t.f.ClientSet })

	It("should be able to change stubDomain configuration [Slow][Serial]", func() {
		t.run()
	})
})
