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
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"

	"github.com/onsi/ginkgo"
)

type dnsFederationsConfigMapTest struct {
	dnsTestCommon

	fedMap  map[string]string
	isValid bool
}

var (
	googleDNSHostname = "dns.google"
	// The ConfigMap update mechanism takes longer than the standard
	// wait.ForeverTestTimeout.
	moreForeverTestTimeout = 2 * 60 * time.Second
)

var _ = SIGDescribe("DNS configMap federations [Feature:Federation]", func() {

	t := &dnsFederationsConfigMapTest{dnsTestCommon: newDNSTestCommon()}

	ginkgo.It("should be able to change federation configuration [Slow][Serial]", func() {
		t.c = t.f.ClientSet
		t.run()
	})
})

func (t *dnsFederationsConfigMapTest) run() {
	t.init()

	defer t.c.CoreV1().ConfigMaps(t.ns).Delete(context.TODO(), t.name, metav1.DeleteOptions{})
	t.createUtilPodLabel("e2e-dns-configmap")
	defer t.deleteUtilPod()
	originalConfigMapData := t.fetchDNSConfigMapData()
	defer t.restoreDNSConfigMap(originalConfigMapData)

	t.validate(framework.TestContext.ClusterDNSDomain)

	if t.name == "coredns" {
		t.labels = []string{"abc", "ghi"}
		valid1 := map[string]string{
			"Corefile": fmt.Sprintf(`.:53 {
        health
        ready
        kubernetes %v in-addr.arpa ip6.arpa {
            pods insecure
            upstream
            fallthrough in-addr.arpa ip6.arpa
            ttl 30
        }
        federation %v {
           abc def.com
        }
        forward . /etc/resolv.conf
    }`, framework.TestContext.ClusterDNSDomain, framework.TestContext.ClusterDNSDomain)}
		valid1m := map[string]string{t.labels[0]: "def.com"}

		valid2 := map[string]string{
			"Corefile": fmt.Sprintf(`:53 {
        health
        ready
        kubernetes %v in-addr.arpa ip6.arpa {
            pods insecure
            upstream
            fallthrough in-addr.arpa ip6.arpa
            ttl 30
        }
        federation %v {
           ghi xyz.com
        }
        forward . /etc/resolv.conf
    }`, framework.TestContext.ClusterDNSDomain, framework.TestContext.ClusterDNSDomain)}
		valid2m := map[string]string{t.labels[1]: "xyz.com"}

		ginkgo.By("default -> valid1")
		t.setConfigMap(&v1.ConfigMap{Data: valid1}, valid1m, true)
		t.deleteCoreDNSPods()
		t.validate(framework.TestContext.ClusterDNSDomain)

		ginkgo.By("valid1 -> valid2")
		t.setConfigMap(&v1.ConfigMap{Data: valid2}, valid2m, true)
		t.deleteCoreDNSPods()
		t.validate(framework.TestContext.ClusterDNSDomain)

		ginkgo.By("valid2 -> default")
		t.setConfigMap(&v1.ConfigMap{Data: originalConfigMapData}, nil, false)
		t.deleteCoreDNSPods()
		t.validate(framework.TestContext.ClusterDNSDomain)

		t.restoreDNSConfigMap(originalConfigMapData)

	} else {
		t.labels = []string{"abc", "ghi"}
		valid1 := map[string]string{"federations": t.labels[0] + "=def"}
		valid1m := map[string]string{t.labels[0]: "def"}
		valid2 := map[string]string{"federations": t.labels[1] + "=xyz"}
		valid2m := map[string]string{t.labels[1]: "xyz"}
		invalid := map[string]string{"federations": "invalid.map=xyz"}

		ginkgo.By("empty -> valid1")
		t.setConfigMap(&v1.ConfigMap{Data: valid1}, valid1m, true)
		t.validate(framework.TestContext.ClusterDNSDomain)

		ginkgo.By("valid1 -> valid2")
		t.setConfigMap(&v1.ConfigMap{Data: valid2}, valid2m, true)
		t.validate(framework.TestContext.ClusterDNSDomain)

		ginkgo.By("valid2 -> invalid")
		t.setConfigMap(&v1.ConfigMap{Data: invalid}, nil, false)
		t.validate(framework.TestContext.ClusterDNSDomain)

		ginkgo.By("invalid -> valid1")
		t.setConfigMap(&v1.ConfigMap{Data: valid1}, valid1m, true)
		t.validate(framework.TestContext.ClusterDNSDomain)

		ginkgo.By("valid1 -> deleted")
		t.deleteConfigMap()
		t.validate(framework.TestContext.ClusterDNSDomain)

		ginkgo.By("deleted -> invalid")
		t.setConfigMap(&v1.ConfigMap{Data: invalid}, nil, false)
		t.validate(framework.TestContext.ClusterDNSDomain)
	}
}

func (t *dnsFederationsConfigMapTest) validate(dnsDomain string) {
	federations := t.fedMap

	if len(federations) == 0 {
		ginkgo.By(fmt.Sprintf("Validating federation labels %v do not exist", t.labels))

		for _, label := range t.labels {
			var federationDNS = fmt.Sprintf("e2e-dns-configmap.%s.%s.svc.%s.",
				t.f.Namespace.Name, label, framework.TestContext.ClusterDNSDomain)
			predicate := func(actual []string) bool {
				return len(actual) == 0
			}
			t.checkDNSRecordFrom(federationDNS, predicate, "cluster-dns", wait.ForeverTestTimeout)
		}
	} else {
		for label := range federations {
			var federationDNS = fmt.Sprintf("%s.%s.%s.svc.%s.",
				t.utilService.ObjectMeta.Name, t.f.Namespace.Name, label, framework.TestContext.ClusterDNSDomain)
			var localDNS = fmt.Sprintf("%s.%s.svc.%s.",
				t.utilService.ObjectMeta.Name, t.f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
			if t.name == "coredns" {
				localDNS = t.utilService.Spec.ClusterIP
			}
			// Check local mapping. Checking a remote mapping requires
			// creating an arbitrary DNS record which is not possible at the
			// moment.
			ginkgo.By(fmt.Sprintf("Validating federation record %v", label))
			predicate := func(actual []string) bool {
				for _, v := range actual {
					if v == localDNS {
						return true
					}
				}
				return false
			}
			t.checkDNSRecordFrom(federationDNS, predicate, "cluster-dns", wait.ForeverTestTimeout)
		}
	}
}

func (t *dnsFederationsConfigMapTest) setConfigMap(cm *v1.ConfigMap, fedMap map[string]string, isValid bool) {
	t.fedMap = nil

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

func (t *dnsNameserverTest) run(isIPv6 bool) {
	t.init()

	t.createUtilPodLabel("e2e-dns-configmap")
	defer t.deleteUtilPod()
	originalConfigMapData := t.fetchDNSConfigMapData()
	defer t.restoreDNSConfigMap(originalConfigMapData)

	if isIPv6 {
		t.createDNSServer(t.f.Namespace.Name, map[string]string{
			"abc.acme.local": "2606:4700:4700::1111",
			"def.acme.local": "2606:4700:4700::2222",
			"widget.local":   "2606:4700:4700::3333",
		})
	} else {
		t.createDNSServer(t.f.Namespace.Name, map[string]string{
			"abc.acme.local": "1.1.1.1",
			"def.acme.local": "2.2.2.2",
			"widget.local":   "3.3.3.3",
		})
	}
	defer t.deleteDNSServerPod()

	if t.name == "coredns" {
		t.setConfigMap(&v1.ConfigMap{Data: map[string]string{
			"Corefile": fmt.Sprintf(`.:53 {
        health
        ready
        kubernetes %v in-addr.arpa ip6.arpa {
           pods insecure
           upstream
           fallthrough in-addr.arpa ip6.arpa
           ttl 30
        }
        forward . %v
    }
     acme.local:53 {
       forward . %v
    }`, framework.TestContext.ClusterDNSDomain, t.dnsServerPod.Status.PodIP, t.dnsServerPod.Status.PodIP),
		}})

		t.deleteCoreDNSPods()
	} else {
		t.setConfigMap(&v1.ConfigMap{Data: map[string]string{
			"stubDomains":         fmt.Sprintf(`{"acme.local":["%v"]}`, t.dnsServerPod.Status.PodIP),
			"upstreamNameservers": fmt.Sprintf(`["%v"]`, t.dnsServerPod.Status.PodIP),
		}})
	}

	if isIPv6 {
		t.checkDNSRecordFrom(
			"abc.acme.local",
			func(actual []string) bool { return len(actual) == 1 && actual[0] == "2606:4700:4700::1111" },
			"cluster-dns-ipv6",
			moreForeverTestTimeout)
		t.checkDNSRecordFrom(
			"def.acme.local",
			func(actual []string) bool { return len(actual) == 1 && actual[0] == "2606:4700:4700::2222" },
			"cluster-dns-ipv6",
			moreForeverTestTimeout)
		t.checkDNSRecordFrom(
			"widget.local",
			func(actual []string) bool { return len(actual) == 1 && actual[0] == "2606:4700:4700::3333" },
			"cluster-dns-ipv6",
			moreForeverTestTimeout)
	} else {
		t.checkDNSRecordFrom(
			"abc.acme.local",
			func(actual []string) bool { return len(actual) == 1 && actual[0] == "1.1.1.1" },
			"cluster-dns",
			moreForeverTestTimeout)
		t.checkDNSRecordFrom(
			"def.acme.local",
			func(actual []string) bool { return len(actual) == 1 && actual[0] == "2.2.2.2" },
			"cluster-dns",
			moreForeverTestTimeout)
		t.checkDNSRecordFrom(
			"widget.local",
			func(actual []string) bool { return len(actual) == 1 && actual[0] == "3.3.3.3" },
			"cluster-dns",
			moreForeverTestTimeout)
	}

	t.restoreDNSConfigMap(originalConfigMapData)
	// Wait for the deleted ConfigMap to take effect, otherwise the
	// configuration can bleed into other tests.
	t.checkDNSRecordFrom(
		"abc.acme.local",
		func(actual []string) bool { return len(actual) == 0 },
		"cluster-dns",
		moreForeverTestTimeout)
}

type dnsPtrFwdTest struct {
	dnsTestCommon
}

func (t *dnsPtrFwdTest) run(isIPv6 bool) {
	t.init()

	t.createUtilPodLabel("e2e-dns-configmap")
	defer t.deleteUtilPod()
	originalConfigMapData := t.fetchDNSConfigMapData()
	defer t.restoreDNSConfigMap(originalConfigMapData)

	t.createDNSServerWithPtrRecord(t.f.Namespace.Name, isIPv6)
	defer t.deleteDNSServerPod()

	// Should still be able to lookup public nameserver without explicit upstream nameserver set.
	if isIPv6 {
		t.checkDNSRecordFrom(
			"2001:4860:4860::8888",
			func(actual []string) bool { return len(actual) == 1 && actual[0] == googleDNSHostname+"." },
			"ptr-record",
			moreForeverTestTimeout)
	} else {
		t.checkDNSRecordFrom(
			"8.8.8.8",
			func(actual []string) bool { return len(actual) == 1 && actual[0] == googleDNSHostname+"." },
			"ptr-record",
			moreForeverTestTimeout)
	}

	if t.name == "coredns" {
		t.setConfigMap(&v1.ConfigMap{Data: map[string]string{
			"Corefile": fmt.Sprintf(`.:53 {
        health
        ready
        kubernetes %v in-addr.arpa ip6.arpa {
           pods insecure
           upstream
           fallthrough in-addr.arpa ip6.arpa
           ttl 30
        }
        forward . %v
    }`, framework.TestContext.ClusterDNSDomain, t.dnsServerPod.Status.PodIP),
		}})

		t.deleteCoreDNSPods()
	} else {
		t.setConfigMap(&v1.ConfigMap{Data: map[string]string{
			"upstreamNameservers": fmt.Sprintf(`["%v"]`, t.dnsServerPod.Status.PodIP),
		}})
	}

	if isIPv6 {
		t.checkDNSRecordFrom(
			"2001:db8::29",
			func(actual []string) bool { return len(actual) == 1 && actual[0] == "my.test." },
			"ptr-record",
			moreForeverTestTimeout)

		t.restoreDNSConfigMap(originalConfigMapData)
		t.checkDNSRecordFrom(
			"2001:db8::29",
			func(actual []string) bool { return len(actual) == 0 },
			"ptr-record",
			moreForeverTestTimeout)

	} else {
		t.checkDNSRecordFrom(
			"192.0.2.123",
			func(actual []string) bool { return len(actual) == 1 && actual[0] == "my.test." },
			"ptr-record",
			moreForeverTestTimeout)

		t.restoreDNSConfigMap(originalConfigMapData)
		t.checkDNSRecordFrom(
			"192.0.2.123",
			func(actual []string) bool { return len(actual) == 0 },
			"ptr-record",
			moreForeverTestTimeout)
	}
}

type dnsExternalNameTest struct {
	dnsTestCommon
}

func (t *dnsExternalNameTest) run(isIPv6 bool) {
	t.init()

	t.createUtilPodLabel("e2e-dns-configmap")
	defer t.deleteUtilPod()
	originalConfigMapData := t.fetchDNSConfigMapData()
	defer t.restoreDNSConfigMap(originalConfigMapData)

	fooHostname := "foo.example.com"
	if isIPv6 {
		t.createDNSServer(t.f.Namespace.Name, map[string]string{
			fooHostname: "2001:db8::29",
		})
	} else {
		t.createDNSServer(t.f.Namespace.Name, map[string]string{
			fooHostname: "192.0.2.123",
		})
	}
	defer t.deleteDNSServerPod()

	f := t.f
	serviceName := "dns-externalname-upstream-test"
	externalNameService := e2eservice.CreateServiceSpec(serviceName, googleDNSHostname, false, nil)
	if _, err := f.ClientSet.CoreV1().Services(f.Namespace.Name).Create(context.TODO(), externalNameService, metav1.CreateOptions{}); err != nil {
		ginkgo.Fail(fmt.Sprintf("ginkgo.Failed when creating service: %v", err))
	}
	serviceNameLocal := "dns-externalname-upstream-local"
	externalNameServiceLocal := e2eservice.CreateServiceSpec(serviceNameLocal, fooHostname, false, nil)
	if _, err := f.ClientSet.CoreV1().Services(f.Namespace.Name).Create(context.TODO(), externalNameServiceLocal, metav1.CreateOptions{}); err != nil {
		ginkgo.Fail(fmt.Sprintf("ginkgo.Failed when creating service: %v", err))
	}
	defer func() {
		ginkgo.By("deleting the test externalName service")
		defer ginkgo.GinkgoRecover()
		f.ClientSet.CoreV1().Services(f.Namespace.Name).Delete(context.TODO(), externalNameService.Name, metav1.DeleteOptions{})
		f.ClientSet.CoreV1().Services(f.Namespace.Name).Delete(context.TODO(), externalNameServiceLocal.Name, metav1.DeleteOptions{})
	}()

	if isIPv6 {
		t.checkDNSRecordFrom(
			fmt.Sprintf("%s.%s.svc.%s", serviceName, f.Namespace.Name, framework.TestContext.ClusterDNSDomain),
			func(actual []string) bool {
				return len(actual) >= 1 && actual[0] == googleDNSHostname+"."
			},
			"cluster-dns-ipv6",
			moreForeverTestTimeout)
	} else {
		t.checkDNSRecordFrom(
			fmt.Sprintf("%s.%s.svc.%s", serviceName, f.Namespace.Name, framework.TestContext.ClusterDNSDomain),
			func(actual []string) bool {
				return len(actual) >= 1 && actual[0] == googleDNSHostname+"."
			},
			"cluster-dns",
			moreForeverTestTimeout)
	}

	if t.name == "coredns" {
		t.setConfigMap(&v1.ConfigMap{Data: map[string]string{
			"Corefile": fmt.Sprintf(`.:53 {
        health
        ready
        kubernetes %v in-addr.arpa ip6.arpa {
           pods insecure
           upstream
           fallthrough in-addr.arpa ip6.arpa
           ttl 30
        }
        forward . %v
    }`, framework.TestContext.ClusterDNSDomain, t.dnsServerPod.Status.PodIP),
		}})

		t.deleteCoreDNSPods()
	} else {
		t.setConfigMap(&v1.ConfigMap{Data: map[string]string{
			"upstreamNameservers": fmt.Sprintf(`["%v"]`, t.dnsServerPod.Status.PodIP),
		}})
	}
	if isIPv6 {
		t.checkDNSRecordFrom(
			fmt.Sprintf("%s.%s.svc.%s", serviceNameLocal, f.Namespace.Name, framework.TestContext.ClusterDNSDomain),
			func(actual []string) bool {
				return len(actual) >= 1 && actual[0] == fooHostname+"." && actual[1] == "2001:db8::29"
			},
			"cluster-dns-ipv6",
			moreForeverTestTimeout)
	} else {
		t.checkDNSRecordFrom(
			fmt.Sprintf("%s.%s.svc.%s", serviceNameLocal, f.Namespace.Name, framework.TestContext.ClusterDNSDomain),
			func(actual []string) bool {
				return len(actual) == 2 && actual[0] == fooHostname+"." && actual[1] == "192.0.2.123"
			},
			"cluster-dns",
			moreForeverTestTimeout)
	}

	t.restoreDNSConfigMap(originalConfigMapData)
}

var _ = SIGDescribe("DNS configMap nameserver [IPv4]", func() {

	ginkgo.Context("Change stubDomain", func() {
		nsTest := &dnsNameserverTest{dnsTestCommon: newDNSTestCommon()}

		ginkgo.It("should be able to change stubDomain configuration [Slow][Serial]", func() {
			nsTest.c = nsTest.f.ClientSet
			nsTest.run(false)
		})
	})

	ginkgo.Context("Forward PTR lookup", func() {
		fwdTest := &dnsPtrFwdTest{dnsTestCommon: newDNSTestCommon()}

		ginkgo.It("should forward PTR records lookup to upstream nameserver [Slow][Serial]", func() {
			fwdTest.c = fwdTest.f.ClientSet
			fwdTest.run(false)
		})
	})

	ginkgo.Context("Forward external name lookup", func() {
		externalNameTest := &dnsExternalNameTest{dnsTestCommon: newDNSTestCommon()}

		ginkgo.It("should forward externalname lookup to upstream nameserver [Slow][Serial]", func() {
			externalNameTest.c = externalNameTest.f.ClientSet
			externalNameTest.run(false)
		})
	})
})

var _ = SIGDescribe("DNS configMap nameserver [Feature:Networking-IPv6] [LinuxOnly]", func() {

	ginkgo.BeforeEach(func() {
		// IPv6 is not supported on Windows.
		e2eskipper.SkipIfNodeOSDistroIs("windows")
	})

	ginkgo.Context("Change stubDomain", func() {
		nsTest := &dnsNameserverTest{dnsTestCommon: newDNSTestCommon()}

		ginkgo.It("should be able to change stubDomain configuration [Slow][Serial]", func() {
			nsTest.c = nsTest.f.ClientSet
			nsTest.run(true)
		})
	})

	ginkgo.Context("Forward PTR lookup", func() {
		fwdTest := &dnsPtrFwdTest{dnsTestCommon: newDNSTestCommon()}

		ginkgo.It("should forward PTR records lookup to upstream nameserver [Slow][Serial]", func() {
			fwdTest.c = fwdTest.f.ClientSet
			fwdTest.run(true)
		})
	})

	ginkgo.Context("Forward external name lookup", func() {
		externalNameTest := &dnsExternalNameTest{dnsTestCommon: newDNSTestCommon()}

		ginkgo.It("should forward externalname lookup to upstream nameserver [Slow][Serial]", func() {
			externalNameTest.c = externalNameTest.f.ClientSet
			externalNameTest.run(true)
		})
	})
})
