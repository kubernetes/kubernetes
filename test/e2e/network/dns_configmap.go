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
	"k8s.io/kubernetes/test/e2e/framework"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	"k8s.io/kubernetes/test/e2e/network/common"

	"github.com/onsi/ginkgo/v2"
)

var (
	googleDNSHostname = "dns.google"
	// The ConfigMap update mechanism takes longer than the standard
	// wait.ForeverTestTimeout.
	moreForeverTestTimeout = 2 * 60 * time.Second
)

type dnsNameserverTest struct {
	dnsTestCommon
}

func (t *dnsNameserverTest) run(ctx context.Context, isIPv6 bool) {
	t.init(ctx)

	t.createUtilPodLabel(ctx, "e2e-dns-configmap")
	ginkgo.DeferCleanup(t.deleteUtilPod)
	originalConfigMapData := t.fetchDNSConfigMapData(ctx)
	ginkgo.DeferCleanup(t.restoreDNSConfigMap, originalConfigMapData)

	if isIPv6 {
		t.createDNSServer(ctx, t.f.Namespace.Name, map[string]string{
			"abc.acme.local": "2606:4700:4700::1111",
			"def.acme.local": "2606:4700:4700::2222",
			"widget.local":   "2606:4700:4700::3333",
		})
	} else {
		t.createDNSServer(ctx, t.f.Namespace.Name, map[string]string{
			"abc.acme.local": "1.1.1.1",
			"def.acme.local": "2.2.2.2",
			"widget.local":   "3.3.3.3",
		})
	}
	ginkgo.DeferCleanup(t.deleteDNSServerPod)

	if t.name == "coredns" {
		t.setConfigMap(ctx, &v1.ConfigMap{Data: map[string]string{
			"Corefile": fmt.Sprintf(`.:53 {
        health
        ready
        kubernetes %v in-addr.arpa ip6.arpa {
           pods insecure
           fallthrough in-addr.arpa ip6.arpa
           ttl 30
        }
        forward . %v
    }
     acme.local:53 {
       forward . %v
    }`, framework.TestContext.ClusterDNSDomain, t.dnsServerPod.Status.PodIP, t.dnsServerPod.Status.PodIP),
		}})

		t.deleteCoreDNSPods(ctx)
	} else {
		t.setConfigMap(ctx, &v1.ConfigMap{Data: map[string]string{
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

	t.restoreDNSConfigMap(ctx, originalConfigMapData)
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

func (t *dnsPtrFwdTest) run(ctx context.Context, isIPv6 bool) {
	t.init(ctx)

	t.createUtilPodLabel(ctx, "e2e-dns-configmap")
	ginkgo.DeferCleanup(t.deleteUtilPod)
	originalConfigMapData := t.fetchDNSConfigMapData(ctx)
	ginkgo.DeferCleanup(t.restoreDNSConfigMap, originalConfigMapData)

	t.createDNSServerWithPtrRecord(ctx, t.f.Namespace.Name, isIPv6)
	ginkgo.DeferCleanup(t.deleteDNSServerPod)

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
		t.setConfigMap(ctx, &v1.ConfigMap{Data: map[string]string{
			"Corefile": fmt.Sprintf(`.:53 {
        health
        ready
        kubernetes %v in-addr.arpa ip6.arpa {
           pods insecure
           fallthrough in-addr.arpa ip6.arpa
           ttl 30
        }
        forward . %v
    }`, framework.TestContext.ClusterDNSDomain, t.dnsServerPod.Status.PodIP),
		}})

		t.deleteCoreDNSPods(ctx)
	} else {
		t.setConfigMap(ctx, &v1.ConfigMap{Data: map[string]string{
			"upstreamNameservers": fmt.Sprintf(`["%v"]`, t.dnsServerPod.Status.PodIP),
		}})
	}

	if isIPv6 {
		t.checkDNSRecordFrom(
			"2001:db8::29",
			func(actual []string) bool { return len(actual) == 1 && actual[0] == "my.test." },
			"ptr-record",
			moreForeverTestTimeout)

		t.restoreDNSConfigMap(ctx, originalConfigMapData)
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

		t.restoreDNSConfigMap(ctx, originalConfigMapData)
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

func (t *dnsExternalNameTest) run(ctx context.Context, isIPv6 bool) {
	t.init(ctx)

	t.createUtilPodLabel(ctx, "e2e-dns-configmap")
	ginkgo.DeferCleanup(t.deleteUtilPod)
	originalConfigMapData := t.fetchDNSConfigMapData(ctx)
	ginkgo.DeferCleanup(t.restoreDNSConfigMap, originalConfigMapData)

	fooHostname := "foo.example.com"
	if isIPv6 {
		t.createDNSServer(ctx, t.f.Namespace.Name, map[string]string{
			fooHostname: "2001:db8::29",
		})
	} else {
		t.createDNSServer(ctx, t.f.Namespace.Name, map[string]string{
			fooHostname: "192.0.2.123",
		})
	}
	ginkgo.DeferCleanup(t.deleteDNSServerPod)

	f := t.f
	serviceName := "dns-externalname-upstream-test"
	externalNameService := e2eservice.CreateServiceSpec(serviceName, googleDNSHostname, false, nil)
	if _, err := f.ClientSet.CoreV1().Services(f.Namespace.Name).Create(ctx, externalNameService, metav1.CreateOptions{}); err != nil {
		ginkgo.Fail(fmt.Sprintf("ginkgo.Failed when creating service: %v", err))
	}
	ginkgo.DeferCleanup(f.ClientSet.CoreV1().Services(f.Namespace.Name).Delete, externalNameService.Name, metav1.DeleteOptions{})
	serviceNameLocal := "dns-externalname-upstream-local"
	externalNameServiceLocal := e2eservice.CreateServiceSpec(serviceNameLocal, fooHostname, false, nil)
	if _, err := f.ClientSet.CoreV1().Services(f.Namespace.Name).Create(ctx, externalNameServiceLocal, metav1.CreateOptions{}); err != nil {
		ginkgo.Fail(fmt.Sprintf("ginkgo.Failed when creating service: %v", err))
	}
	ginkgo.DeferCleanup(f.ClientSet.CoreV1().Services(f.Namespace.Name).Delete, externalNameServiceLocal.Name, metav1.DeleteOptions{})

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
		t.setConfigMap(ctx, &v1.ConfigMap{Data: map[string]string{
			"Corefile": fmt.Sprintf(`.:53 {
        health
        ready
        kubernetes %v in-addr.arpa ip6.arpa {
           pods insecure
           fallthrough in-addr.arpa ip6.arpa
           ttl 30
        }
        forward . %v
    }`, framework.TestContext.ClusterDNSDomain, t.dnsServerPod.Status.PodIP),
		}})

		t.deleteCoreDNSPods(ctx)
	} else {
		t.setConfigMap(ctx, &v1.ConfigMap{Data: map[string]string{
			"upstreamNameservers": fmt.Sprintf(`["%v"]`, t.dnsServerPod.Status.PodIP),
		}})
	}
	if isIPv6 {
		t.checkDNSRecordFrom(
			fmt.Sprintf("%s.%s.svc.%s", serviceNameLocal, f.Namespace.Name, framework.TestContext.ClusterDNSDomain),
			func(actual []string) bool {
				return len(actual) >= 2 && actual[0] == fooHostname+"." && actual[1] == "2001:db8::29"
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

	t.restoreDNSConfigMap(ctx, originalConfigMapData)
}

var _ = common.SIGDescribe("DNS configMap nameserver", func() {

	ginkgo.Context("Change stubDomain", func() {
		nsTest := &dnsNameserverTest{dnsTestCommon: newDNSTestCommon()}

		framework.It("should be able to change stubDomain configuration", framework.WithSlow(), framework.WithSerial(), func(ctx context.Context) {
			nsTest.c = nsTest.f.ClientSet
			nsTest.run(ctx, framework.TestContext.ClusterIsIPv6())
		})
	})

	ginkgo.Context("Forward PTR lookup", func() {
		fwdTest := &dnsPtrFwdTest{dnsTestCommon: newDNSTestCommon()}

		framework.It("should forward PTR records lookup to upstream nameserver", framework.WithSlow(), framework.WithSerial(), func(ctx context.Context) {
			fwdTest.c = fwdTest.f.ClientSet
			fwdTest.run(ctx, framework.TestContext.ClusterIsIPv6())
		})
	})

	ginkgo.Context("Forward external name lookup", func() {
		externalNameTest := &dnsExternalNameTest{dnsTestCommon: newDNSTestCommon()}

		framework.It("should forward externalname lookup to upstream nameserver", framework.WithSlow(), framework.WithSerial(), func(ctx context.Context) {
			externalNameTest.c = externalNameTest.f.ClientSet
			externalNameTest.run(ctx, framework.TestContext.ClusterIsIPv6())
		})
	})
})
