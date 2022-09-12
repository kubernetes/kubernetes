package iptables

import (
	"fmt"
	"net"
	"net/http"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/apiserver/pkg/server/mux"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/events"
	"k8s.io/component-base/configz"
	"k8s.io/component-base/logs"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	utilsysctl "k8s.io/component-helpers/node/util/sysctl"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/proxy/apis/config/scheme"
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	"k8s.io/kubernetes/pkg/proxy/metrics"
	proxymetrics "k8s.io/kubernetes/pkg/proxy/metrics"
	proxyutiliptables "k8s.io/kubernetes/pkg/proxy/util/iptables"
	utilproxytest "k8s.io/kubernetes/pkg/proxy/util/testing"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	"k8s.io/utils/exec"
	netutils "k8s.io/utils/net"
)

func TestIPTablesRestore(t *testing.T) {
	logs.GlogSetter("2")
	proxyServer := newProxier()
	proxyServer.OnServiceSynced()
	proxyServer.OnEndpointSlicesSynced()

	go proxyServer.SyncLoop()

	// dump metrics
	go func() {
		for {
			natRules, _ := testutil.GetGaugeMetricValue(metrics.IptablesRulesTotal.WithLabelValues(string(utiliptables.TableNAT)))
			//latencyCount, _ := testutil.GetHistogramMetricCount(metrics.SyncProxyRulesLatency)
			//latencyValue, _ := testutil.GetHistogramMetricValue(metrics.SyncProxyRulesLatency)
			svcChanges, _ := testutil.GetCounterMetricValue(metrics.ServiceChangesTotal)
			epChanges, _ := testutil.GetCounterMetricValue(metrics.EndpointChangesTotal)
			restoreFailures, _ := testutil.GetCounterMetricValue(metrics.IptablesRestoreFailuresTotal)

			fmt.Printf("%v: rules: %0.2f svc: %0.2f eps: %0.2f restoreFailures: %0.2f \n", time.Now(), natRules, svcChanges, epChanges, restoreFailures)
			time.Sleep(1 * time.Second)
		}

	}()

	syncPeriod := 10 * time.Second
	n := 100
	services := 1000
	endpointsPerService := 100
	// 100 * 10 sec = 1000 sec
	for j := 0; j < n; j++ {

		svcs, eps := generateServiceEndpoints(services, endpointsPerService, func(eps *discovery.EndpointSlice) {}, func(svc *v1.Service) {})

		// add incrementally
		fmt.Println("Adding services")
		for i := range svcs {
			proxyServer.OnServiceAdd(svcs[i])
		}
		fmt.Println("Adding endpoints")
		for i := range eps {
			proxyServer.OnEndpointSliceAdd(eps[i])
		}

		// full sync period
		time.Sleep(syncPeriod / 2)
		fmt.Println("Removing services")
		for i := range svcs {
			proxyServer.OnServiceDelete(svcs[i])
		}
		fmt.Println("Removing endpoints")
		for i := range eps {
			proxyServer.OnEndpointSliceDelete(eps[i])
		}
		time.Sleep(syncPeriod / 2)

	}

}

func newProxier() proxy.Provider {
	client := fake.NewSimpleClientset()

	execer := exec.New()

	// cleanup IPv6 and IPv4 iptables rules, regardless of current configuration
	ipts := [2]utiliptables.Interface{
		utiliptables.New(execer, utiliptables.ProtocolIPv4),
		utiliptables.New(execer, utiliptables.ProtocolIPv6),
	}

	detectLocal, _ := proxyutiliptables.NewDetectLocalByCIDR("10.0.0.0/8", ipts[0])
	detectLocalV6, _ := proxyutiliptables.NewDetectLocalByCIDR("fd00:1:2:3::/64", ipts[1])

	detectors := [2]proxyutiliptables.LocalTrafficDetector{
		detectLocal, detectLocalV6,
	}
	networkInterfacer := utilproxytest.NewFakeNetwork()
	itf := net.Interface{Index: 0, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0}
	addrs := []net.Addr{
		&net.IPNet{IP: netutils.ParseIPSloppy("127.0.0.1"), Mask: net.CIDRMask(8, 32)},
		&net.IPNet{IP: netutils.ParseIPSloppy("::1/128"), Mask: net.CIDRMask(128, 128)},
	}
	networkInterfacer.AddInterfaceAddr(&itf, addrs)
	itf1 := net.Interface{Index: 1, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0}
	addrs1 := []net.Addr{
		&net.IPNet{IP: netutils.ParseIPSloppy(testNodeIP), Mask: net.CIDRMask(24, 32)},
	}
	networkInterfacer.AddInterfaceAddr(&itf1, addrs1)

	eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1()})
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, "kube-proxy")
	var healthzServer healthcheck.ProxierHealthUpdater

	proxyMux := mux.NewPathRecorderMux("kube-proxy")
	proxyMux.Handle("/metrics", legacyregistry.Handler())
	configz.InstallHandler(proxyMux)

	go func() {
		err := http.ListenAndServe(":9191", proxyMux)
		if err != nil {
			fmt.Println("error running metrics server")
		}
	}()
	proxymetrics.RegisterMetrics()

	proxier, err := NewDualStackProxier(
		ipts,
		utilsysctl.New(),
		execer,
		30*time.Second, // config.IPTables.SyncPeriod.Duration,
		1*time.Second,  //config.IPTables.MinSyncPeriod.Duration,
		false,          // config.IPTables.MasqueradeAll,
		14,             // MasqueradeBit
		detectors,
		"localhost",
		[2]net.IP{net.IPv4zero, net.IPv6zero},
		recorder,
		healthzServer,
		make([]string, 0),
	)
	if err != nil {
		panic(err)
	}

	return proxier
}
