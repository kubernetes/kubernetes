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

package ipvs

import (
	"bytes"
	"fmt"
	"net"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	netlinktest "k8s.io/kubernetes/pkg/proxy/ipvs/testing"
	"k8s.io/kubernetes/pkg/proxy/metrics"
	utilproxy "k8s.io/kubernetes/pkg/proxy/util"
	proxyutiliptables "k8s.io/kubernetes/pkg/proxy/util/iptables"
	proxyutiltest "k8s.io/kubernetes/pkg/proxy/util/testing"
	"k8s.io/kubernetes/pkg/util/async"
	utilipset "k8s.io/kubernetes/pkg/util/ipset"
	ipsettest "k8s.io/kubernetes/pkg/util/ipset/testing"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	iptablestest "k8s.io/kubernetes/pkg/util/iptables/testing"
	utilipvs "k8s.io/kubernetes/pkg/util/ipvs"
	ipvstest "k8s.io/kubernetes/pkg/util/ipvs/testing"
	"k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"
	netutils "k8s.io/utils/net"
	"k8s.io/utils/pointer"
)

const testHostname = "test-hostname"

type fakeIPGetter struct {
	nodeIPs   []net.IP
	bindedIPs sets.String
}

func (f *fakeIPGetter) NodeIPs() ([]net.IP, error) {
	return f.nodeIPs, nil
}

func (f *fakeIPGetter) BindedIPs() (sets.String, error) {
	return f.bindedIPs, nil
}

// fakeKernelHandler implements KernelHandler.
type fakeKernelHandler struct {
	modules       []string
	kernelVersion string
}

func (fake *fakeKernelHandler) GetModules() ([]string, error) {
	return fake.modules, nil
}

func (fake *fakeKernelHandler) GetKernelVersion() (string, error) {
	return fake.kernelVersion, nil
}

// fakeKernelHandler implements KernelHandler.
type fakeIPSetVersioner struct {
	version string
	err     error
}

func (fake *fakeIPSetVersioner) GetVersion() (string, error) {
	return fake.version, fake.err
}

func NewFakeProxier(ipt utiliptables.Interface, ipvs utilipvs.Interface, ipset utilipset.Interface, nodeIPs []net.IP, excludeCIDRs []*net.IPNet, ipFamily v1.IPFamily) *Proxier {
	// unlike actual proxier, this fake proxier does not filter node IPs per family requested
	// which can lead to false postives.

	// filter node IPs by proxier ipfamily
	idx := 0
	for _, nodeIP := range nodeIPs {
		if (ipFamily == v1.IPv6Protocol) == netutils.IsIPv6(nodeIP) {
			nodeIPs[idx] = nodeIP
			idx++
		}
	}

	// reset slice to filtered entries
	nodeIPs = nodeIPs[:idx]

	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			func() ([]byte, []byte, error) { return []byte("dummy device have been created"), nil, nil },
			func() ([]byte, []byte, error) { return []byte(""), nil, nil },
		},
	}
	fexec := &fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
		LookPathFunc: func(cmd string) (string, error) { return cmd, nil },
	}
	// initialize ipsetList with all sets we needed
	ipsetList := make(map[string]*IPSet)
	for _, is := range ipsetInfo {
		ipsetList[is.name] = NewIPSet(ipset, is.name, is.setType, false, is.comment)
	}
	p := &Proxier{
		exec:                  fexec,
		serviceMap:            make(proxy.ServiceMap),
		serviceChanges:        proxy.NewServiceChangeTracker(newServiceInfo, ipFamily, nil, nil),
		endpointsMap:          make(proxy.EndpointsMap),
		endpointsChanges:      proxy.NewEndpointChangeTracker(testHostname, nil, ipFamily, nil, nil),
		excludeCIDRs:          excludeCIDRs,
		iptables:              ipt,
		ipvs:                  ipvs,
		ipset:                 ipset,
		strictARP:             false,
		localDetector:         proxyutiliptables.NewNoOpLocalDetector(),
		hostname:              testHostname,
		serviceHealthServer:   healthcheck.NewFakeServiceHealthServer(),
		ipvsScheduler:         defaultScheduler,
		ipGetter:              &fakeIPGetter{nodeIPs: nodeIPs},
		iptablesData:          bytes.NewBuffer(nil),
		filterChainsData:      bytes.NewBuffer(nil),
		natChains:             utilproxy.LineBuffer{},
		natRules:              utilproxy.LineBuffer{},
		filterChains:          utilproxy.LineBuffer{},
		filterRules:           utilproxy.LineBuffer{},
		netlinkHandle:         netlinktest.NewFakeNetlinkHandle(),
		ipsetList:             ipsetList,
		nodePortAddresses:     make([]string, 0),
		networkInterfacer:     proxyutiltest.NewFakeNetwork(),
		gracefuldeleteManager: NewGracefulTerminationManager(ipvs),
		ipFamily:              ipFamily,
	}
	p.setInitialized(true)
	p.syncRunner = async.NewBoundedFrequencyRunner("test-sync-runner", p.syncProxyRules, 0, time.Minute, 1)
	return p
}

func makeNSN(namespace, name string) types.NamespacedName {
	return types.NamespacedName{Namespace: namespace, Name: name}
}

func makeServiceMap(proxier *Proxier, allServices ...*v1.Service) {
	for i := range allServices {
		proxier.OnServiceAdd(allServices[i])
	}

	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	proxier.servicesSynced = true
}

func makeTestService(namespace, name string, svcFunc func(*v1.Service)) *v1.Service {
	svc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Namespace:   namespace,
			Annotations: map[string]string{},
		},
		Spec:   v1.ServiceSpec{},
		Status: v1.ServiceStatus{},
	}
	svcFunc(svc)
	return svc
}

func populateEndpointSlices(proxier *Proxier, allEndpointSlices ...*discovery.EndpointSlice) {
	for i := range allEndpointSlices {
		proxier.OnEndpointSliceAdd(allEndpointSlices[i])
	}
}

func makeTestEndpointSlice(namespace, name string, sliceNum int, epsFunc func(*discovery.EndpointSlice)) *discovery.EndpointSlice {
	eps := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-%d", name, sliceNum),
			Namespace: namespace,
			Labels:    map[string]string{discovery.LabelServiceName: name},
		},
	}
	epsFunc(eps)
	return eps
}

func TestCleanupLeftovers(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, v1.IPv4Protocol)
	svcIP := "10.20.30.41"
	svcPort := 80
	svcNodePort := 3001
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	makeServiceMap(fp,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: v1.ProtocolTCP,
				NodePort: int32(svcNodePort),
			}}
		}),
	)
	epIP := "10.180.0.1"
	tcpProtocol := v1.ProtocolTCP
	populateEndpointSlices(fp,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIP},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     pointer.String(svcPortName.Port),
				Port:     pointer.Int32(int32(svcPort)),
				Protocol: &tcpProtocol,
			}}
		}),
	)

	fp.syncProxyRules()

	// test cleanup left over
	if CleanupLeftovers(ipvs, ipt, ipset) {
		t.Errorf("Cleanup leftovers failed")
	}
}

func TestCanUseIPVSProxier(t *testing.T) {
	testCases := []struct {
		mods          []string
		scheduler     string
		kernelVersion string
		kernelErr     error
		ipsetVersion  string
		ipsetErr      error
		ok            bool
	}{
		// case 0, kernel error
		{
			mods:          []string{"foo", "bar", "baz"},
			scheduler:     "",
			kernelVersion: "4.19",
			kernelErr:     fmt.Errorf("oops"),
			ipsetVersion:  "0.0",
			ok:            false,
		},
		// case 1, ipset error
		{
			mods:          []string{"foo", "bar", "baz"},
			scheduler:     "",
			kernelVersion: "4.19",
			ipsetVersion:  MinIPSetCheckVersion,
			ipsetErr:      fmt.Errorf("oops"),
			ok:            false,
		},
		// case 2, missing required kernel modules and ipset version too low
		{
			mods:          []string{"foo", "bar", "baz"},
			scheduler:     "rr",
			kernelVersion: "4.19",
			ipsetVersion:  "1.1",
			ok:            false,
		},
		// case 3, missing required ip_vs_* kernel modules
		{
			mods:          []string{"ip_vs", "a", "bc", "def"},
			scheduler:     "sed",
			kernelVersion: "4.19",
			ipsetVersion:  MinIPSetCheckVersion,
			ok:            false,
		},
		// case 4, ipset version too low
		{
			mods:          []string{"ip_vs", "ip_vs_rr", "ip_vs_wrr", "ip_vs_sh", "nf_conntrack"},
			scheduler:     "rr",
			kernelVersion: "4.19",
			ipsetVersion:  "4.3.0",
			ok:            false,
		},
		// case 5, ok for linux kernel 4.19
		{
			mods:          []string{"ip_vs", "ip_vs_rr", "ip_vs_wrr", "ip_vs_sh", "nf_conntrack"},
			scheduler:     "rr",
			kernelVersion: "4.19",
			ipsetVersion:  MinIPSetCheckVersion,
			ok:            true,
		},
		// case 6, ok for linux kernel 4.18
		{
			mods:          []string{"ip_vs", "ip_vs_rr", "ip_vs_wrr", "ip_vs_sh", "nf_conntrack_ipv4"},
			scheduler:     "rr",
			kernelVersion: "4.18",
			ipsetVersion:  MinIPSetCheckVersion,
			ok:            true,
		},
		// case 7. ok when module list has extra modules
		{
			mods:          []string{"foo", "ip_vs", "ip_vs_rr", "ip_vs_wrr", "ip_vs_sh", "nf_conntrack", "bar"},
			scheduler:     "rr",
			kernelVersion: "4.19",
			ipsetVersion:  "6.19",
			ok:            true,
		},
		// case 8, not ok for sed based IPVS scheduling
		{
			mods:          []string{"ip_vs", "ip_vs_rr", "ip_vs_wrr", "ip_vs_sh", "nf_conntrack"},
			scheduler:     "sed",
			kernelVersion: "4.19",
			ipsetVersion:  MinIPSetCheckVersion,
			ok:            false,
		},
		// case 9, ok for dh based IPVS scheduling
		{
			mods:          []string{"ip_vs", "ip_vs_rr", "ip_vs_wrr", "ip_vs_sh", "nf_conntrack", "ip_vs_dh"},
			scheduler:     "dh",
			kernelVersion: "4.19",
			ipsetVersion:  MinIPSetCheckVersion,
			ok:            true,
		},
		// case 10, non-existent scheduler, error due to modules not existing
		{
			mods:          []string{"ip_vs", "ip_vs_rr", "ip_vs_wrr", "ip_vs_sh", "nf_conntrack", "ip_vs_dh"},
			scheduler:     "foobar",
			kernelVersion: "4.19",
			ipsetVersion:  MinIPSetCheckVersion,
			ok:            false,
		},
	}

	for i := range testCases {
		handle := &fakeKernelHandler{modules: testCases[i].mods, kernelVersion: testCases[i].kernelVersion}
		versioner := &fakeIPSetVersioner{version: testCases[i].ipsetVersion, err: testCases[i].ipsetErr}
		ok, err := CanUseIPVSProxier(handle, versioner, testCases[i].scheduler)
		if ok != testCases[i].ok {
			t.Errorf("Case [%d], expect %v, got %v: err: %v", i, testCases[i].ok, ok, err)
		}
	}
}

func TestGetNodeIPs(t *testing.T) {
	testCases := []struct {
		isIPv6       bool
		devAddresses map[string][]string
		expectIPs    []string
	}{
		// case 0
		{
			devAddresses: map[string][]string{"eth0": {"1.2.3.4"}, "lo": {"127.0.0.1"}},
			expectIPs:    []string{"1.2.3.4"},
		},
		// case 1
		{
			devAddresses: map[string][]string{"lo": {"127.0.0.1"}},
			expectIPs:    []string{},
		},
		// case 2
		{
			devAddresses: map[string][]string{},
			expectIPs:    []string{},
		},
		// case 3
		{
			devAddresses: map[string][]string{"encap0": {"10.20.30.40", "fe80::200:ff:fe01:1"}, "lo": {"127.0.0.1", "::1"}, "docker0": {"172.17.0.1"}},
			expectIPs:    []string{"10.20.30.40", "172.17.0.1"},
		},
		// case 4
		{
			devAddresses: map[string][]string{"encaps9": {"10.20.30.40"}, "lo": {"127.0.0.1", "::1"}, "encap7": {"1000::", "10.20.30.31"}},
			expectIPs:    []string{"10.20.30.40", "10.20.30.31"},
		},
		// case 5
		{
			devAddresses: map[string][]string{"kube-ipvs0": {"2000::", "1.2.3.4"}, "lo": {"127.0.0.1", "::1"}, "encap7": {"1000::", "10.20.30.31"}},
			expectIPs:    []string{"10.20.30.31"},
		},
		// case 6
		{
			devAddresses: map[string][]string{"kube-ipvs0": {"1.2.3.4", "2.3.4.5"}, "lo": {"127.0.0.1", "::1"}},
			expectIPs:    []string{},
		},
		// case 7
		{
			devAddresses: map[string][]string{"kube-ipvs0": {"1.2.3.4", "2.3.4.5"}},
			expectIPs:    []string{},
		},
		// case 8
		{
			devAddresses: map[string][]string{"kube-ipvs0": {"1.2.3.4", "2.3.4.5"}, "eth5": {"3.4.5.6"}, "lo": {"127.0.0.1", "::1"}},
			expectIPs:    []string{"3.4.5.6"},
		},
		// case 9
		{
			devAddresses: map[string][]string{"ipvs0": {"1.2.3.4"}, "lo": {"127.0.0.1", "::1"}, "encap7": {"10.20.30.31"}},
			expectIPs:    []string{"10.20.30.31", "1.2.3.4"},
		},
		// case 10
		{
			isIPv6:       true,
			devAddresses: map[string][]string{"ipvs0": {"1.2.3.4", "1000::"}, "lo": {"127.0.0.1", "::1"}, "encap7": {"10.20.30.31", "2000::", "fe80::200:ff:fe01:1"}},
			expectIPs:    []string{"1000::", "2000::"},
		},
		// case 11
		{
			isIPv6:       true,
			devAddresses: map[string][]string{"ipvs0": {"1.2.3.4", "1000::"}, "lo": {"127.0.0.1", "::1"}, "encap7": {"10.20.30.31", "2000::", "fe80::200:ff:fe01:1"}, "kube-ipvs0": {"1.2.3.4", "2.3.4.5", "2000::"}},
			expectIPs:    []string{"1000::"},
		},
	}

	for i := range testCases {
		fake := netlinktest.NewFakeNetlinkHandle()
		fake.IsIPv6 = testCases[i].isIPv6
		for dev, addresses := range testCases[i].devAddresses {
			fake.SetLocalAddresses(dev, addresses...)
		}
		r := realIPGetter{nl: fake}
		ips, err := r.NodeIPs()
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		ipStrs := sets.NewString()
		for _, ip := range ips {
			ipStrs.Insert(ip.String())
		}
		if !ipStrs.Equal(sets.NewString(testCases[i].expectIPs...)) {
			t.Errorf("case[%d], unexpected mismatch, expected: %v, got: %v", i, testCases[i].expectIPs, ips)
		}
	}
}

func TestNodePortIPv4(t *testing.T) {
	tcpProtocol := v1.ProtocolTCP
	udpProtocol := v1.ProtocolUDP
	sctpProtocol := v1.ProtocolSCTP
	tests := []struct {
		name                   string
		services               []*v1.Service
		endpoints              []*discovery.EndpointSlice
		nodeIPs                []net.IP
		nodePortAddresses      []string
		expectedIPVS           *ipvstest.FakeIPVS
		expectedIPSets         netlinktest.ExpectedIPSet
		expectedIptablesChains netlinktest.ExpectedIptablesChain
	}{
		{
			name: "1 service with node port, has 2 endpoints",
			services: []*v1.Service{
				makeTestService("ns1", "svc1", func(svc *v1.Service) {
					svc.Spec.Type = "NodePort"
					svc.Spec.ClusterIP = "10.20.30.41"
					svc.Spec.Ports = []v1.ServicePort{{
						Name:     "p80",
						Port:     int32(80),
						Protocol: v1.ProtocolTCP,
						NodePort: int32(3001),
					}}
				}),
			},
			endpoints: []*discovery.EndpointSlice{
				makeTestEndpointSlice("ns1", "svc1", 1, func(eps *discovery.EndpointSlice) {
					eps.AddressType = discovery.AddressTypeIPv4
					eps.Endpoints = []discovery.Endpoint{{
						Addresses: []string{"10.180.0.1"},
					}}
					eps.Ports = []discovery.EndpointPort{{
						Name:     pointer.String("p80"),
						Port:     pointer.Int32(80),
						Protocol: &tcpProtocol,
					}}
				}),
				makeTestEndpointSlice("ns1", "svc1", 2, func(eps *discovery.EndpointSlice) {
					eps.AddressType = discovery.AddressTypeIPv6
					eps.Endpoints = []discovery.Endpoint{{
						Addresses: []string{"1002:ab8::2:10"},
					}}
					eps.Ports = []discovery.EndpointPort{{
						Name:     pointer.String("p80"),
						Port:     pointer.Int32(80),
						Protocol: &tcpProtocol,
					}}
				}),
			},
			nodeIPs: []net.IP{
				netutils.ParseIPSloppy("100.101.102.103"),
				netutils.ParseIPSloppy("2001:db8::1:1"),
			},
			nodePortAddresses: []string{},
			expectedIPVS: &ipvstest.FakeIPVS{
				Services: map[ipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "10.20.30.41",
						Port:     80,
						Protocol: "TCP",
					}: {
						Address:   netutils.ParseIPSloppy("10.20.30.41"),
						Protocol:  "TCP",
						Port:      uint16(80),
						Scheduler: "rr",
					},
					{
						IP:       "100.101.102.103",
						Port:     3001,
						Protocol: "TCP",
					}: {
						Address:   netutils.ParseIPSloppy("100.101.102.103"),
						Protocol:  "TCP",
						Port:      uint16(3001),
						Scheduler: "rr",
					},
				},
				Destinations: map[ipvstest.ServiceKey][]*utilipvs.RealServer{
					{
						IP:       "10.20.30.41",
						Port:     80,
						Protocol: "TCP",
					}: {
						{
							Address: netutils.ParseIPSloppy("10.180.0.1"),
							Port:    uint16(80),
							Weight:  1,
						},
					},
					{
						IP:       "100.101.102.103",
						Port:     3001,
						Protocol: "TCP",
					}: {
						{
							Address: netutils.ParseIPSloppy("10.180.0.1"),
							Port:    uint16(80),
							Weight:  1,
						},
					},
				},
			},
		},
		{
			name: "1 UDP service with node port, has endpoints",
			services: []*v1.Service{
				makeTestService("ns1", "svc1", func(svc *v1.Service) {
					svc.Spec.Type = "NodePort"
					svc.Spec.ClusterIP = "10.20.30.41"
					svc.Spec.Ports = []v1.ServicePort{{
						Name:     "p80",
						Port:     int32(80),
						Protocol: v1.ProtocolUDP,
						NodePort: int32(3001),
					}}
				}),
			},
			endpoints: []*discovery.EndpointSlice{
				makeTestEndpointSlice("ns1", "svc1", 1, func(eps *discovery.EndpointSlice) {
					eps.AddressType = discovery.AddressTypeIPv4
					eps.Endpoints = []discovery.Endpoint{{
						Addresses: []string{"10.180.0.1"},
					}}
					eps.Ports = []discovery.EndpointPort{{
						Name:     pointer.String("p80"),
						Port:     pointer.Int32(80),
						Protocol: &udpProtocol,
					}}
				}),
			},
			nodeIPs: []net.IP{
				netutils.ParseIPSloppy("100.101.102.103"),
			},
			nodePortAddresses: []string{"0.0.0.0/0"},
			expectedIPVS: &ipvstest.FakeIPVS{
				Services: map[ipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "10.20.30.41",
						Port:     80,
						Protocol: "UDP",
					}: {
						Address:   netutils.ParseIPSloppy("10.20.30.41"),
						Protocol:  "UDP",
						Port:      uint16(80),
						Scheduler: "rr",
					},
					{
						IP:       "100.101.102.103",
						Port:     3001,
						Protocol: "UDP",
					}: {
						Address:   netutils.ParseIPSloppy("100.101.102.103"),
						Protocol:  "UDP",
						Port:      uint16(3001),
						Scheduler: "rr",
					},
				},
				Destinations: map[ipvstest.ServiceKey][]*utilipvs.RealServer{
					{
						IP:       "10.20.30.41",
						Port:     80,
						Protocol: "UDP",
					}: {
						{
							Address: netutils.ParseIPSloppy("10.180.0.1"),
							Port:    uint16(80),
							Weight:  1,
						},
					},
					{
						IP:       "100.101.102.103",
						Port:     3001,
						Protocol: "UDP",
					}: {
						{
							Address: netutils.ParseIPSloppy("10.180.0.1"),
							Port:    uint16(80),
							Weight:  1,
						},
					},
				},
			},
			expectedIPSets: netlinktest.ExpectedIPSet{
				kubeNodePortSetUDP: {{
					Port:     3001,
					Protocol: strings.ToLower(string(v1.ProtocolUDP)),
					SetType:  utilipset.BitmapPort,
				}},
			},
			expectedIptablesChains: netlinktest.ExpectedIptablesChain{
				string(kubeNodePortChain): {{
					JumpChain: string(kubeMarkMasqChain), MatchSet: kubeNodePortSetUDP,
				}, {
					JumpChain: "ACCEPT", MatchSet: kubeHealthCheckNodePortSet,
				}},
				string(kubeServicesChain): {{
					JumpChain: string(kubeMarkMasqChain), MatchSet: kubeClusterIPSet,
				}, {
					JumpChain: string(kubeNodePortChain), MatchSet: "",
				}, {
					JumpChain: "ACCEPT", MatchSet: kubeClusterIPSet,
				}},
			},
		},
		{
			name: "service has node port but no endpoints",
			services: []*v1.Service{
				makeTestService("ns1", "svc1", func(svc *v1.Service) {
					svc.Spec.Type = "NodePort"
					svc.Spec.ClusterIP = "10.20.30.41"
					svc.Spec.Ports = []v1.ServicePort{{
						Name:     "p80",
						Port:     int32(80),
						Protocol: v1.ProtocolTCP,
						NodePort: int32(3001),
					}}
				}),
			},
			endpoints: []*discovery.EndpointSlice{},
			nodeIPs: []net.IP{
				netutils.ParseIPSloppy("100.101.102.103"),
			},
			nodePortAddresses: []string{},
			expectedIPVS: &ipvstest.FakeIPVS{
				Services: map[ipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "10.20.30.41",
						Port:     80,
						Protocol: "TCP",
					}: {
						Address:   netutils.ParseIPSloppy("10.20.30.41"),
						Protocol:  "TCP",
						Port:      uint16(80),
						Scheduler: "rr",
					},
					{
						IP:       "100.101.102.103",
						Port:     3001,
						Protocol: "TCP",
					}: {
						Address:   netutils.ParseIPSloppy("100.101.102.103"),
						Protocol:  "TCP",
						Port:      uint16(3001),
						Scheduler: "rr",
					},
				},
				Destinations: map[ipvstest.ServiceKey][]*utilipvs.RealServer{
					{
						IP:       "10.20.30.41",
						Port:     80,
						Protocol: "TCP",
					}: {}, // no real servers corresponding to no endpoints
					{
						IP:       "100.101.102.103",
						Port:     3001,
						Protocol: "TCP",
					}: {}, // no real servers corresponding to no endpoints
				},
			},
		},
		{
			name: "node port service with protocol sctp on a node with multiple nodeIPs",
			services: []*v1.Service{
				makeTestService("ns1", "svc1", func(svc *v1.Service) {
					svc.Spec.Type = "NodePort"
					svc.Spec.ClusterIP = "10.20.30.41"
					svc.Spec.Ports = []v1.ServicePort{{
						Name:     "p80",
						Port:     int32(80),
						Protocol: v1.ProtocolSCTP,
						NodePort: int32(3001),
					}}
				}),
			},
			endpoints: []*discovery.EndpointSlice{
				makeTestEndpointSlice("ns1", "svc1", 1, func(eps *discovery.EndpointSlice) {
					eps.AddressType = discovery.AddressTypeIPv4
					eps.Endpoints = []discovery.Endpoint{{
						Addresses: []string{"10.180.0.1"},
					}}
					eps.Ports = []discovery.EndpointPort{{
						Name:     pointer.String("p80"),
						Port:     pointer.Int32(80),
						Protocol: &sctpProtocol,
					}}
				}),
			},
			nodeIPs: []net.IP{
				netutils.ParseIPSloppy("100.101.102.103"),
				netutils.ParseIPSloppy("100.101.102.104"),
				netutils.ParseIPSloppy("100.101.102.105"),
				netutils.ParseIPSloppy("2001:db8::1:1"),
				netutils.ParseIPSloppy("2001:db8::1:2"),
				netutils.ParseIPSloppy("2001:db8::1:3"),
			},
			nodePortAddresses: []string{},
			expectedIPVS: &ipvstest.FakeIPVS{
				Services: map[ipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "10.20.30.41",
						Port:     80,
						Protocol: "SCTP",
					}: {
						Address:   netutils.ParseIPSloppy("10.20.30.41"),
						Protocol:  "SCTP",
						Port:      uint16(80),
						Scheduler: "rr",
					},
					{
						IP:       "100.101.102.103",
						Port:     3001,
						Protocol: "SCTP",
					}: {
						Address:   netutils.ParseIPSloppy("100.101.102.103"),
						Protocol:  "SCTP",
						Port:      uint16(3001),
						Scheduler: "rr",
					},
					{
						IP:       "100.101.102.104",
						Port:     3001,
						Protocol: "SCTP",
					}: {
						Address:   netutils.ParseIPSloppy("100.101.102.104"),
						Protocol:  "SCTP",
						Port:      uint16(3001),
						Scheduler: "rr",
					},
					{
						IP:       "100.101.102.105",
						Port:     3001,
						Protocol: "SCTP",
					}: {
						Address:   netutils.ParseIPSloppy("100.101.102.105"),
						Protocol:  "SCTP",
						Port:      uint16(3001),
						Scheduler: "rr",
					},
				},
				Destinations: map[ipvstest.ServiceKey][]*utilipvs.RealServer{
					{
						IP:       "10.20.30.41",
						Port:     80,
						Protocol: "SCTP",
					}: {
						{
							Address: netutils.ParseIPSloppy("10.180.0.1"),
							Port:    uint16(80),
							Weight:  1,
						},
					},
					{
						IP:       "100.101.102.103",
						Port:     3001,
						Protocol: "SCTP",
					}: {
						{
							Address: netutils.ParseIPSloppy("10.180.0.1"),
							Port:    uint16(80),
							Weight:  1,
						},
					},
					{
						IP:       "100.101.102.104",
						Port:     3001,
						Protocol: "SCTP",
					}: {
						{
							Address: netutils.ParseIPSloppy("10.180.0.1"),
							Port:    uint16(80),
							Weight:  1,
						},
					},
					{
						IP:       "100.101.102.105",
						Port:     3001,
						Protocol: "SCTP",
					}: {
						{
							Address: netutils.ParseIPSloppy("10.180.0.1"),
							Port:    uint16(80),
							Weight:  1,
						},
					},
				},
			},
			expectedIPSets: netlinktest.ExpectedIPSet{
				kubeNodePortSetSCTP: {
					{
						IP:       "100.101.102.103",
						Port:     3001,
						Protocol: strings.ToLower(string(v1.ProtocolSCTP)),
						SetType:  utilipset.HashIPPort,
					},
					{
						IP:       "100.101.102.104",
						Port:     3001,
						Protocol: strings.ToLower(string(v1.ProtocolSCTP)),
						SetType:  utilipset.HashIPPort,
					},
					{
						IP:       "100.101.102.105",
						Port:     3001,
						Protocol: strings.ToLower(string(v1.ProtocolSCTP)),
						SetType:  utilipset.HashIPPort,
					},
				},
			},
		},
		{
			name: "node port service with protocol sctp and externalTrafficPolicy local",
			services: []*v1.Service{
				makeTestService("ns1", "svc1", func(svc *v1.Service) {
					svc.Spec.Type = "NodePort"
					svc.Spec.ClusterIP = "10.20.30.41"
					svc.Spec.Ports = []v1.ServicePort{{
						Name:     "p80",
						Port:     int32(80),
						Protocol: v1.ProtocolSCTP,
						NodePort: int32(3001),
					}}
					svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
				}),
			},
			endpoints: []*discovery.EndpointSlice{
				makeTestEndpointSlice("ns1", "svc1", 1, func(eps *discovery.EndpointSlice) {
					eps.AddressType = discovery.AddressTypeIPv4
					eps.Endpoints = []discovery.Endpoint{{
						Addresses: []string{"10.180.0.1"},
						NodeName:  pointer.String(testHostname),
					}, {
						Addresses: []string{"10.180.1.1"},
						NodeName:  pointer.String("otherHost"),
					}}
					eps.Ports = []discovery.EndpointPort{{
						Name:     pointer.String("p80"),
						Port:     pointer.Int32(80),
						Protocol: &sctpProtocol,
					}}
				}),
			},
			nodeIPs: []net.IP{
				netutils.ParseIPSloppy("100.101.102.103"),
			},
			nodePortAddresses: []string{},
			expectedIPVS: &ipvstest.FakeIPVS{
				Services: map[ipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "10.20.30.41",
						Port:     80,
						Protocol: "SCTP",
					}: {
						Address:   netutils.ParseIPSloppy("10.20.30.41"),
						Protocol:  "SCTP",
						Port:      uint16(80),
						Scheduler: "rr",
					},
					{
						IP:       "100.101.102.103",
						Port:     3001,
						Protocol: "SCTP",
					}: {
						Address:   netutils.ParseIPSloppy("100.101.102.103"),
						Protocol:  "SCTP",
						Port:      uint16(3001),
						Scheduler: "rr",
					},
				},
				Destinations: map[ipvstest.ServiceKey][]*utilipvs.RealServer{
					{
						IP:       "10.20.30.41",
						Port:     80,
						Protocol: "SCTP",
					}: {
						{
							Address: netutils.ParseIPSloppy("10.180.0.1"),
							Port:    uint16(80),
							Weight:  1,
						},
						{
							Address: netutils.ParseIPSloppy("10.180.1.1"),
							Port:    uint16(80),
							Weight:  1,
						},
					},
					{
						IP:       "100.101.102.103",
						Port:     3001,
						Protocol: "SCTP",
					}: {
						{
							Address: netutils.ParseIPSloppy("10.180.0.1"),
							Port:    uint16(80),
							Weight:  1,
						},
					},
				},
			},
			expectedIPSets: netlinktest.ExpectedIPSet{
				kubeNodePortSetSCTP: {
					{
						IP:       "100.101.102.103",
						Port:     3001,
						Protocol: strings.ToLower(string(v1.ProtocolSCTP)),
						SetType:  utilipset.HashIPPort,
					},
				},
				kubeNodePortLocalSetSCTP: {
					{
						IP:       "100.101.102.103",
						Port:     3001,
						Protocol: strings.ToLower(string(v1.ProtocolSCTP)),
						SetType:  utilipset.HashIPPort,
					},
				},
			},
			expectedIptablesChains: netlinktest.ExpectedIptablesChain{
				string(kubeNodePortChain): {{
					JumpChain: "RETURN", MatchSet: kubeNodePortLocalSetSCTP,
				}, {
					JumpChain: string(kubeMarkMasqChain), MatchSet: kubeNodePortSetSCTP,
				}, {
					JumpChain: "ACCEPT", MatchSet: kubeHealthCheckNodePortSet,
				}},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ipt := iptablestest.NewFake()
			ipvs := ipvstest.NewFake()
			ipset := ipsettest.NewFake(testIPSetVersion)
			fp := NewFakeProxier(ipt, ipvs, ipset, test.nodeIPs, nil, v1.IPv4Protocol)
			fp.nodePortAddresses = test.nodePortAddresses

			makeServiceMap(fp, test.services...)
			populateEndpointSlices(fp, test.endpoints...)

			fp.syncProxyRules()

			if !reflect.DeepEqual(ipvs, test.expectedIPVS) {
				t.Logf("actual ipvs state: %+v", ipvs)
				t.Logf("expected ipvs state: %+v", test.expectedIPVS)
				t.Errorf("unexpected IPVS state")
			}

			if test.expectedIPSets != nil {
				checkIPSet(t, fp, test.expectedIPSets)
			}

			if test.expectedIptablesChains != nil {
				checkIptables(t, ipt, test.expectedIptablesChains)
			}
		})
	}
}

func TestNodePortIPv6(t *testing.T) {
	tcpProtocol := v1.ProtocolTCP
	udpProtocol := v1.ProtocolUDP
	sctpProtocol := v1.ProtocolSCTP
	tests := []struct {
		name                   string
		services               []*v1.Service
		endpoints              []*discovery.EndpointSlice
		nodeIPs                []net.IP
		nodePortAddresses      []string
		expectedIPVS           *ipvstest.FakeIPVS
		expectedIPSets         netlinktest.ExpectedIPSet
		expectedIptablesChains netlinktest.ExpectedIptablesChain
	}{
		{
			name: "1 service with node port, has 2 endpoints",
			services: []*v1.Service{
				makeTestService("ns1", "svc1", func(svc *v1.Service) {
					svc.Spec.Type = "NodePort"
					svc.Spec.ClusterIP = "2020::1"
					svc.Spec.Ports = []v1.ServicePort{{
						Name:     "p80",
						Port:     int32(80),
						Protocol: v1.ProtocolTCP,
						NodePort: int32(3001),
					}}
				}),
			},
			endpoints: []*discovery.EndpointSlice{
				makeTestEndpointSlice("ns1", "svc1", 1, func(eps *discovery.EndpointSlice) {
					eps.AddressType = discovery.AddressTypeIPv4
					eps.Endpoints = []discovery.Endpoint{{
						Addresses: []string{"10.180.0.1"},
					}}
					eps.Ports = []discovery.EndpointPort{{
						Name:     pointer.String("p80"),
						Port:     pointer.Int32(80),
						Protocol: &tcpProtocol,
					}}
				}),
				makeTestEndpointSlice("ns1", "svc1", 2, func(eps *discovery.EndpointSlice) {
					eps.AddressType = discovery.AddressTypeIPv6
					eps.Endpoints = []discovery.Endpoint{{
						Addresses: []string{"1002:ab8::2:10"},
					}}
					eps.Ports = []discovery.EndpointPort{{
						Name:     pointer.String("p80"),
						Port:     pointer.Int32(80),
						Protocol: &tcpProtocol,
					}}
				}),
			},
			nodeIPs: []net.IP{
				netutils.ParseIPSloppy("100.101.102.103"),
				netutils.ParseIPSloppy("2001:db8::1:1"),
			},
			nodePortAddresses: []string{},
			expectedIPVS: &ipvstest.FakeIPVS{
				Services: map[ipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "2001:db8::1:1",
						Port:     3001,
						Protocol: "TCP",
					}: {
						Address:   netutils.ParseIPSloppy("2001:db8::1:1"),
						Protocol:  "TCP",
						Port:      uint16(3001),
						Scheduler: "rr",
					},
					{
						IP:       "2020::1",
						Port:     80,
						Protocol: "TCP",
					}: {
						Address:   netutils.ParseIPSloppy("2020::1"),
						Protocol:  "TCP",
						Port:      uint16(80),
						Scheduler: "rr",
					},
				},
				Destinations: map[ipvstest.ServiceKey][]*utilipvs.RealServer{
					{
						IP:       "2001:db8::1:1",
						Port:     3001,
						Protocol: "TCP",
					}: {
						{
							Address: netutils.ParseIPSloppy("1002:ab8::2:10"),
							Port:    uint16(80),
							Weight:  1,
						},
					},

					{
						IP:       "2020::1",
						Port:     80,
						Protocol: "TCP",
					}: {
						{
							Address: netutils.ParseIPSloppy("1002:ab8::2:10"),
							Port:    uint16(80),
							Weight:  1,
						},
					},
				},
			},
		},

		{
			name: "1 UDP service with node port, has endpoints (no action on IPv6 Proxier)",
			services: []*v1.Service{
				makeTestService("ns1", "svc1", func(svc *v1.Service) {
					svc.Spec.Type = "NodePort"
					svc.Spec.ClusterIP = "10.20.30.41"
					svc.Spec.Ports = []v1.ServicePort{{
						Name:     "p80",
						Port:     int32(80),
						Protocol: v1.ProtocolUDP,
						NodePort: int32(3001),
					}}
				}),
			},
			endpoints: []*discovery.EndpointSlice{
				makeTestEndpointSlice("ns1", "svc1", 1, func(eps *discovery.EndpointSlice) {
					eps.AddressType = discovery.AddressTypeIPv6
					eps.Endpoints = []discovery.Endpoint{{
						Addresses: []string{"10.180.0.1"},
					}}
					eps.Ports = []discovery.EndpointPort{{
						Name:     pointer.String("p80"),
						Port:     pointer.Int32(80),
						Protocol: &udpProtocol,
					}}
				}),
			},
			nodeIPs: []net.IP{
				netutils.ParseIPSloppy("100.101.102.103"),
			},
			nodePortAddresses: []string{"0.0.0.0/0"},
			/*since this is a node with only IPv4, proxier should not do anything */
			expectedIPVS: &ipvstest.FakeIPVS{
				Services:     map[ipvstest.ServiceKey]*utilipvs.VirtualServer{},
				Destinations: map[ipvstest.ServiceKey][]*utilipvs.RealServer{},
			},
			expectedIPSets:         nil,
			expectedIptablesChains: nil,
		},

		{
			name: "service has node port but no endpoints",
			services: []*v1.Service{
				makeTestService("ns1", "svc1", func(svc *v1.Service) {
					svc.Spec.Type = "NodePort"
					svc.Spec.ClusterIP = "2020::1"
					svc.Spec.Ports = []v1.ServicePort{{
						Name:     "p80",
						Port:     int32(80),
						Protocol: v1.ProtocolTCP,
						NodePort: int32(3001),
					}}
				}),
			},
			endpoints: []*discovery.EndpointSlice{},
			nodeIPs: []net.IP{
				netutils.ParseIPSloppy("100.101.102.103"),
				netutils.ParseIPSloppy("2001:db8::1:1"),
			},
			nodePortAddresses: []string{},
			expectedIPVS: &ipvstest.FakeIPVS{
				Services: map[ipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "2001:db8::1:1",
						Port:     3001,
						Protocol: "TCP",
					}: {
						Address:   netutils.ParseIPSloppy("2001:db8::1:1"),
						Protocol:  "TCP",
						Port:      uint16(3001),
						Scheduler: "rr",
					},
					{
						IP:       "2020::1",
						Port:     80,
						Protocol: "TCP",
					}: {
						Address:   netutils.ParseIPSloppy("2020::1"),
						Protocol:  "TCP",
						Port:      uint16(80),
						Scheduler: "rr",
					},
				},
				Destinations: map[ipvstest.ServiceKey][]*utilipvs.RealServer{
					{
						IP:       "2020::1",
						Port:     80,
						Protocol: "TCP",
					}: {}, // no real servers corresponding to no endpoints
					{
						IP:       "2001:db8::1:1",
						Port:     3001,
						Protocol: "TCP",
					}: {}, // no real servers corresponding to no endpoints
				},
			},
		},

		{
			name: "node port service with protocol sctp on a node with multiple nodeIPs",
			services: []*v1.Service{
				makeTestService("ns1", "svc1", func(svc *v1.Service) {
					svc.Spec.Type = "NodePort"
					svc.Spec.ClusterIP = "2020::1"
					svc.Spec.Ports = []v1.ServicePort{{
						Name:     "p80",
						Port:     int32(80),
						Protocol: v1.ProtocolSCTP,
						NodePort: int32(3001),
					}}
				}),
			},
			endpoints: []*discovery.EndpointSlice{
				makeTestEndpointSlice("ns1", "svc1", 1, func(eps *discovery.EndpointSlice) {
					eps.AddressType = discovery.AddressTypeIPv6
					eps.Endpoints = []discovery.Endpoint{{
						Addresses: []string{"2001::1"},
					}}
					eps.Ports = []discovery.EndpointPort{{
						Name:     pointer.String("p80"),
						Port:     pointer.Int32(80),
						Protocol: &sctpProtocol,
					}}
				}),
			},
			nodeIPs: []net.IP{
				netutils.ParseIPSloppy("2001:db8::1:1"),
				netutils.ParseIPSloppy("2001:db8::1:2"),
			},
			nodePortAddresses: []string{},
			expectedIPVS: &ipvstest.FakeIPVS{
				Services: map[ipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "2001:db8::1:1",
						Port:     3001,
						Protocol: "SCTP",
					}: {
						Address:   netutils.ParseIPSloppy("2001:db8::1:1"),
						Protocol:  "SCTP",
						Port:      uint16(3001),
						Scheduler: "rr",
					},
					{
						IP:       "2001:db8::1:2",
						Port:     3001,
						Protocol: "SCTP",
					}: {
						Address:   netutils.ParseIPSloppy("2001:db8::1:2"),
						Protocol:  "SCTP",
						Port:      uint16(3001),
						Scheduler: "rr",
					},
					{
						IP:       "2020::1",
						Port:     80,
						Protocol: "SCTP",
					}: {
						Address:   netutils.ParseIPSloppy("2020::1"),
						Protocol:  "SCTP",
						Port:      uint16(80),
						Scheduler: "rr",
					},
				},
				Destinations: map[ipvstest.ServiceKey][]*utilipvs.RealServer{
					{
						IP:       "2001:db8::1:1",
						Port:     3001,
						Protocol: "SCTP",
					}: {
						{
							Address: netutils.ParseIPSloppy("2001::1"),
							Port:    uint16(80),
							Weight:  1,
						},
					},
					{
						IP:       "2001:db8::1:2",
						Port:     3001,
						Protocol: "SCTP",
					}: {
						{
							Address: netutils.ParseIPSloppy("2001::1"),
							Port:    uint16(80),
							Weight:  1,
						},
					},
					{
						IP:       "2020::1",
						Port:     80,
						Protocol: "SCTP",
					}: {
						{
							Address: netutils.ParseIPSloppy("2001::1"),
							Port:    uint16(80),
							Weight:  1,
						},
					},
				},
			},
			expectedIPSets: netlinktest.ExpectedIPSet{
				kubeNodePortSetSCTP: {
					{
						IP:       "2001:db8::1:1",
						Port:     3001,
						Protocol: strings.ToLower(string(v1.ProtocolSCTP)),
						SetType:  utilipset.HashIPPort,
					},
					{
						IP:       "2001:db8::1:2",
						Port:     3001,
						Protocol: strings.ToLower(string(v1.ProtocolSCTP)),
						SetType:  utilipset.HashIPPort,
					},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ipt := iptablestest.NewFake()
			ipvs := ipvstest.NewFake()
			ipset := ipsettest.NewFake(testIPSetVersion)
			fp := NewFakeProxier(ipt, ipvs, ipset, test.nodeIPs, nil, v1.IPv6Protocol)
			fp.nodePortAddresses = test.nodePortAddresses

			makeServiceMap(fp, test.services...)
			populateEndpointSlices(fp, test.endpoints...)

			fp.syncProxyRules()

			if !reflect.DeepEqual(ipvs, test.expectedIPVS) {
				t.Logf("actual ipvs state: %+v", ipvs)
				t.Logf("expected ipvs state: %+v", test.expectedIPVS)
				t.Errorf("unexpected IPVS state")
			}

			if test.expectedIPSets != nil {
				checkIPSet(t, fp, test.expectedIPSets)
			}

			if test.expectedIptablesChains != nil {
				checkIptables(t, ipt, test.expectedIptablesChains)
			}
		})
	}
}

func TestIPv4Proxier(t *testing.T) {
	tcpProtocol := v1.ProtocolTCP
	tests := []struct {
		name         string
		services     []*v1.Service
		endpoints    []*discovery.EndpointSlice
		expectedIPVS *ipvstest.FakeIPVS
	}{
		{
			name: "2 services with Cluster IP, each with endpoints",
			services: []*v1.Service{
				makeTestService("ns1", "svc1", func(svc *v1.Service) {
					svc.Spec.ClusterIP = "10.20.30.41"
					svc.Spec.Ports = []v1.ServicePort{{
						Name:     "p80",
						Port:     int32(80),
						Protocol: v1.ProtocolTCP,
					}}
				}),
				makeTestService("ns2", "svc2", func(svc *v1.Service) {
					svc.Spec.ClusterIP = "1002:ab8::2:1"
					svc.Spec.Ports = []v1.ServicePort{{
						Name:     "p8080",
						Port:     int32(8080),
						Protocol: v1.ProtocolTCP,
					}}
				}),
			},
			endpoints: []*discovery.EndpointSlice{
				makeTestEndpointSlice("ns1", "svc1", 1, func(eps *discovery.EndpointSlice) {
					eps.AddressType = discovery.AddressTypeIPv4
					eps.Endpoints = []discovery.Endpoint{{
						Addresses: []string{"10.180.0.1"},
					}}
					eps.Ports = []discovery.EndpointPort{{
						Name:     pointer.String("p80"),
						Port:     pointer.Int32(80),
						Protocol: &tcpProtocol,
					}}
				}),
				makeTestEndpointSlice("ns2", "svc2", 1, func(eps *discovery.EndpointSlice) {
					eps.AddressType = discovery.AddressTypeIPv6
					eps.Endpoints = []discovery.Endpoint{{
						Addresses: []string{"1009:ab8::5:6"},
					}}
					eps.Ports = []discovery.EndpointPort{{
						Name:     pointer.String("p8080"),
						Port:     pointer.Int32(8080),
						Protocol: &tcpProtocol,
					}}
				}),
			},
			expectedIPVS: &ipvstest.FakeIPVS{
				Services: map[ipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "10.20.30.41",
						Port:     80,
						Protocol: "TCP",
					}: {
						Address:   netutils.ParseIPSloppy("10.20.30.41"),
						Protocol:  "TCP",
						Port:      uint16(80),
						Scheduler: "rr",
					},
				},
				Destinations: map[ipvstest.ServiceKey][]*utilipvs.RealServer{
					{
						IP:       "10.20.30.41",
						Port:     80,
						Protocol: "TCP",
					}: {
						{
							Address: netutils.ParseIPSloppy("10.180.0.1"),
							Port:    uint16(80),
							Weight:  1,
						},
					},
				},
			},
		},
		{
			name: "cluster IP service with no endpoints",
			services: []*v1.Service{
				makeTestService("ns1", "svc1", func(svc *v1.Service) {
					svc.Spec.ClusterIP = "10.20.30.41"
					svc.Spec.Ports = []v1.ServicePort{{
						Name:     "p80",
						Port:     int32(80),
						Protocol: v1.ProtocolTCP,
					}}
				}),
			},
			endpoints: []*discovery.EndpointSlice{},
			expectedIPVS: &ipvstest.FakeIPVS{
				Services: map[ipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "10.20.30.41",
						Port:     80,
						Protocol: "TCP",
					}: {
						Address:   netutils.ParseIPSloppy("10.20.30.41"),
						Protocol:  "TCP",
						Port:      uint16(80),
						Scheduler: "rr",
					},
				},
				Destinations: map[ipvstest.ServiceKey][]*utilipvs.RealServer{
					{
						IP:       "10.20.30.41",
						Port:     80,
						Protocol: "TCP",
					}: {},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ipt := iptablestest.NewFake()
			ipvs := ipvstest.NewFake()
			ipset := ipsettest.NewFake(testIPSetVersion)
			fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, v1.IPv4Protocol)

			makeServiceMap(fp, test.services...)
			populateEndpointSlices(fp, test.endpoints...)

			fp.syncProxyRules()

			if !reflect.DeepEqual(ipvs, test.expectedIPVS) {
				t.Logf("actual ipvs state: %v", ipvs)
				t.Logf("expected ipvs state: %v", test.expectedIPVS)
				t.Errorf("unexpected IPVS state")
			}
		})
	}
}

func TestIPv6Proxier(t *testing.T) {
	tcpProtocol := v1.ProtocolTCP
	tests := []struct {
		name         string
		services     []*v1.Service
		endpoints    []*discovery.EndpointSlice
		expectedIPVS *ipvstest.FakeIPVS
	}{
		{
			name: "2 services with Cluster IP, each with endpoints",
			services: []*v1.Service{
				makeTestService("ns1", "svc1", func(svc *v1.Service) {
					svc.Spec.ClusterIP = "10.20.30.41"
					svc.Spec.Ports = []v1.ServicePort{{
						Name:     "p80",
						Port:     int32(80),
						Protocol: v1.ProtocolTCP,
					}}
				}),
				makeTestService("ns2", "svc2", func(svc *v1.Service) {
					svc.Spec.ClusterIP = "1002:ab8::2:1"
					svc.Spec.Ports = []v1.ServicePort{{
						Name:     "p8080",
						Port:     int32(8080),
						Protocol: v1.ProtocolTCP,
					}}
				}),
			},
			endpoints: []*discovery.EndpointSlice{
				makeTestEndpointSlice("ns1", "svc1", 1, func(eps *discovery.EndpointSlice) {
					eps.AddressType = discovery.AddressTypeIPv4
					eps.Endpoints = []discovery.Endpoint{{
						Addresses: []string{"10.180.0.1"},
					}}
					eps.Ports = []discovery.EndpointPort{{
						Name:     pointer.String("p80"),
						Port:     pointer.Int32(80),
						Protocol: &tcpProtocol,
					}}
				}),
				makeTestEndpointSlice("ns2", "svc2", 1, func(eps *discovery.EndpointSlice) {
					eps.AddressType = discovery.AddressTypeIPv6
					eps.Endpoints = []discovery.Endpoint{{
						Addresses: []string{"1009:ab8::5:6"},
					}}
					eps.Ports = []discovery.EndpointPort{{
						Name:     pointer.String("p8080"),
						Port:     pointer.Int32(8080),
						Protocol: &tcpProtocol,
					}}
				}),
			},
			expectedIPVS: &ipvstest.FakeIPVS{
				Services: map[ipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "1002:ab8::2:1",
						Port:     8080,
						Protocol: "TCP",
					}: {
						Address:   netutils.ParseIPSloppy("1002:ab8::2:1"),
						Protocol:  "TCP",
						Port:      uint16(8080),
						Scheduler: "rr",
					},
				},
				Destinations: map[ipvstest.ServiceKey][]*utilipvs.RealServer{
					{
						IP:       "1002:ab8::2:1",
						Port:     8080,
						Protocol: "TCP",
					}: {
						{
							Address: netutils.ParseIPSloppy("1009:ab8::5:6"),
							Port:    uint16(8080),
							Weight:  1,
						},
					},
				},
			},
		},
		{
			name: "cluster IP service with no endpoints",
			services: []*v1.Service{
				makeTestService("ns1", "svc1", func(svc *v1.Service) {
					svc.Spec.ClusterIP = "2001::1"
					svc.Spec.Ports = []v1.ServicePort{{
						Name:     "p80",
						Port:     int32(80),
						Protocol: v1.ProtocolTCP,
					}}
				}),
			},
			endpoints: []*discovery.EndpointSlice{},
			expectedIPVS: &ipvstest.FakeIPVS{
				Services: map[ipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "2001::1",
						Port:     80,
						Protocol: "TCP",
					}: {
						Address:   netutils.ParseIPSloppy("2001::1"),
						Protocol:  "TCP",
						Port:      uint16(80),
						Scheduler: "rr",
					},
				},
				Destinations: map[ipvstest.ServiceKey][]*utilipvs.RealServer{
					{
						IP:       "2001::1",
						Port:     80,
						Protocol: "TCP",
					}: {},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ipt := iptablestest.NewFake()
			ipvs := ipvstest.NewFake()
			ipset := ipsettest.NewFake(testIPSetVersion)
			fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, v1.IPv6Protocol)

			makeServiceMap(fp, test.services...)
			populateEndpointSlices(fp, test.endpoints...)

			fp.syncProxyRules()

			if !reflect.DeepEqual(ipvs, test.expectedIPVS) {
				t.Logf("actual ipvs state: %v", ipvs)
				t.Logf("expected ipvs state: %v", test.expectedIPVS)
				t.Errorf("unexpected IPVS state")
			}
		})
	}
}

func TestMasqueradeRule(t *testing.T) {
	for _, testcase := range []bool{false, true} {
		ipt := iptablestest.NewFake().SetHasRandomFully(testcase)
		ipvs := ipvstest.NewFake()
		ipset := ipsettest.NewFake(testIPSetVersion)
		fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, v1.IPv4Protocol)
		makeServiceMap(fp)
		fp.syncProxyRules()

		buf := bytes.NewBuffer(nil)
		_ = ipt.SaveInto(utiliptables.TableNAT, buf)
		natRules := strings.Split(string(buf.Bytes()), "\n")
		var hasMasqueradeJump, hasMasqRandomFully bool
		for _, line := range natRules {
			rule, _ := iptablestest.ParseRule(line, false)
			if rule != nil && rule.Chain == kubePostroutingChain && rule.Jump != nil && rule.Jump.Value == "MASQUERADE" {
				hasMasqueradeJump = true
				if rule.RandomFully != nil {
					hasMasqRandomFully = true
				}
				break
			}
		}

		if !hasMasqueradeJump {
			t.Errorf("Failed to find -j MASQUERADE in %s chain", kubePostroutingChain)
		}
		if hasMasqRandomFully != testcase {
			probs := map[bool]string{false: "found", true: "did not find"}
			t.Errorf("%s --random-fully in -j MASQUERADE rule in %s chain for HasRandomFully()=%v", probs[testcase], kubePostroutingChain, testcase)
		}
	}
}

func TestExternalIPsNoEndpoint(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, v1.IPv4Protocol)
	svcIP := "10.20.30.41"
	svcPort := 80
	svcExternalIPs := "50.60.70.81"
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	makeServiceMap(fp,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "ClusterIP"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.ExternalIPs = []string{svcExternalIPs}
			svc.Spec.Ports = []v1.ServicePort{{
				Name:       svcPortName.Port,
				Port:       int32(svcPort),
				Protocol:   v1.ProtocolTCP,
				TargetPort: intstr.FromInt(svcPort),
			}}
		}),
	)
	fp.syncProxyRules()

	// check ipvs service and destinations
	services, err := ipvs.GetVirtualServers()
	if err != nil {
		t.Errorf("Failed to get ipvs services, err: %v", err)
	}
	if len(services) != 2 {
		t.Errorf("Expect 2 ipvs services, got %d", len(services))
	}
	found := false
	for _, svc := range services {
		if svc.Address.String() == svcExternalIPs && svc.Port == uint16(svcPort) && svc.Protocol == string(v1.ProtocolTCP) {
			found = true
			destinations, _ := ipvs.GetRealServers(svc)
			if len(destinations) != 0 {
				t.Errorf("Unexpected %d destinations, expect 0 destinations", len(destinations))
			}
			break
		}
	}
	if !found {
		t.Errorf("Expect external ip type service, got none")
	}
}

func TestExternalIPs(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, v1.IPv4Protocol)
	svcIP := "10.20.30.41"
	svcPort := 80
	svcExternalIPs := sets.NewString("50.60.70.81", "2012::51", "127.0.0.1")
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	makeServiceMap(fp,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "ClusterIP"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.ExternalIPs = svcExternalIPs.UnsortedList()
			svc.Spec.Ports = []v1.ServicePort{{
				Name:       svcPortName.Port,
				Port:       int32(svcPort),
				Protocol:   v1.ProtocolTCP,
				TargetPort: intstr.FromInt(svcPort),
			}}
		}),
	)

	epIP := "10.180.0.1"
	udpProtocol := v1.ProtocolUDP
	populateEndpointSlices(fp,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIP},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     pointer.String(svcPortName.Port),
				Port:     pointer.Int32(int32(svcPort)),
				Protocol: &udpProtocol,
			}}
		}),
	)

	fp.syncProxyRules()

	// check ipvs service and destinations
	services, err := ipvs.GetVirtualServers()
	if err != nil {
		t.Errorf("Failed to get ipvs services, err: %v", err)
	}
	if len(services) != 3 { // ipvs filters out by ipfamily
		t.Errorf("Expect 3 ipvs services, got %d", len(services))
	}
	found := false
	for _, svc := range services {
		if svcExternalIPs.Has(svc.Address.String()) && svc.Port == uint16(svcPort) && svc.Protocol == string(v1.ProtocolTCP) {
			found = true
			destinations, _ := ipvs.GetRealServers(svc)
			for _, dest := range destinations {
				if dest.Address.String() != epIP || dest.Port != uint16(svcPort) {
					t.Errorf("service Endpoint mismatch ipvs service destination")
				}
			}
			break
		}
	}
	if !found {
		t.Errorf("Expect external ip type service, got none")
	}
}

func TestOnlyLocalExternalIPs(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, v1.IPv4Protocol)
	svcIP := "10.20.30.41"
	svcPort := 80
	svcExternalIPs := sets.NewString("50.60.70.81", "2012::51", "127.0.0.1")
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	makeServiceMap(fp,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.ExternalIPs = svcExternalIPs.UnsortedList()
			svc.Spec.Ports = []v1.ServicePort{{
				Name:       svcPortName.Port,
				Port:       int32(svcPort),
				Protocol:   v1.ProtocolTCP,
				TargetPort: intstr.FromInt(svcPort),
			}}
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
		}),
	)
	epIP := "10.180.0.1"
	epIP1 := "10.180.1.1"
	thisHostname := testHostname
	otherHostname := "other-hostname"
	tcpProtocol := v1.ProtocolTCP
	populateEndpointSlices(fp,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIP},
				NodeName:  pointer.String(thisHostname),
			},
				{
					Addresses: []string{epIP1},
					NodeName:  pointer.String(otherHostname),
				}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     pointer.String(svcPortName.Port),
				Port:     pointer.Int32(int32(svcPort)),
				Protocol: &tcpProtocol,
			}}
		}),
	)

	fp.syncProxyRules()

	// check ipvs service and destinations
	services, err := ipvs.GetVirtualServers()
	if err != nil {
		t.Errorf("Failed to get ipvs services, err: %v", err)
	}
	if len(services) != 3 { // ipvs filters out by IPFamily
		t.Errorf("Expect 3 ipvs services, got %d", len(services))
	}
	found := false
	for _, svc := range services {
		if svcExternalIPs.Has(svc.Address.String()) && svc.Port == uint16(svcPort) && svc.Protocol == string(v1.ProtocolTCP) {
			found = true
			destinations, _ := ipvs.GetRealServers(svc)
			if len(destinations) != 1 {
				t.Errorf("Expect only 1 local endpoint. but got %v", len(destinations))
			}
			for _, dest := range destinations {
				if dest.Address.String() != epIP || dest.Port != uint16(svcPort) {
					t.Errorf("service Endpoint mismatch ipvs service destination")
				}
			}
			break
		}
	}
	if !found {
		t.Errorf("Expect external ip type service, got none")
	}
}

func TestLoadBalancer(t *testing.T) {
	ipt, fp := buildFakeProxier()
	svcIP := "10.20.30.41"
	svcPort := 80
	svcNodePort := 3001
	svcLBIP := "1.2.3.4"
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	makeServiceMap(fp,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "LoadBalancer"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: v1.ProtocolTCP,
				NodePort: int32(svcNodePort),
			}}
			svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
				IP: svcLBIP,
			}}
		}),
	)

	epIP := "10.180.0.1"
	udpProtocol := v1.ProtocolUDP
	populateEndpointSlices(fp,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIP},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     pointer.String(svcPortName.Port),
				Port:     pointer.Int32(int32(svcPort)),
				Protocol: &udpProtocol,
			}}
		}),
	)

	fp.syncProxyRules()

	// Expect 2 services and 1 destination
	epVS := &netlinktest.ExpectedVirtualServer{
		VSNum: 2, IP: svcLBIP, Port: uint16(svcNodePort), Protocol: string(v1.ProtocolTCP),
		RS: []netlinktest.ExpectedRealServer{{
			IP: epIP, Port: uint16(svcPort),
		}}}
	checkIPVS(t, fp, epVS)

	// check ipSet rules
	epIPSet := netlinktest.ExpectedIPSet{
		kubeLoadBalancerSet: {{
			IP:       svcLBIP,
			Port:     svcPort,
			Protocol: strings.ToLower(string(v1.ProtocolTCP)),
			SetType:  utilipset.HashIPPort,
		}},
	}
	checkIPSet(t, fp, epIPSet)

	// Check iptables chain and rules
	epIpt := netlinktest.ExpectedIptablesChain{
		string(kubeServicesChain): {{
			JumpChain: string(kubeLoadBalancerChain), MatchSet: kubeLoadBalancerSet,
		}, {
			JumpChain: string(kubeMarkMasqChain), MatchSet: kubeClusterIPSet,
		}, {
			JumpChain: string(kubeNodePortChain), MatchSet: "",
		}, {
			JumpChain: "ACCEPT", MatchSet: kubeClusterIPSet,
		}, {
			JumpChain: "ACCEPT", MatchSet: kubeLoadBalancerSet,
		}},
		string(kubeLoadBalancerSet): {{
			JumpChain: string(kubeMarkMasqChain), MatchSet: "",
		}},
	}
	checkIptables(t, ipt, epIpt)
}

func TestOnlyLocalNodePorts(t *testing.T) {
	nodeIP := netutils.ParseIPSloppy("100.101.102.103")
	ipt, fp := buildFakeProxier()

	svcIP := "10.20.30.41"
	svcPort := 80
	svcNodePort := 3001
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	makeServiceMap(fp,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: v1.ProtocolTCP,
				NodePort: int32(svcNodePort),
			}}
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
		}),
	)

	epIP := "10.180.0.1"
	epIP1 := "10.180.1.1"
	thisHostname := testHostname
	otherHostname := "other-hostname"
	tcpProtocol := v1.ProtocolTCP

	populateEndpointSlices(fp,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIP},
				NodeName:  &thisHostname,
			}, {
				Addresses: []string{epIP1},
				NodeName:  &otherHostname,
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     pointer.String(svcPortName.Port),
				Port:     pointer.Int32(int32(svcPort)),
				Protocol: &tcpProtocol,
			}}
		}),
	)

	itf := net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0}
	addrs := []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("100.101.102.103"), Mask: net.CIDRMask(24, 32)}}
	itf1 := net.Interface{Index: 1, MTU: 0, Name: "eth1", HardwareAddr: nil, Flags: 0}
	addrs1 := []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("2001:db8::"), Mask: net.CIDRMask(64, 128)}}
	fp.networkInterfacer.(*proxyutiltest.FakeNetwork).AddInterfaceAddr(&itf, addrs)
	fp.networkInterfacer.(*proxyutiltest.FakeNetwork).AddInterfaceAddr(&itf1, addrs1)
	fp.nodePortAddresses = []string{"100.101.102.0/24", "2001:db8::0/64"}

	fp.syncProxyRules()

	// Expect 2 (matching ipvs IPFamily field)  services and 1 destination
	epVS := &netlinktest.ExpectedVirtualServer{
		VSNum: 2, IP: nodeIP.String(), Port: uint16(svcNodePort), Protocol: string(v1.ProtocolTCP),
		RS: []netlinktest.ExpectedRealServer{{
			IP: epIP, Port: uint16(svcPort),
		}}}
	checkIPVS(t, fp, epVS)

	// check ipSet rules
	epEntry := &utilipset.Entry{
		Port:     svcNodePort,
		Protocol: strings.ToLower(string(v1.ProtocolTCP)),
		SetType:  utilipset.BitmapPort,
	}
	epIPSet := netlinktest.ExpectedIPSet{
		kubeNodePortSetTCP:      {epEntry},
		kubeNodePortLocalSetTCP: {epEntry},
	}
	checkIPSet(t, fp, epIPSet)

	// Check iptables chain and rules
	epIpt := netlinktest.ExpectedIptablesChain{
		string(kubeServicesChain): {{
			JumpChain: string(kubeMarkMasqChain), MatchSet: kubeClusterIPSet,
		}, {
			JumpChain: string(kubeNodePortChain), MatchSet: "",
		}, {
			JumpChain: "ACCEPT", MatchSet: kubeClusterIPSet,
		}},
		string(kubeNodePortChain): {{
			JumpChain: "RETURN", MatchSet: kubeNodePortLocalSetTCP,
		}, {
			JumpChain: string(kubeMarkMasqChain), MatchSet: kubeNodePortSetTCP,
		}, {
			JumpChain: "ACCEPT", MatchSet: kubeHealthCheckNodePortSet,
		}},
	}
	checkIptables(t, ipt, epIpt)
}

func TestHealthCheckNodePort(t *testing.T) {
	ipt, fp := buildFakeProxier()

	svcIP := "10.20.30.41"
	svcPort := 80
	svcNodePort := 3000
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	sampleSvc := makeTestService(svcPortName.Namespace, "", func(svc *v1.Service) {
		svc.Spec.Type = "LoadBalancer"
		svc.Spec.ClusterIP = svcIP
		svc.Spec.Ports = []v1.ServicePort{{
			Name:     svcPortName.Port,
			Port:     int32(svcPort),
			Protocol: v1.ProtocolTCP,
			NodePort: int32(svcNodePort),
		}}
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
	})

	svc1, svc2, invalidSvc3 := *sampleSvc, *sampleSvc, *sampleSvc
	svc1.Name, svc1.Spec.HealthCheckNodePort = "valid-svc1", 30000
	svc2.Name, svc2.Spec.HealthCheckNodePort = "valid-svc2", 30001
	// make svc3 invalid by setting external traffic policy to cluster
	invalidSvc3.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeCluster
	invalidSvc3.Name, invalidSvc3.Spec.HealthCheckNodePort = "invalid-svc3", 30002

	makeServiceMap(fp,
		&svc1,
		&svc2,
		&invalidSvc3,
	)

	itf := net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0}
	addrs := []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("100.101.102.103"), Mask: net.CIDRMask(24, 32)}}
	itf1 := net.Interface{Index: 1, MTU: 0, Name: "eth1", HardwareAddr: nil, Flags: 0}
	addrs1 := []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("2001:db8::"), Mask: net.CIDRMask(64, 128)}}
	fp.networkInterfacer.(*proxyutiltest.FakeNetwork).AddInterfaceAddr(&itf, addrs)
	fp.networkInterfacer.(*proxyutiltest.FakeNetwork).AddInterfaceAddr(&itf1, addrs1)
	fp.nodePortAddresses = []string{"100.101.102.0/24", "2001:db8::0/64"}

	fp.syncProxyRules()

	// check ipSet rules
	makeTCPEntry := func(port int) *utilipset.Entry {
		return &utilipset.Entry{
			Port:     port,
			Protocol: strings.ToLower(string(v1.ProtocolTCP)),
			SetType:  utilipset.BitmapPort,
		}
	}
	epIPSet := netlinktest.ExpectedIPSet{
		// healthcheck node port set should only contain valid HC node ports
		kubeHealthCheckNodePortSet: {makeTCPEntry(30000), makeTCPEntry(30001)},
	}
	checkIPSet(t, fp, epIPSet)

	// Check iptables chain and rules
	epIpt := netlinktest.ExpectedIptablesChain{
		string(kubeNodePortChain): {{
			JumpChain: "RETURN", MatchSet: kubeNodePortLocalSetTCP,
		}, {
			JumpChain: string(kubeMarkMasqChain), MatchSet: kubeNodePortSetTCP,
		}, {
			JumpChain: "ACCEPT", MatchSet: kubeHealthCheckNodePortSet,
		}},
	}
	checkIptables(t, ipt, epIpt)
}

func TestLoadBalancerSourceRanges(t *testing.T) {
	ipt, fp := buildFakeProxier()

	svcIP := "10.20.30.41"
	svcPort := 80
	svcLBIP := "1.2.3.4"
	svcLBSource := "10.0.0.0/8"
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}
	epIP := "10.180.0.1"
	tcpProtocol := v1.ProtocolTCP

	makeServiceMap(fp,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "LoadBalancer"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: v1.ProtocolTCP,
			}}
			svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
				IP: svcLBIP,
			}}
			svc.Spec.LoadBalancerSourceRanges = []string{
				svcLBSource,
			}
		}),
	)
	populateEndpointSlices(fp,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIP},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     pointer.String(svcPortName.Port),
				Port:     pointer.Int32(int32(svcPort)),
				Protocol: &tcpProtocol,
			}}
		}),
	)

	fp.syncProxyRules()

	// Check ipvs service and destinations
	epVS := &netlinktest.ExpectedVirtualServer{
		VSNum: 2, IP: svcLBIP, Port: uint16(svcPort), Protocol: string(v1.ProtocolTCP),
		RS: []netlinktest.ExpectedRealServer{{
			IP: epIP, Port: uint16(svcPort),
		}}}
	checkIPVS(t, fp, epVS)

	// Check ipset entry
	epIPSet := netlinktest.ExpectedIPSet{
		kubeLoadBalancerSet: {{
			IP:       svcLBIP,
			Port:     svcPort,
			Protocol: strings.ToLower(string(v1.ProtocolTCP)),
			SetType:  utilipset.HashIPPort,
		}},
		kubeLoadBalancerFWSet: {{
			IP:       svcLBIP,
			Port:     svcPort,
			Protocol: strings.ToLower(string(v1.ProtocolTCP)),
			SetType:  utilipset.HashIPPort,
		}},
		kubeLoadBalancerSourceCIDRSet: {{
			IP:       svcLBIP,
			Port:     svcPort,
			Protocol: strings.ToLower(string(v1.ProtocolTCP)),
			Net:      svcLBSource,
			SetType:  utilipset.HashIPPortNet,
		}},
	}
	checkIPSet(t, fp, epIPSet)

	// Check iptables chain and rules
	epIpt := netlinktest.ExpectedIptablesChain{
		string(kubeServicesChain): {{
			JumpChain: string(kubeLoadBalancerChain), MatchSet: kubeLoadBalancerSet,
		}, {
			JumpChain: string(kubeMarkMasqChain), MatchSet: kubeClusterIPSet,
		}, {
			JumpChain: string(kubeNodePortChain), MatchSet: "",
		}, {
			JumpChain: "ACCEPT", MatchSet: kubeClusterIPSet,
		}, {
			JumpChain: "ACCEPT", MatchSet: kubeLoadBalancerSet,
		}},
		string(kubeProxyFirewallChain): {{
			JumpChain: string(kubeSourceRangesFirewallChain), MatchSet: kubeLoadBalancerFWSet,
		}},
		string(kubeSourceRangesFirewallChain): {{
			JumpChain: "RETURN", MatchSet: kubeLoadBalancerSourceCIDRSet,
		}, {
			JumpChain: "DROP", MatchSet: "",
		}},
	}
	checkIptables(t, ipt, epIpt)
}

func TestAcceptIPVSTraffic(t *testing.T) {
	ipt, fp := buildFakeProxier()

	ingressIP := "1.2.3.4"
	externalIP := []string{"5.6.7.8"}
	svcInfos := []struct {
		svcType v1.ServiceType
		svcIP   string
		svcName string
		epIP    string
	}{
		{v1.ServiceTypeClusterIP, "10.20.30.40", "svc1", "10.180.0.1"},
		{v1.ServiceTypeLoadBalancer, "10.20.30.41", "svc2", "10.180.0.2"},
		{v1.ServiceTypeNodePort, "10.20.30.42", "svc3", "10.180.0.3"},
	}

	for _, svcInfo := range svcInfos {
		makeServiceMap(fp,
			makeTestService("ns1", svcInfo.svcName, func(svc *v1.Service) {
				svc.Spec.Type = svcInfo.svcType
				svc.Spec.ClusterIP = svcInfo.svcIP
				svc.Spec.Ports = []v1.ServicePort{{
					Name:     "p80",
					Port:     80,
					Protocol: v1.ProtocolTCP,
					NodePort: 80,
				}}
				if svcInfo.svcType == v1.ServiceTypeLoadBalancer {
					svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
						IP: ingressIP,
					}}
				}
				if svcInfo.svcType == v1.ServiceTypeClusterIP {
					svc.Spec.ExternalIPs = externalIP
				}
			}),
		)

		udpProtocol := v1.ProtocolUDP
		populateEndpointSlices(fp,
			makeTestEndpointSlice("ns1", "p80", 1, func(eps *discovery.EndpointSlice) {
				eps.Endpoints = []discovery.Endpoint{{
					Addresses: []string{svcInfo.epIP},
				}}
				eps.Ports = []discovery.EndpointPort{{
					Name:     pointer.String("p80"),
					Port:     pointer.Int32(80),
					Protocol: &udpProtocol,
				}}
			}),
		)
	}
	fp.syncProxyRules()

	// Check iptables chain and rules
	epIpt := netlinktest.ExpectedIptablesChain{
		string(kubeServicesChain): {
			{JumpChain: string(kubeLoadBalancerChain), MatchSet: kubeLoadBalancerSet},
			{JumpChain: string(kubeMarkMasqChain), MatchSet: kubeClusterIPSet},
			{JumpChain: string(kubeMarkMasqChain), MatchSet: kubeExternalIPSet},
			{JumpChain: "ACCEPT", MatchSet: kubeExternalIPSet}, // With externalTrafficOnlyArgs
			{JumpChain: "ACCEPT", MatchSet: kubeExternalIPSet}, // With dstLocalOnlyArgs
			{JumpChain: string(kubeNodePortChain), MatchSet: ""},
			{JumpChain: "ACCEPT", MatchSet: kubeClusterIPSet},
			{JumpChain: "ACCEPT", MatchSet: kubeLoadBalancerSet},
		},
	}
	checkIptables(t, ipt, epIpt)
}

func TestOnlyLocalLoadBalancing(t *testing.T) {
	ipt, fp := buildFakeProxier()

	svcIP := "10.20.30.41"
	svcPort := 80
	svcNodePort := 3001
	svcLBIP := "1.2.3.4"
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	makeServiceMap(fp,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "LoadBalancer"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: v1.ProtocolTCP,
				NodePort: int32(svcNodePort),
			}}
			svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
				IP: svcLBIP,
			}}
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
		}),
	)

	epIP := "10.180.0.1"
	epIP1 := "10.180.1.1"
	thisHostname := testHostname
	otherHostname := "other-hostname"
	tcpProtocol := v1.ProtocolTCP

	populateEndpointSlices(fp,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{
				{ // **local** endpoint address, should be added as RS
					Addresses: []string{epIP},
					NodeName:  &thisHostname,
				},
				{ // **remote** endpoint address, should not be added as RS
					Addresses: []string{epIP1},
					NodeName:  &otherHostname,
				}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     pointer.String(svcPortName.Port),
				Port:     pointer.Int32(int32(svcPort)),
				Protocol: &tcpProtocol,
			}}
		}),
	)

	fp.syncProxyRules()

	// Expect 2 services and 1 destination
	epVS := &netlinktest.ExpectedVirtualServer{
		VSNum: 2, IP: svcLBIP, Port: uint16(svcPort), Protocol: string(v1.ProtocolTCP),
		RS: []netlinktest.ExpectedRealServer{{
			IP: epIP, Port: uint16(svcPort),
		}}}
	checkIPVS(t, fp, epVS)

	// check ipSet rules
	epIPSet := netlinktest.ExpectedIPSet{
		kubeLoadBalancerSet: {{
			IP:       svcLBIP,
			Port:     svcPort,
			Protocol: strings.ToLower(string(v1.ProtocolTCP)),
			SetType:  utilipset.HashIPPort,
		}},
		kubeLoadBalancerLocalSet: {{
			IP:       svcLBIP,
			Port:     svcPort,
			Protocol: strings.ToLower(string(v1.ProtocolTCP)),
			SetType:  utilipset.HashIPPort,
		}},
	}
	checkIPSet(t, fp, epIPSet)

	// Check iptables chain and rules
	epIpt := netlinktest.ExpectedIptablesChain{
		string(kubeServicesChain): {{
			JumpChain: string(kubeLoadBalancerChain), MatchSet: kubeLoadBalancerSet,
		}, {
			JumpChain: string(kubeMarkMasqChain), MatchSet: kubeClusterIPSet,
		}, {
			JumpChain: string(kubeNodePortChain), MatchSet: "",
		}, {
			JumpChain: "ACCEPT", MatchSet: kubeClusterIPSet,
		}, {
			JumpChain: "ACCEPT", MatchSet: kubeLoadBalancerSet,
		}},
		string(kubeLoadBalancerChain): {{
			JumpChain: "RETURN", MatchSet: kubeLoadBalancerLocalSet,
		}, {
			JumpChain: string(kubeMarkMasqChain), MatchSet: "",
		}},
	}
	checkIptables(t, ipt, epIpt)
}

func addTestPort(array []v1.ServicePort, name string, protocol v1.Protocol, port, nodeport int32, targetPort int) []v1.ServicePort {
	svcPort := v1.ServicePort{
		Name:       name,
		Protocol:   protocol,
		Port:       port,
		NodePort:   nodeport,
		TargetPort: intstr.FromInt(targetPort),
	}
	return append(array, svcPort)
}

func TestBuildServiceMapAddRemove(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, v1.IPv4Protocol)

	services := []*v1.Service{
		makeTestService("somewhere-else", "cluster-ip", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.ClusterIP = "172.16.55.4"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "something", "UDP", 1234, 4321, 0)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "somethingelse", "UDP", 1235, 5321, 0)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "somesctp", "SCTP", 1236, 6321, 0)
		}),
		makeTestService("somewhere-else", "node-port", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeNodePort
			svc.Spec.ClusterIP = "172.16.55.10"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "blahblah", "UDP", 345, 678, 0)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "moreblahblah", "TCP", 344, 677, 0)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "sctpblah", "SCTP", 343, 676, 0)
		}),
		makeTestService("somewhere", "load-balancer", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
			svc.Spec.ClusterIP = "172.16.55.11"
			svc.Spec.LoadBalancerIP = "5.6.7.8"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "foobar", "UDP", 8675, 30061, 7000)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "baz", "UDP", 8676, 30062, 7001)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "sctpfoo", "SCTP", 8677, 30063, 7002)
			svc.Status.LoadBalancer = v1.LoadBalancerStatus{
				Ingress: []v1.LoadBalancerIngress{
					{IP: "10.1.2.4"},
				},
			}
		}),
		makeTestService("somewhere", "only-local-load-balancer", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
			svc.Spec.ClusterIP = "172.16.55.12"
			svc.Spec.LoadBalancerIP = "5.6.7.8"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "foobar2", "UDP", 8677, 30063, 7002)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "baz", "UDP", 8678, 30064, 7003)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "sctpbaz", "SCTP", 8679, 30065, 7004)
			svc.Status.LoadBalancer = v1.LoadBalancerStatus{
				Ingress: []v1.LoadBalancerIngress{
					{IP: "10.1.2.3"},
				},
			}
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
			svc.Spec.HealthCheckNodePort = 345
		}),
	}

	for i := range services {
		fp.OnServiceAdd(services[i])
	}
	result := fp.serviceMap.Update(fp.serviceChanges)
	if len(fp.serviceMap) != 12 {
		t.Errorf("expected service map length 12, got %v", fp.serviceMap)
	}

	// The only-local-loadbalancer ones get added
	if len(result.HCServiceNodePorts) != 1 {
		t.Errorf("expected 1 healthcheck port, got %v", result.HCServiceNodePorts)
	} else {
		nsn := makeNSN("somewhere", "only-local-load-balancer")
		if port, found := result.HCServiceNodePorts[nsn]; !found || port != 345 {
			t.Errorf("expected healthcheck port [%q]=345: got %v", nsn, result.HCServiceNodePorts)
		}
	}

	if len(result.UDPStaleClusterIP) != 0 {
		// Services only added, so nothing stale yet
		t.Errorf("expected stale UDP services length 0, got %d", len(result.UDPStaleClusterIP))
	}

	// Remove some stuff
	// oneService is a modification of services[0] with removed first port.
	oneService := makeTestService("somewhere-else", "cluster-ip", func(svc *v1.Service) {
		svc.Spec.Type = v1.ServiceTypeClusterIP
		svc.Spec.ClusterIP = "172.16.55.4"
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "somethingelse", "UDP", 1235, 5321, 0)
	})

	fp.OnServiceUpdate(services[0], oneService)
	fp.OnServiceDelete(services[1])
	fp.OnServiceDelete(services[2])
	fp.OnServiceDelete(services[3])

	result = fp.serviceMap.Update(fp.serviceChanges)
	if len(fp.serviceMap) != 1 {
		t.Errorf("expected service map length 1, got %v", fp.serviceMap)
	}

	if len(result.HCServiceNodePorts) != 0 {
		t.Errorf("expected 0 healthcheck ports, got %v", result.HCServiceNodePorts)
	}

	// All services but one were deleted. While you'd expect only the ClusterIPs
	// from the three deleted services here, we still have the ClusterIP for
	// the not-deleted service, because one of it's ServicePorts was deleted.
	expectedStaleUDPServices := []string{"172.16.55.10", "172.16.55.4", "172.16.55.11", "172.16.55.12"}
	if len(result.UDPStaleClusterIP) != len(expectedStaleUDPServices) {
		t.Errorf("expected stale UDP services length %d, got %v", len(expectedStaleUDPServices), result.UDPStaleClusterIP.List())
	}
	for _, ip := range expectedStaleUDPServices {
		if !result.UDPStaleClusterIP.Has(ip) {
			t.Errorf("expected stale UDP service service %s", ip)
		}
	}
}

func TestBuildServiceMapServiceHeadless(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, v1.IPv4Protocol)

	makeServiceMap(fp,
		makeTestService("somewhere-else", "headless", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.ClusterIP = v1.ClusterIPNone
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "rpc", "UDP", 1234, 0, 0)
		}),
		makeTestService("somewhere-else", "headless-without-port", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.ClusterIP = v1.ClusterIPNone
		}),
		makeTestService("somewhere-else", "headless-sctp", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.ClusterIP = v1.ClusterIPNone
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "sip", "SCTP", 1235, 0, 0)
		}),
	)

	// Headless service should be ignored
	result := fp.serviceMap.Update(fp.serviceChanges)
	if len(fp.serviceMap) != 0 {
		t.Errorf("expected service map length 0, got %d", len(fp.serviceMap))
	}

	// No proxied services, so no healthchecks
	if len(result.HCServiceNodePorts) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %d", len(result.HCServiceNodePorts))
	}

	if len(result.UDPStaleClusterIP) != 0 {
		t.Errorf("expected stale UDP services length 0, got %d", len(result.UDPStaleClusterIP))
	}
}

func TestBuildServiceMapServiceTypeExternalName(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, v1.IPv4Protocol)

	makeServiceMap(fp,
		makeTestService("somewhere-else", "external-name", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeExternalName
			svc.Spec.ClusterIP = "172.16.55.4" // Should be ignored
			svc.Spec.ExternalName = "foo2.bar.com"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "blah", "UDP", 1235, 5321, 0)
		}),
	)

	result := fp.serviceMap.Update(fp.serviceChanges)
	if len(fp.serviceMap) != 0 {
		t.Errorf("expected service map length 0, got %v", fp.serviceMap)
	}
	// No proxied services, so no healthchecks
	if len(result.HCServiceNodePorts) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %v", result.HCServiceNodePorts)
	}
	if len(result.UDPStaleClusterIP) != 0 {
		t.Errorf("expected stale UDP services length 0, got %v", result.UDPStaleClusterIP)
	}
}

func TestBuildServiceMapServiceUpdate(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, v1.IPv4Protocol)

	servicev1 := makeTestService("somewhere", "some-service", func(svc *v1.Service) {
		svc.Spec.Type = v1.ServiceTypeClusterIP
		svc.Spec.ClusterIP = "172.16.55.4"
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "something", "UDP", 1234, 4321, 0)
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "somethingelse", "TCP", 1235, 5321, 0)
	})
	servicev2 := makeTestService("somewhere", "some-service", func(svc *v1.Service) {
		svc.Spec.Type = v1.ServiceTypeLoadBalancer
		svc.Spec.ClusterIP = "172.16.55.4"
		svc.Spec.LoadBalancerIP = "5.6.7.8"
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "something", "UDP", 1234, 4321, 7002)
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "somethingelse", "TCP", 1235, 5321, 7003)
		svc.Status.LoadBalancer = v1.LoadBalancerStatus{
			Ingress: []v1.LoadBalancerIngress{
				{IP: "10.1.2.3"},
			},
		}
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
		svc.Spec.HealthCheckNodePort = 345
	})

	fp.OnServiceAdd(servicev1)

	result := fp.serviceMap.Update(fp.serviceChanges)
	if len(fp.serviceMap) != 2 {
		t.Errorf("expected service map length 2, got %v", fp.serviceMap)
	}
	if len(result.HCServiceNodePorts) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %v", result.HCServiceNodePorts)
	}
	if len(result.UDPStaleClusterIP) != 0 {
		// Services only added, so nothing stale yet
		t.Errorf("expected stale UDP services length 0, got %d", len(result.UDPStaleClusterIP))
	}

	// Change service to load-balancer
	fp.OnServiceUpdate(servicev1, servicev2)
	result = fp.serviceMap.Update(fp.serviceChanges)
	if len(fp.serviceMap) != 2 {
		t.Errorf("expected service map length 2, got %v", fp.serviceMap)
	}
	if len(result.HCServiceNodePorts) != 1 {
		t.Errorf("expected healthcheck ports length 1, got %v", result.HCServiceNodePorts)
	}
	if len(result.UDPStaleClusterIP) != 0 {
		t.Errorf("expected stale UDP services length 0, got %v", result.UDPStaleClusterIP.List())
	}

	// No change; make sure the service map stays the same and there are
	// no health-check changes
	fp.OnServiceUpdate(servicev2, servicev2)
	result = fp.serviceMap.Update(fp.serviceChanges)
	if len(fp.serviceMap) != 2 {
		t.Errorf("expected service map length 2, got %v", fp.serviceMap)
	}
	if len(result.HCServiceNodePorts) != 1 {
		t.Errorf("expected healthcheck ports length 1, got %v", result.HCServiceNodePorts)
	}
	if len(result.UDPStaleClusterIP) != 0 {
		t.Errorf("expected stale UDP services length 0, got %v", result.UDPStaleClusterIP.List())
	}

	// And back to ClusterIP
	fp.OnServiceUpdate(servicev2, servicev1)
	result = fp.serviceMap.Update(fp.serviceChanges)
	if len(fp.serviceMap) != 2 {
		t.Errorf("expected service map length 2, got %v", fp.serviceMap)
	}
	if len(result.HCServiceNodePorts) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %v", result.HCServiceNodePorts)
	}
	if len(result.UDPStaleClusterIP) != 0 {
		// Services only added, so nothing stale yet
		t.Errorf("expected stale UDP services length 0, got %d", len(result.UDPStaleClusterIP))
	}
}

func TestSessionAffinity(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)
	nodeIP := netutils.ParseIPSloppy("100.101.102.103")
	fp := NewFakeProxier(ipt, ipvs, ipset, []net.IP{nodeIP}, nil, v1.IPv4Protocol)
	svcIP := "10.20.30.41"
	svcPort := 80
	svcNodePort := 3001
	svcExternalIPs := "50.60.70.81"
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}
	timeoutSeconds := v1.DefaultClientIPServiceAffinitySeconds

	makeServiceMap(fp,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.ExternalIPs = []string{svcExternalIPs}
			svc.Spec.SessionAffinity = v1.ServiceAffinityClientIP
			svc.Spec.SessionAffinityConfig = &v1.SessionAffinityConfig{
				ClientIP: &v1.ClientIPConfig{
					TimeoutSeconds: &timeoutSeconds,
				},
			}
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: v1.ProtocolTCP,
				NodePort: int32(svcNodePort),
			}}
		}),
	)
	fp.syncProxyRules()

	// check ipvs service and destinations
	services, err := ipvs.GetVirtualServers()
	if err != nil {
		t.Errorf("Failed to get ipvs services, err: %v", err)
	}
	for _, svc := range services {
		if svc.Timeout != uint32(v1.DefaultClientIPServiceAffinitySeconds) {
			t.Errorf("Unexpected mismatch ipvs service session affinity timeout: %d, expected: %d", svc.Timeout, v1.DefaultClientIPServiceAffinitySeconds)
		}
	}
}

func makeServicePortName(ns, name, port string, protocol v1.Protocol) proxy.ServicePortName {
	return proxy.ServicePortName{
		NamespacedName: makeNSN(ns, name),
		Port:           port,
		Protocol:       protocol,
	}
}

func Test_updateEndpointsMap(t *testing.T) {
	var nodeName = testHostname
	udpProtocol := v1.ProtocolUDP

	emptyEndpointSlices := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1, func(*discovery.EndpointSlice) {}),
	}
	subset1 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"1.1.1.1"},
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     pointer.String("p11"),
			Port:     pointer.Int32(11),
			Protocol: &udpProtocol,
		}}
	}
	subset2 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"1.1.1.2"},
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     pointer.String("p12"),
			Port:     pointer.Int32(12),
			Protocol: &udpProtocol,
		}}
	}
	namedPortLocal := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1,
			func(eps *discovery.EndpointSlice) {
				eps.AddressType = discovery.AddressTypeIPv4
				eps.Endpoints = []discovery.Endpoint{{
					Addresses: []string{"1.1.1.1"},
					NodeName:  &nodeName,
				}}
				eps.Ports = []discovery.EndpointPort{{
					Name:     pointer.String("p11"),
					Port:     pointer.Int32(11),
					Protocol: &udpProtocol,
				}}
			}),
	}
	namedPort := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1, subset1),
	}
	namedPortRenamed := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1,
			func(eps *discovery.EndpointSlice) {
				eps.AddressType = discovery.AddressTypeIPv4
				eps.Endpoints = []discovery.Endpoint{{
					Addresses: []string{"1.1.1.1"},
				}}
				eps.Ports = []discovery.EndpointPort{{
					Name:     pointer.String("p11-2"),
					Port:     pointer.Int32(11),
					Protocol: &udpProtocol,
				}}
			}),
	}
	namedPortRenumbered := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1,
			func(eps *discovery.EndpointSlice) {
				eps.AddressType = discovery.AddressTypeIPv4
				eps.Endpoints = []discovery.Endpoint{{
					Addresses: []string{"1.1.1.1"},
				}}
				eps.Ports = []discovery.EndpointPort{{
					Name:     pointer.String("p11"),
					Port:     pointer.Int32(22),
					Protocol: &udpProtocol,
				}}
			}),
	}
	namedPortsLocalNoLocal := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1,
			func(eps *discovery.EndpointSlice) {
				eps.AddressType = discovery.AddressTypeIPv4
				eps.Endpoints = []discovery.Endpoint{{
					Addresses: []string{"1.1.1.1"},
				}, {
					Addresses: []string{"1.1.1.2"},
					NodeName:  &nodeName,
				}}
				eps.Ports = []discovery.EndpointPort{{
					Name:     pointer.String("p11"),
					Port:     pointer.Int32(11),
					Protocol: &udpProtocol,
				}, {
					Name:     pointer.String("p12"),
					Port:     pointer.Int32(12),
					Protocol: &udpProtocol,
				}}
			}),
	}
	multipleSubsets := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1, subset1),
		makeTestEndpointSlice("ns1", "ep1", 2, subset2),
	}
	subsetLocal := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"1.1.1.2"},
			NodeName:  &nodeName,
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     pointer.String("p12"),
			Port:     pointer.Int32(12),
			Protocol: &udpProtocol,
		}}
	}
	multipleSubsetsWithLocal := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1, subset1),
		makeTestEndpointSlice("ns1", "ep1", 2, subsetLocal),
	}
	subsetMultiplePortsLocal := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"1.1.1.1"},
			NodeName:  &nodeName,
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     pointer.String("p11"),
			Port:     pointer.Int32(11),
			Protocol: &udpProtocol,
		}, {
			Name:     pointer.String("p12"),
			Port:     pointer.Int32(12),
			Protocol: &udpProtocol,
		}}
	}
	subset3 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"1.1.1.3"},
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     pointer.String("p13"),
			Port:     pointer.Int32(13),
			Protocol: &udpProtocol,
		}}
	}
	multipleSubsetsMultiplePortsLocal := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1, subsetMultiplePortsLocal),
		makeTestEndpointSlice("ns1", "ep1", 2, subset3),
	}
	subsetMultipleIPsPorts1 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"1.1.1.1"},
		}, {
			Addresses: []string{"1.1.1.2"},
			NodeName:  &nodeName,
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     pointer.String("p11"),
			Port:     pointer.Int32(11),
			Protocol: &udpProtocol,
		}, {
			Name:     pointer.String("p12"),
			Port:     pointer.Int32(12),
			Protocol: &udpProtocol,
		}}
	}
	subsetMultipleIPsPorts2 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"1.1.1.3"},
		}, {
			Addresses: []string{"1.1.1.4"},
			NodeName:  &nodeName,
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     pointer.String("p13"),
			Port:     pointer.Int32(13),
			Protocol: &udpProtocol,
		}, {
			Name:     pointer.String("p14"),
			Port:     pointer.Int32(14),
			Protocol: &udpProtocol,
		}}
	}
	subsetMultipleIPsPorts3 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"2.2.2.1"},
		}, {
			Addresses: []string{"2.2.2.2"},
			NodeName:  &nodeName,
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     pointer.String("p21"),
			Port:     pointer.Int32(21),
			Protocol: &udpProtocol,
		}, {
			Name:     pointer.String("p22"),
			Port:     pointer.Int32(22),
			Protocol: &udpProtocol,
		}}
	}
	multipleSubsetsIPsPorts := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1, subsetMultipleIPsPorts1),
		makeTestEndpointSlice("ns1", "ep1", 2, subsetMultipleIPsPorts2),
		makeTestEndpointSlice("ns2", "ep2", 1, subsetMultipleIPsPorts3),
	}
	complexSubset1 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"2.2.2.2"},
			NodeName:  &nodeName,
		}, {
			Addresses: []string{"2.2.2.22"},
			NodeName:  &nodeName,
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     pointer.String("p22"),
			Port:     pointer.Int32(22),
			Protocol: &udpProtocol,
		}}
	}
	complexSubset2 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"2.2.2.3"},
			NodeName:  &nodeName,
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     pointer.String("p23"),
			Port:     pointer.Int32(23),
			Protocol: &udpProtocol,
		}}
	}
	complexSubset3 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"4.4.4.4"},
			NodeName:  &nodeName,
		}, {
			Addresses: []string{"4.4.4.5"},
			NodeName:  &nodeName,
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     pointer.String("p44"),
			Port:     pointer.Int32(44),
			Protocol: &udpProtocol,
		}}
	}
	complexSubset4 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"4.4.4.6"},
			NodeName:  &nodeName,
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     pointer.String("p45"),
			Port:     pointer.Int32(45),
			Protocol: &udpProtocol,
		}}
	}
	complexSubset5 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"1.1.1.1"},
		}, {
			Addresses: []string{"1.1.1.11"},
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     pointer.String("p11"),
			Port:     pointer.Int32(11),
			Protocol: &udpProtocol,
		}}
	}
	complexSubset6 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"1.1.1.2"},
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     pointer.String("p12"),
			Port:     pointer.Int32(12),
			Protocol: &udpProtocol,
		}, {
			Name:     pointer.String("p122"),
			Port:     pointer.Int32(122),
			Protocol: &udpProtocol,
		}}
	}
	complexSubset7 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"3.3.3.3"},
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     pointer.String("p33"),
			Port:     pointer.Int32(33),
			Protocol: &udpProtocol,
		}}
	}
	complexSubset8 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"4.4.4.4"},
			NodeName:  &nodeName,
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     pointer.String("p44"),
			Port:     pointer.Int32(44),
			Protocol: &udpProtocol,
		}}
	}
	complexBefore := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1, subset1),
		nil,
		makeTestEndpointSlice("ns2", "ep2", 1, complexSubset1),
		makeTestEndpointSlice("ns2", "ep2", 2, complexSubset2),
		nil,
		makeTestEndpointSlice("ns4", "ep4", 1, complexSubset3),
		makeTestEndpointSlice("ns4", "ep4", 2, complexSubset4),
	}
	complexAfter := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1, complexSubset5),
		makeTestEndpointSlice("ns1", "ep1", 2, complexSubset6),
		nil,
		nil,
		makeTestEndpointSlice("ns3", "ep3", 1, complexSubset7),
		makeTestEndpointSlice("ns4", "ep4", 1, complexSubset8),
		nil,
	}

	testCases := []struct {
		// previousEndpoints and currentEndpoints are used to call appropriate
		// handlers OnEndpoints* (based on whether corresponding values are nil
		// or non-nil) and must be of equal length.
		name                      string
		previousEndpoints         []*discovery.EndpointSlice
		currentEndpoints          []*discovery.EndpointSlice
		oldEndpoints              map[proxy.ServicePortName][]*proxy.BaseEndpointInfo
		expectedResult            map[proxy.ServicePortName][]*proxy.BaseEndpointInfo
		expectedStaleEndpoints    []proxy.ServiceEndpoint
		expectedStaleServiceNames map[proxy.ServicePortName]bool
		expectedHealthchecks      map[types.NamespacedName]int
	}{{
		// Case[0]: nothing
		name:                      "nothing",
		oldEndpoints:              map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{},
		expectedResult:            map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{},
		expectedStaleEndpoints:    []proxy.ServiceEndpoint{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[1]: no change, named port, local
		name:              "no change, named port, local",
		previousEndpoints: namedPortLocal,
		currentEndpoints:  namedPortLocal,
		oldEndpoints: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
		},
		expectedStaleEndpoints:    []proxy.ServiceEndpoint{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[2]: no change, multiple subsets
		name:              "no change, multiple subsets",
		previousEndpoints: multipleSubsets,
		currentEndpoints:  multipleSubsets,
		oldEndpoints: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.2:12", IsLocal: false, Ready: true, Serving: true, Terminating: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.2:12", IsLocal: false, Ready: true, Serving: true, Terminating: false},
			},
		},
		expectedStaleEndpoints:    []proxy.ServiceEndpoint{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[3]: no change, multiple subsets, multiple ports, local
		name:              "no change, multiple subsets, multiple ports, local",
		previousEndpoints: multipleSubsetsMultiplePortsLocal,
		currentEndpoints:  multipleSubsetsMultiplePortsLocal,
		oldEndpoints: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:12", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p13", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.3:13", IsLocal: false, Ready: true, Serving: true, Terminating: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:12", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p13", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.3:13", IsLocal: false, Ready: true, Serving: true, Terminating: false},
			},
		},
		expectedStaleEndpoints:    []proxy.ServiceEndpoint{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[4]: no change, multiple endpoints, subsets, IPs, and ports
		name:              "no change, multiple endpoints, subsets, IPs, and ports",
		previousEndpoints: multipleSubsetsIPsPorts,
		currentEndpoints:  multipleSubsetsIPsPorts,
		oldEndpoints: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false, Ready: true, Serving: true, Terminating: false},
				{Endpoint: "1.1.1.2:11", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:12", IsLocal: false, Ready: true, Serving: true, Terminating: false},
				{Endpoint: "1.1.1.2:12", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p13", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.3:13", IsLocal: false, Ready: true, Serving: true, Terminating: false},
				{Endpoint: "1.1.1.4:13", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p14", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.3:14", IsLocal: false, Ready: true, Serving: true, Terminating: false},
				{Endpoint: "1.1.1.4:14", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns2", "ep2", "p21", v1.ProtocolUDP): {
				{Endpoint: "2.2.2.1:21", IsLocal: false, Ready: true, Serving: true, Terminating: false},
				{Endpoint: "2.2.2.2:21", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns2", "ep2", "p22", v1.ProtocolUDP): {
				{Endpoint: "2.2.2.1:22", IsLocal: false, Ready: true, Serving: true, Terminating: false},
				{Endpoint: "2.2.2.2:22", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false, Ready: true, Serving: true, Terminating: false},
				{Endpoint: "1.1.1.2:11", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:12", IsLocal: false, Ready: true, Serving: true, Terminating: false},
				{Endpoint: "1.1.1.2:12", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p13", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.3:13", IsLocal: false, Ready: true, Serving: true, Terminating: false},
				{Endpoint: "1.1.1.4:13", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p14", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.3:14", IsLocal: false, Ready: true, Serving: true, Terminating: false},
				{Endpoint: "1.1.1.4:14", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns2", "ep2", "p21", v1.ProtocolUDP): {
				{Endpoint: "2.2.2.1:21", IsLocal: false, Ready: true, Serving: true, Terminating: false},
				{Endpoint: "2.2.2.2:21", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns2", "ep2", "p22", v1.ProtocolUDP): {
				{Endpoint: "2.2.2.1:22", IsLocal: false, Ready: true, Serving: true, Terminating: false},
				{Endpoint: "2.2.2.2:22", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
		},
		expectedStaleEndpoints:    []proxy.ServiceEndpoint{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 2,
			makeNSN("ns2", "ep2"): 1,
		},
	}, {
		// Case[5]: add an Endpoints
		name:              "add an Endpoints",
		previousEndpoints: []*discovery.EndpointSlice{nil},
		currentEndpoints:  namedPortLocal,
		oldEndpoints:      map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
		},
		expectedStaleEndpoints: []proxy.ServiceEndpoint{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[6]: remove an Endpoints
		name:              "remove an Endpoints",
		previousEndpoints: namedPortLocal,
		currentEndpoints:  []*discovery.EndpointSlice{nil},
		oldEndpoints: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{},
		expectedStaleEndpoints: []proxy.ServiceEndpoint{{
			Endpoint:        "1.1.1.1:11",
			ServicePortName: makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP),
		}},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[7]: add an IP and port
		name:              "add an IP and port",
		previousEndpoints: namedPort,
		currentEndpoints:  namedPortsLocalNoLocal,
		oldEndpoints: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false, Ready: true, Serving: true, Terminating: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false, Ready: true, Serving: true, Terminating: false},
				{Endpoint: "1.1.1.2:11", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:12", IsLocal: false, Ready: true, Serving: true, Terminating: false},
				{Endpoint: "1.1.1.2:12", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
		},
		expectedStaleEndpoints: []proxy.ServiceEndpoint{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[8]: remove an IP and port
		name:              "remove an IP and port",
		previousEndpoints: namedPortsLocalNoLocal,
		currentEndpoints:  namedPort,
		oldEndpoints: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false, Ready: true, Serving: true, Terminating: false},
				{Endpoint: "1.1.1.2:11", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:12", IsLocal: false, Ready: true, Serving: true, Terminating: false},
				{Endpoint: "1.1.1.2:12", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false, Ready: true, Serving: true, Terminating: false},
			},
		},
		expectedStaleEndpoints: []proxy.ServiceEndpoint{{
			Endpoint:        "1.1.1.2:11",
			ServicePortName: makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP),
		}, {
			Endpoint:        "1.1.1.1:12",
			ServicePortName: makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP),
		}, {
			Endpoint:        "1.1.1.2:12",
			ServicePortName: makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP),
		}},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[9]: add a subset
		name:              "add a subset",
		previousEndpoints: []*discovery.EndpointSlice{namedPort[0], nil},
		currentEndpoints:  multipleSubsetsWithLocal,
		oldEndpoints: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false, Ready: true, Serving: true, Terminating: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.2:12", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
		},
		expectedStaleEndpoints: []proxy.ServiceEndpoint{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[10]: remove a subset
		name:              "remove a subset",
		previousEndpoints: multipleSubsets,
		currentEndpoints:  []*discovery.EndpointSlice{namedPort[0], nil},
		oldEndpoints: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.2:12", IsLocal: false, Ready: true, Serving: true, Terminating: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false, Ready: true, Serving: true, Terminating: false},
			},
		},
		expectedStaleEndpoints: []proxy.ServiceEndpoint{{
			Endpoint:        "1.1.1.2:12",
			ServicePortName: makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP),
		}},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[11]: rename a port
		name:              "rename a port",
		previousEndpoints: namedPort,
		currentEndpoints:  namedPortRenamed,
		oldEndpoints: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false, Ready: true, Serving: true, Terminating: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11-2", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false, Ready: true, Serving: true, Terminating: false},
			},
		},
		expectedStaleEndpoints: []proxy.ServiceEndpoint{{
			Endpoint:        "1.1.1.1:11",
			ServicePortName: makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP),
		}},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "p11-2", v1.ProtocolUDP): true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{},
	}, {
		// Case[12]: renumber a port
		name:              "renumber a port",
		previousEndpoints: namedPort,
		currentEndpoints:  namedPortRenumbered,
		oldEndpoints: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false, Ready: true, Serving: true, Terminating: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:22", IsLocal: false, Ready: true, Serving: true, Terminating: false},
			},
		},
		expectedStaleEndpoints: []proxy.ServiceEndpoint{{
			Endpoint:        "1.1.1.1:11",
			ServicePortName: makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP),
		}},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[13]: complex add and remove
		name:              "complex add and remove",
		previousEndpoints: complexBefore,
		currentEndpoints:  complexAfter,
		oldEndpoints: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns2", "ep2", "p22", v1.ProtocolUDP): {
				{Endpoint: "2.2.2.22:22", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
				{Endpoint: "2.2.2.2:22", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns2", "ep2", "p23", v1.ProtocolUDP): {
				{Endpoint: "2.2.2.3:23", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns4", "ep4", "p44", v1.ProtocolUDP): {
				{Endpoint: "4.4.4.4:44", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
				{Endpoint: "4.4.4.5:44", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns4", "ep4", "p45", v1.ProtocolUDP): {
				{Endpoint: "4.4.4.6:45", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.11:11", IsLocal: false, Ready: true, Serving: true, Terminating: false},
				{Endpoint: "1.1.1.1:11", IsLocal: false, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.2:12", IsLocal: false, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p122", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.2:122", IsLocal: false, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns3", "ep3", "p33", v1.ProtocolUDP): {
				{Endpoint: "3.3.3.3:33", IsLocal: false, Ready: true, Serving: true, Terminating: false},
			},
			makeServicePortName("ns4", "ep4", "p44", v1.ProtocolUDP): {
				{Endpoint: "4.4.4.4:44", NodeName: nodeName, IsLocal: true, Ready: true, Serving: true, Terminating: false},
			},
		},
		expectedStaleEndpoints: []proxy.ServiceEndpoint{{
			Endpoint:        "2.2.2.2:22",
			ServicePortName: makeServicePortName("ns2", "ep2", "p22", v1.ProtocolUDP),
		}, {
			Endpoint:        "2.2.2.22:22",
			ServicePortName: makeServicePortName("ns2", "ep2", "p22", v1.ProtocolUDP),
		}, {
			Endpoint:        "2.2.2.3:23",
			ServicePortName: makeServicePortName("ns2", "ep2", "p23", v1.ProtocolUDP),
		}, {
			Endpoint:        "4.4.4.5:44",
			ServicePortName: makeServicePortName("ns4", "ep4", "p44", v1.ProtocolUDP),
		}, {
			Endpoint:        "4.4.4.6:45",
			ServicePortName: makeServicePortName("ns4", "ep4", "p45", v1.ProtocolUDP),
		}},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP):  true,
			makeServicePortName("ns1", "ep1", "p122", v1.ProtocolUDP): true,
			makeServicePortName("ns3", "ep3", "p33", v1.ProtocolUDP):  true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns4", "ep4"): 1,
		},
	}, {
		// Case[14]: change from 0 endpoint address to 1 named port
		name:              "change from 0 endpoint address to 1 named port",
		previousEndpoints: emptyEndpointSlices,
		currentEndpoints:  namedPort,
		oldEndpoints:      map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false, Ready: true, Serving: true, Terminating: false},
			},
		},
		expectedStaleEndpoints: []proxy.ServiceEndpoint{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{},
	},
	}

	for tci, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ipt := iptablestest.NewFake()
			ipvs := ipvstest.NewFake()
			ipset := ipsettest.NewFake(testIPSetVersion)
			fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, v1.IPv4Protocol)
			fp.hostname = nodeName

			// First check that after adding all previous versions of endpoints,
			// the fp.oldEndpoints is as we expect.
			for i := range tc.previousEndpoints {
				if tc.previousEndpoints[i] != nil {
					fp.OnEndpointSliceAdd(tc.previousEndpoints[i])
				}
			}
			fp.endpointsMap.Update(fp.endpointsChanges)
			compareEndpointsMaps(t, tci, fp.endpointsMap, tc.oldEndpoints)

			// Now let's call appropriate handlers to get to state we want to be.
			if len(tc.previousEndpoints) != len(tc.currentEndpoints) {
				t.Fatalf("[%d] different lengths of previous and current endpoints", tci)
			}

			for i := range tc.previousEndpoints {
				prev, curr := tc.previousEndpoints[i], tc.currentEndpoints[i]
				switch {
				case prev == nil:
					fp.OnEndpointSliceAdd(curr)
				case curr == nil:
					fp.OnEndpointSliceDelete(prev)
				default:
					fp.OnEndpointSliceUpdate(prev, curr)
				}
			}
			result := fp.endpointsMap.Update(fp.endpointsChanges)
			newMap := fp.endpointsMap
			compareEndpointsMaps(t, tci, newMap, tc.expectedResult)
			if len(result.StaleEndpoints) != len(tc.expectedStaleEndpoints) {
				t.Errorf("[%d] expected %d staleEndpoints, got %d: %v", tci, len(tc.expectedStaleEndpoints), len(result.StaleEndpoints), result.StaleEndpoints)
			}
			for _, x := range tc.expectedStaleEndpoints {
				found := false
				for _, stale := range result.StaleEndpoints {
					if stale == x {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("[%d] expected staleEndpoints[%v], but didn't find it: %v", tci, x, result.StaleEndpoints)
				}
			}
			if len(result.StaleServiceNames) != len(tc.expectedStaleServiceNames) {
				t.Errorf("[%d] expected %d staleServiceNames, got %d: %v", tci, len(tc.expectedStaleServiceNames), len(result.StaleServiceNames), result.StaleServiceNames)
			}
			for svcName := range tc.expectedStaleServiceNames {
				found := false
				for _, stale := range result.StaleServiceNames {
					if stale == svcName {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("[%d] expected staleServiceNames[%v], but didn't find it: %v", tci, svcName, result.StaleServiceNames)
				}
			}
			if !reflect.DeepEqual(result.HCEndpointsLocalIPSize, tc.expectedHealthchecks) {
				t.Errorf("[%d] expected healthchecks %v, got %v", tci, tc.expectedHealthchecks, result.HCEndpointsLocalIPSize)
			}
		})
	}
}

func compareEndpointsMaps(t *testing.T, tci int, newMap proxy.EndpointsMap, expected map[proxy.ServicePortName][]*proxy.BaseEndpointInfo) {
	if len(newMap) != len(expected) {
		t.Errorf("[%d] expected %d results, got %d: %v", tci, len(expected), len(newMap), newMap)
	}
	for x := range expected {
		if len(newMap[x]) != len(expected[x]) {
			t.Errorf("[%d] expected %d endpoints for %v, got %d", tci, len(expected[x]), x, len(newMap[x]))
		} else {
			for i := range expected[x] {
				newEp, ok := newMap[x][i].(*proxy.BaseEndpointInfo)
				if !ok {
					t.Errorf("Failed to cast proxy.BaseEndpointInfo")
					continue
				}
				if !reflect.DeepEqual(*newEp, *(expected[x][i])) {
					t.Errorf("[%d] expected new[%v][%d] to be %v, got %v", tci, x, i, expected[x][i], newEp)
				}
			}
		}
	}
}

func Test_syncService(t *testing.T) {
	testCases := []struct {
		oldVirtualServer *utilipvs.VirtualServer
		svcName          string
		newVirtualServer *utilipvs.VirtualServer
		bindAddr         bool
		bindedAddrs      sets.String
	}{
		{
			// case 0, old virtual server is same as new virtual server
			oldVirtualServer: &utilipvs.VirtualServer{
				Address:   netutils.ParseIPSloppy("1.2.3.4"),
				Protocol:  string(v1.ProtocolTCP),
				Port:      80,
				Scheduler: "rr",
				Flags:     utilipvs.FlagHashed,
			},
			svcName: "foo",
			newVirtualServer: &utilipvs.VirtualServer{
				Address:   netutils.ParseIPSloppy("1.2.3.4"),
				Protocol:  string(v1.ProtocolTCP),
				Port:      80,
				Scheduler: "rr",
				Flags:     utilipvs.FlagHashed,
			},
			bindAddr:    false,
			bindedAddrs: sets.NewString(),
		},
		{
			// case 1, old virtual server is different from new virtual server
			oldVirtualServer: &utilipvs.VirtualServer{
				Address:   netutils.ParseIPSloppy("1.2.3.4"),
				Protocol:  string(v1.ProtocolTCP),
				Port:      8080,
				Scheduler: "rr",
				Flags:     utilipvs.FlagHashed,
			},
			svcName: "bar",
			newVirtualServer: &utilipvs.VirtualServer{
				Address:   netutils.ParseIPSloppy("1.2.3.4"),
				Protocol:  string(v1.ProtocolTCP),
				Port:      8080,
				Scheduler: "rr",
				Flags:     utilipvs.FlagPersistent,
			},
			bindAddr:    false,
			bindedAddrs: sets.NewString(),
		},
		{
			// case 2, old virtual server is different from new virtual server
			oldVirtualServer: &utilipvs.VirtualServer{
				Address:   netutils.ParseIPSloppy("1.2.3.4"),
				Protocol:  string(v1.ProtocolTCP),
				Port:      8080,
				Scheduler: "rr",
				Flags:     utilipvs.FlagHashed,
			},
			svcName: "bar",
			newVirtualServer: &utilipvs.VirtualServer{
				Address:   netutils.ParseIPSloppy("1.2.3.4"),
				Protocol:  string(v1.ProtocolTCP),
				Port:      8080,
				Scheduler: "wlc",
				Flags:     utilipvs.FlagHashed,
			},
			bindAddr:    false,
			bindedAddrs: sets.NewString(),
		},
		{
			// case 3, old virtual server is nil, and create new virtual server
			oldVirtualServer: nil,
			svcName:          "baz",
			newVirtualServer: &utilipvs.VirtualServer{
				Address:   netutils.ParseIPSloppy("1.2.3.4"),
				Protocol:  string(v1.ProtocolUDP),
				Port:      53,
				Scheduler: "rr",
				Flags:     utilipvs.FlagHashed,
			},
			bindAddr:    true,
			bindedAddrs: sets.NewString(),
		},
		{
			// case 4, SCTP, old virtual server is same as new virtual server
			oldVirtualServer: &utilipvs.VirtualServer{
				Address:   netutils.ParseIPSloppy("1.2.3.4"),
				Protocol:  string(v1.ProtocolSCTP),
				Port:      80,
				Scheduler: "rr",
				Flags:     utilipvs.FlagHashed,
			},
			svcName: "foo",
			newVirtualServer: &utilipvs.VirtualServer{
				Address:   netutils.ParseIPSloppy("1.2.3.4"),
				Protocol:  string(v1.ProtocolSCTP),
				Port:      80,
				Scheduler: "rr",
				Flags:     utilipvs.FlagHashed,
			},
			bindAddr:    false,
			bindedAddrs: sets.NewString(),
		},
		{
			// case 5, old virtual server is different from new virtual server
			oldVirtualServer: &utilipvs.VirtualServer{
				Address:   netutils.ParseIPSloppy("1.2.3.4"),
				Protocol:  string(v1.ProtocolSCTP),
				Port:      8080,
				Scheduler: "rr",
				Flags:     utilipvs.FlagHashed,
			},
			svcName: "bar",
			newVirtualServer: &utilipvs.VirtualServer{
				Address:   netutils.ParseIPSloppy("1.2.3.4"),
				Protocol:  string(v1.ProtocolSCTP),
				Port:      8080,
				Scheduler: "rr",
				Flags:     utilipvs.FlagPersistent,
			},
			bindAddr:    false,
			bindedAddrs: sets.NewString(),
		},
		{
			// case 6, old virtual server is different from new virtual server
			oldVirtualServer: &utilipvs.VirtualServer{
				Address:   netutils.ParseIPSloppy("1.2.3.4"),
				Protocol:  string(v1.ProtocolSCTP),
				Port:      8080,
				Scheduler: "rr",
				Flags:     utilipvs.FlagHashed,
			},
			svcName: "bar",
			newVirtualServer: &utilipvs.VirtualServer{
				Address:   netutils.ParseIPSloppy("1.2.3.4"),
				Protocol:  string(v1.ProtocolSCTP),
				Port:      8080,
				Scheduler: "wlc",
				Flags:     utilipvs.FlagHashed,
			},
			bindAddr:    false,
			bindedAddrs: sets.NewString(),
		},
		{
			// case 7, old virtual server is nil, and create new virtual server
			oldVirtualServer: nil,
			svcName:          "baz",
			newVirtualServer: &utilipvs.VirtualServer{
				Address:   netutils.ParseIPSloppy("1.2.3.4"),
				Protocol:  string(v1.ProtocolSCTP),
				Port:      53,
				Scheduler: "rr",
				Flags:     utilipvs.FlagHashed,
			},
			bindAddr:    true,
			bindedAddrs: sets.NewString(),
		},
		{
			// case 8, virtual server address already binded, skip sync
			oldVirtualServer: &utilipvs.VirtualServer{
				Address:   netutils.ParseIPSloppy("1.2.3.4"),
				Protocol:  string(v1.ProtocolSCTP),
				Port:      53,
				Scheduler: "rr",
				Flags:     utilipvs.FlagHashed,
			},
			svcName: "baz",
			newVirtualServer: &utilipvs.VirtualServer{
				Address:   netutils.ParseIPSloppy("1.2.3.4"),
				Protocol:  string(v1.ProtocolSCTP),
				Port:      53,
				Scheduler: "rr",
				Flags:     utilipvs.FlagHashed,
			},
			bindAddr:    true,
			bindedAddrs: sets.NewString("1.2.3.4"),
		},
	}

	for i := range testCases {
		ipt := iptablestest.NewFake()
		ipvs := ipvstest.NewFake()
		ipset := ipsettest.NewFake(testIPSetVersion)
		proxier := NewFakeProxier(ipt, ipvs, ipset, nil, nil, v1.IPv4Protocol)

		proxier.netlinkHandle.EnsureDummyDevice(defaultDummyDevice)
		if testCases[i].oldVirtualServer != nil {
			if err := proxier.ipvs.AddVirtualServer(testCases[i].oldVirtualServer); err != nil {
				t.Errorf("Case [%d], unexpected add IPVS virtual server error: %v", i, err)
			}
		}
		if err := proxier.syncService(testCases[i].svcName, testCases[i].newVirtualServer, testCases[i].bindAddr, testCases[i].bindedAddrs); err != nil {
			t.Errorf("Case [%d], unexpected sync IPVS virtual server error: %v", i, err)
		}
		// check
		list, err := proxier.ipvs.GetVirtualServers()
		if err != nil {
			t.Errorf("Case [%d], unexpected list IPVS virtual server error: %v", i, err)
		}
		if len(list) != 1 {
			t.Errorf("Case [%d], expect %d virtual servers, got %d", i, 1, len(list))
			continue
		}
		if !list[0].Equal(testCases[i].newVirtualServer) {
			t.Errorf("Case [%d], unexpected mismatch, expect: %#v, got: %#v", i, testCases[i].newVirtualServer, list[0])
		}
	}
}

func buildFakeProxier() (*iptablestest.FakeIPTables, *Proxier) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)
	return ipt, NewFakeProxier(ipt, ipvs, ipset, nil, nil, v1.IPv4Protocol)
}

func getRules(ipt *iptablestest.FakeIPTables, chain utiliptables.Chain) []*iptablestest.Rule {
	var rules []*iptablestest.Rule

	buf := bytes.NewBuffer(nil)
	_ = ipt.SaveInto(utiliptables.TableNAT, buf)
	_ = ipt.SaveInto(utiliptables.TableFilter, buf)
	lines := strings.Split(string(buf.Bytes()), "\n")
	for _, l := range lines {
		if !strings.HasPrefix(l, "-A ") {
			continue
		}
		rule, _ := iptablestest.ParseRule(l, false)
		if rule != nil && rule.Chain == chain {
			rules = append(rules, rule)
		}
	}
	return rules
}

// checkIptables to check expected iptables chain and rules. The got rules must have same number and order as the
// expected rules.
func checkIptables(t *testing.T, ipt *iptablestest.FakeIPTables, epIpt netlinktest.ExpectedIptablesChain) {
	for epChain, epRules := range epIpt {
		rules := getRules(ipt, utiliptables.Chain(epChain))
		if len(rules) != len(epRules) {
			t.Errorf("Expected %d iptables rule in chain %s, got %d", len(epRules), epChain, len(rules))
			continue
		}
		for i, epRule := range epRules {
			rule := rules[i]
			if rule.Jump == nil || rule.Jump.Value != epRule.JumpChain {
				t.Errorf("Expected MatchSet=%s JumpChain=%s, got %s", epRule.MatchSet, epRule.JumpChain, rule.Raw)
			}
			if (epRule.MatchSet == "" && rule.MatchSet != nil) || (epRule.MatchSet != "" && (rule.MatchSet == nil || rule.MatchSet.Value != epRule.MatchSet)) {
				t.Errorf("Expected MatchSet=%s JumpChain=%s, got %s", epRule.MatchSet, epRule.JumpChain, rule.Raw)
			}
		}
	}
}

// checkIPSet to check expected ipset and entries
func checkIPSet(t *testing.T, fp *Proxier, ipSet netlinktest.ExpectedIPSet) {
	for set, entries := range ipSet {
		ents, err := fp.ipset.ListEntries(set)
		if err != nil || len(ents) != len(entries) {
			t.Errorf("Check ipset entries failed for ipset: %q, expect %d, got %d", set, len(entries), len(ents))
			continue
		}
		expectedEntries := []string{}
		for _, entry := range entries {
			expectedEntries = append(expectedEntries, entry.String())
		}
		sort.Strings(ents)
		sort.Strings(expectedEntries)
		if !reflect.DeepEqual(ents, expectedEntries) {
			t.Errorf("Check ipset entries failed for ipset: %q", set)
		}
	}
}

// checkIPVS to check expected ipvs service and destination
func checkIPVS(t *testing.T, fp *Proxier, vs *netlinktest.ExpectedVirtualServer) {
	t.Helper()
	services, err := fp.ipvs.GetVirtualServers()
	if err != nil {
		t.Errorf("Failed to get ipvs services, err: %v", err)
	}
	if len(services) != vs.VSNum {
		t.Errorf("Expect %d ipvs services, got %d", vs.VSNum, len(services))
	}
	for _, svc := range services {
		if svc.Address.String() == vs.IP && svc.Port == vs.Port && svc.Protocol == vs.Protocol {
			destinations, _ := fp.ipvs.GetRealServers(svc)
			if len(destinations) != len(vs.RS) {
				t.Errorf("Expected %d destinations, got %d destinations", len(vs.RS), len(destinations))
			}
			if len(vs.RS) == 1 {
				if destinations[0].Address.String() != vs.RS[0].IP || destinations[0].Port != vs.RS[0].Port {
					t.Errorf("Unexpected mismatch destinations")
				}
			}
		}
	}
}

func TestCleanLegacyService(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)
	excludeCIDRs, _ := netutils.ParseCIDRs([]string{"3.3.3.0/24", "4.4.4.0/24"})
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, excludeCIDRs, v1.IPv4Protocol)

	// All ipvs services that were processed in the latest sync loop.
	activeServices := map[string]bool{"ipvs0": true, "ipvs1": true}
	// All ipvs services in the system.
	currentServices := map[string]*utilipvs.VirtualServer{
		// Created by kube-proxy.
		"ipvs0": {
			Address:   netutils.ParseIPSloppy("1.1.1.1"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      53,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
		// Created by kube-proxy.
		"ipvs1": {
			Address:   netutils.ParseIPSloppy("2.2.2.2"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      54,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
		// Created by an external party.
		"ipvs2": {
			Address:   netutils.ParseIPSloppy("3.3.3.3"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      55,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
		// Created by an external party.
		"ipvs3": {
			Address:   netutils.ParseIPSloppy("4.4.4.4"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      56,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
		// Created by an external party.
		"ipvs4": {
			Address:   netutils.ParseIPSloppy("5.5.5.5"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      57,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
		// Created by kube-proxy, but now stale.
		"ipvs5": {
			Address:   netutils.ParseIPSloppy("6.6.6.6"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      58,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
	}
	for v := range currentServices {
		fp.ipvs.AddVirtualServer(currentServices[v])
	}

	fp.netlinkHandle.EnsureDummyDevice(defaultDummyDevice)
	activeBindAddrs := map[string]bool{"1.1.1.1": true, "2.2.2.2": true, "3.3.3.3": true, "4.4.4.4": true}
	// This is ipv4-only so ipv6 addresses should be ignored
	currentBindAddrs := []string{"1.1.1.1", "2.2.2.2", "3.3.3.3", "4.4.4.4", "5.5.5.5", "6.6.6.6", "fd80::1:2:3", "fd80::1:2:4"}
	for i := range currentBindAddrs {
		fp.netlinkHandle.EnsureAddressBind(currentBindAddrs[i], defaultDummyDevice)
	}

	fp.cleanLegacyService(activeServices, currentServices, map[string]bool{"5.5.5.5": true, "6.6.6.6": true})
	// ipvs4 and ipvs5 should have been cleaned.
	remainingVirtualServers, _ := fp.ipvs.GetVirtualServers()
	if len(remainingVirtualServers) != 4 {
		t.Errorf("Expected number of remaining IPVS services after cleanup to be %v. Got %v", 4, len(remainingVirtualServers))
	}
	for _, vs := range remainingVirtualServers {
		// Checking that ipvs4 and ipvs5 were removed.
		if vs.Port == 57 {
			t.Errorf("Expected ipvs4 to be removed after cleanup. It still remains")
		}
		if vs.Port == 58 {
			t.Errorf("Expected ipvs5 to be removed after cleanup. It still remains")
		}
	}

	// Addresses 5.5.5.5 and 6.6.6.6 should not be bound any more, but the ipv6 addresses should remain
	remainingAddrs, _ := fp.netlinkHandle.ListBindAddress(defaultDummyDevice)
	if len(remainingAddrs) != 6 {
		t.Errorf("Expected number of remaining bound addrs after cleanup to be %v. Got %v", 6, len(remainingAddrs))
	}
	// check that address "1.1.1.1", "2.2.2.2", "3.3.3.3", "4.4.4.4" are bound, ignore ipv6 addresses
	remainingAddrsMap := make(map[string]bool)
	for _, a := range remainingAddrs {
		if netutils.ParseIPSloppy(a).To4() == nil {
			continue
		}
		remainingAddrsMap[a] = true
	}
	if !reflect.DeepEqual(activeBindAddrs, remainingAddrsMap) {
		t.Errorf("Expected remainingAddrsMap %v, got %v", activeBindAddrs, remainingAddrsMap)
	}

}

func TestCleanLegacyServiceWithRealServers(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, v1.IPv4Protocol)

	// all deleted expect ipvs2
	activeServices := map[string]bool{"ipvs2": true}
	// All ipvs services in the system.
	currentServices := map[string]*utilipvs.VirtualServer{
		"ipvs0": { // deleted with real servers
			Address:   netutils.ParseIPSloppy("1.1.1.1"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      53,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
		"ipvs1": { // deleted no real server
			Address:   netutils.ParseIPSloppy("2.2.2.2"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      54,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
		"ipvs2": { // not deleted
			Address:   netutils.ParseIPSloppy("3.3.3.3"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      54,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
	}

	// "ipvs0" has a real server, but it should still be deleted since the Service is deleted
	realServers := map[*utilipvs.VirtualServer]*utilipvs.RealServer{
		{
			Address:   netutils.ParseIPSloppy("1.1.1.1"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      53,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		}: {
			Address: netutils.ParseIPSloppy("10.180.0.1"),
			Port:    uint16(53),
			Weight:  1,
		},
	}

	for v := range currentServices {
		fp.ipvs.AddVirtualServer(currentServices[v])
	}

	for v, r := range realServers {
		fp.ipvs.AddRealServer(v, r)
	}

	fp.netlinkHandle.EnsureDummyDevice(defaultDummyDevice)
	activeBindAddrs := map[string]bool{"3.3.3.3": true}
	currentBindAddrs := []string{"1.1.1.1", "2.2.2.2", "3.3.3.3"}
	for i := range currentBindAddrs {
		fp.netlinkHandle.EnsureAddressBind(currentBindAddrs[i], defaultDummyDevice)
	}

	fp.cleanLegacyService(activeServices, currentServices, map[string]bool{"1.1.1.1": true, "2.2.2.2": true})
	remainingVirtualServers, _ := fp.ipvs.GetVirtualServers()
	if len(remainingVirtualServers) != 1 {
		t.Errorf("Expected number of remaining IPVS services after cleanup to be %v. Got %v", 1, len(remainingVirtualServers))
	}

	if remainingVirtualServers[0] != currentServices["ipvs2"] {
		t.Logf("actual virtual server: %v", remainingVirtualServers[0])
		t.Logf("expected virtual server: %v", currentServices["ipvs0"])
		t.Errorf("unexpected IPVS service")
	}

	remainingAddrs, _ := fp.netlinkHandle.ListBindAddress(defaultDummyDevice)
	if len(remainingAddrs) != 1 {
		t.Errorf("Expected number of remaining bound addrs after cleanup to be %v. Got %v", 1, len(remainingAddrs))
	}
	// check that address is "3.3.3.3"
	remainingAddrsMap := make(map[string]bool)
	for _, a := range remainingAddrs {
		if netutils.ParseIPSloppy(a).To4() == nil {
			continue
		}
		remainingAddrsMap[a] = true
	}
	if !reflect.DeepEqual(activeBindAddrs, remainingAddrsMap) {
		t.Errorf("Expected remainingAddrsMap %v, got %v", activeBindAddrs, remainingAddrsMap)
	}

}

func TestCleanLegacyRealServersExcludeCIDRs(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)
	gtm := NewGracefulTerminationManager(ipvs)
	excludeCIDRs, _ := netutils.ParseCIDRs([]string{"4.4.4.4/32"})
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, excludeCIDRs, v1.IPv4Protocol)
	fp.gracefuldeleteManager = gtm

	vs := &utilipvs.VirtualServer{
		Address:   netutils.ParseIPSloppy("4.4.4.4"),
		Protocol:  string(v1.ProtocolUDP),
		Port:      56,
		Scheduler: "rr",
		Flags:     utilipvs.FlagHashed,
	}

	fp.ipvs.AddVirtualServer(vs)

	rss := []*utilipvs.RealServer{
		{
			Address:      netutils.ParseIPSloppy("10.10.10.10"),
			Port:         56,
			ActiveConn:   0,
			InactiveConn: 0,
		},
		{
			Address:      netutils.ParseIPSloppy("11.11.11.11"),
			Port:         56,
			ActiveConn:   0,
			InactiveConn: 0,
		},
	}
	for _, rs := range rss {
		fp.ipvs.AddRealServer(vs, rs)
	}

	fp.netlinkHandle.EnsureDummyDevice(defaultDummyDevice)

	fp.netlinkHandle.EnsureAddressBind("4.4.4.4", defaultDummyDevice)

	fp.cleanLegacyService(
		map[string]bool{},
		map[string]*utilipvs.VirtualServer{"ipvs0": vs},
		map[string]bool{"4.4.4.4": true},
	)

	fp.gracefuldeleteManager.tryDeleteRs()

	remainingRealServers, _ := fp.ipvs.GetRealServers(vs)

	if len(remainingRealServers) != 2 {
		t.Errorf("Expected number of remaining IPVS real servers after cleanup should be %v. Got %v", 2, len(remainingRealServers))
	}
}

func TestCleanLegacyService6(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)
	excludeCIDRs, _ := netutils.ParseCIDRs([]string{"3000::/64", "4000::/64"})
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, excludeCIDRs, v1.IPv4Protocol)
	fp.nodeIP = netutils.ParseIPSloppy("::1")

	// All ipvs services that were processed in the latest sync loop.
	activeServices := map[string]bool{"ipvs0": true, "ipvs1": true}
	// All ipvs services in the system.
	currentServices := map[string]*utilipvs.VirtualServer{
		// Created by kube-proxy.
		"ipvs0": {
			Address:   netutils.ParseIPSloppy("1000::1"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      53,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
		// Created by kube-proxy.
		"ipvs1": {
			Address:   netutils.ParseIPSloppy("1000::2"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      54,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
		// Created by an external party.
		"ipvs2": {
			Address:   netutils.ParseIPSloppy("3000::1"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      55,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
		// Created by an external party.
		"ipvs3": {
			Address:   netutils.ParseIPSloppy("4000::1"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      56,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
		// Created by an external party.
		"ipvs4": {
			Address:   netutils.ParseIPSloppy("5000::1"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      57,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
		// Created by kube-proxy, but now stale.
		"ipvs5": {
			Address:   netutils.ParseIPSloppy("1000::6"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      58,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
	}
	for v := range currentServices {
		fp.ipvs.AddVirtualServer(currentServices[v])
	}

	fp.netlinkHandle.EnsureDummyDevice(defaultDummyDevice)
	activeBindAddrs := map[string]bool{"1000::1": true, "1000::2": true, "3000::1": true, "4000::1": true}
	// This is ipv6-only so ipv4 addresses should be ignored
	currentBindAddrs := []string{"1000::1", "1000::2", "3000::1", "4000::1", "5000::1", "1000::6", "1.1.1.1", "2.2.2.2"}
	for i := range currentBindAddrs {
		fp.netlinkHandle.EnsureAddressBind(currentBindAddrs[i], defaultDummyDevice)
	}

	fp.cleanLegacyService(activeServices, currentServices, map[string]bool{"5000::1": true, "1000::6": true})
	// ipvs4 and ipvs5 should have been cleaned.
	remainingVirtualServers, _ := fp.ipvs.GetVirtualServers()
	if len(remainingVirtualServers) != 4 {
		t.Errorf("Expected number of remaining IPVS services after cleanup to be %v. Got %v", 4, len(remainingVirtualServers))
	}
	for _, vs := range remainingVirtualServers {
		// Checking that ipvs4 and ipvs5 were removed.
		if vs.Port == 57 {
			t.Errorf("Expected ipvs4 to be removed after cleanup. It still remains")
		}
		if vs.Port == 58 {
			t.Errorf("Expected ipvs5 to be removed after cleanup. It still remains")
		}
	}

	// Addresses 5000::1 and 1000::6 should not be bound any more, but the ipv4 addresses should remain
	remainingAddrs, _ := fp.netlinkHandle.ListBindAddress(defaultDummyDevice)
	if len(remainingAddrs) != 6 {
		t.Errorf("Expected number of remaining bound addrs after cleanup to be %v. Got %v", 6, len(remainingAddrs))
	}
	// check that address "1000::1", "1000::2", "3000::1", "4000::1" are still bound, ignore ipv4 addresses
	remainingAddrsMap := make(map[string]bool)
	for _, a := range remainingAddrs {
		if netutils.ParseIPSloppy(a).To4() != nil {
			continue
		}
		remainingAddrsMap[a] = true
	}
	if !reflect.DeepEqual(activeBindAddrs, remainingAddrsMap) {
		t.Errorf("Expected remainingAddrsMap %v, got %v", activeBindAddrs, remainingAddrsMap)
	}

}

func TestMultiPortServiceBindAddr(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, v1.IPv4Protocol)

	service1 := makeTestService("ns1", "svc1", func(svc *v1.Service) {
		svc.Spec.Type = v1.ServiceTypeClusterIP
		svc.Spec.ClusterIP = "172.16.55.4"
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port1", "TCP", 1234, 0, 0)
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port2", "TCP", 1235, 0, 0)
	})
	service2 := makeTestService("ns1", "svc1", func(svc *v1.Service) {
		svc.Spec.Type = v1.ServiceTypeClusterIP
		svc.Spec.ClusterIP = "172.16.55.4"
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port1", "TCP", 1234, 0, 0)
	})
	service3 := makeTestService("ns1", "svc1", func(svc *v1.Service) {
		svc.Spec.Type = v1.ServiceTypeClusterIP
		svc.Spec.ClusterIP = "172.16.55.4"
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port1", "TCP", 1234, 0, 0)
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port2", "TCP", 1235, 0, 0)
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port3", "UDP", 1236, 0, 0)
	})

	fp.servicesSynced = true

	// first, add multi-port service1
	fp.OnServiceAdd(service1)
	fp.syncProxyRules()
	remainingAddrs, _ := fp.netlinkHandle.ListBindAddress(defaultDummyDevice)
	// should only remain address "172.16.55.4"
	if len(remainingAddrs) != 1 {
		t.Errorf("Expected number of remaining bound addrs after cleanup to be %v. Got %v", 1, len(remainingAddrs))
	}
	if remainingAddrs[0] != "172.16.55.4" {
		t.Errorf("Expected remaining address should be %s, got %s", "172.16.55.4", remainingAddrs[0])
	}

	// update multi-port service1 to single-port service2
	fp.OnServiceUpdate(service1, service2)
	fp.syncProxyRules()
	remainingAddrs, _ = fp.netlinkHandle.ListBindAddress(defaultDummyDevice)
	// should still only remain address "172.16.55.4"
	if len(remainingAddrs) != 1 {
		t.Errorf("Expected number of remaining bound addrs after cleanup to be %v. Got %v", 1, len(remainingAddrs))
	} else if remainingAddrs[0] != "172.16.55.4" {
		t.Errorf("Expected remaining address should be %s, got %s", "172.16.55.4", remainingAddrs[0])
	}

	// update single-port service2 to multi-port service3
	fp.OnServiceUpdate(service2, service3)
	fp.syncProxyRules()
	remainingAddrs, _ = fp.netlinkHandle.ListBindAddress(defaultDummyDevice)
	// should still only remain address "172.16.55.4"
	if len(remainingAddrs) != 1 {
		t.Errorf("Expected number of remaining bound addrs after cleanup to be %v. Got %v", 1, len(remainingAddrs))
	} else if remainingAddrs[0] != "172.16.55.4" {
		t.Errorf("Expected remaining address should be %s, got %s", "172.16.55.4", remainingAddrs[0])
	}

	// delete multi-port service3
	fp.OnServiceDelete(service3)
	fp.syncProxyRules()
	remainingAddrs, _ = fp.netlinkHandle.ListBindAddress(defaultDummyDevice)
	// all addresses should be unbound
	if len(remainingAddrs) != 0 {
		t.Errorf("Expected number of remaining bound addrs after cleanup to be %v. Got %v", 0, len(remainingAddrs))
	}
}

func Test_getFirstColumn(t *testing.T) {
	testCases := []struct {
		name        string
		fileContent string
		want        []string
		wantErr     bool
	}{
		{
			name: "valid content",
			fileContent: `libiscsi_tcp 28672 1 iscsi_tcp, Live 0xffffffffc07ae000
libiscsi 57344 3 ib_iser,iscsi_tcp,libiscsi_tcp, Live 0xffffffffc079a000
raid10 57344 0 - Live 0xffffffffc0597000`,
			want:    []string{"libiscsi_tcp", "libiscsi", "raid10"},
			wantErr: false,
		},
	}
	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			got, err := getFirstColumn(strings.NewReader(test.fileContent))
			if (err != nil) != test.wantErr {
				t.Errorf("getFirstColumn() error = %v, wantErr %v", err, test.wantErr)
				return
			}
			if !reflect.DeepEqual(got, test.want) {
				t.Errorf("getFirstColumn() = %v, want %v", got, test.want)
			}
		})
	}
}

// The majority of EndpointSlice specific tests are not ipvs specific and focus on
// the shared EndpointChangeTracker and EndpointSliceCache. This test ensures that the
// ipvs proxier supports translating EndpointSlices to ipvs output.
func TestEndpointSliceE2E(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, v1.IPv4Protocol)
	fp.servicesSynced = true
	fp.endpointSlicesSynced = true

	// Add initial service
	serviceName := "svc1"
	namespaceName := "ns1"
	fp.OnServiceAdd(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: serviceName, Namespace: namespaceName},
		Spec: v1.ServiceSpec{
			ClusterIP: "172.20.1.1",
			Selector:  map[string]string{"foo": "bar"},
			Ports:     []v1.ServicePort{{Name: "", TargetPort: intstr.FromInt(80), Protocol: v1.ProtocolTCP}},
		},
	})

	// Add initial endpoint slice
	tcpProtocol := v1.ProtocolTCP
	endpointSlice := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-1", serviceName),
			Namespace: namespaceName,
			Labels:    map[string]string{discovery.LabelServiceName: serviceName},
		},
		Ports: []discovery.EndpointPort{{
			Name:     pointer.String(""),
			Port:     pointer.Int32(80),
			Protocol: &tcpProtocol,
		}},
		AddressType: discovery.AddressTypeIPv4,
		Endpoints: []discovery.Endpoint{{
			Addresses:  []string{"10.0.1.1"},
			Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			NodeName:   pointer.String(testHostname),
		}, {
			Addresses:  []string{"10.0.1.2"},
			Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			NodeName:   pointer.String("node2"),
		}, {
			Addresses:  []string{"10.0.1.3"},
			Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			NodeName:   pointer.String("node3"),
		}, { // not ready endpoints should be ignored
			Addresses:  []string{"10.0.1.4"},
			Conditions: discovery.EndpointConditions{Ready: pointer.Bool(false)},
			NodeName:   pointer.String("node3"),
		}},
	}

	fp.OnEndpointSliceAdd(endpointSlice)
	fp.syncProxyRules()

	// Ensure that Proxier updates ipvs appropriately after EndpointSlice update
	assert.NotNil(t, fp.ipsetList["KUBE-LOOP-BACK"])
	activeEntries1 := fp.ipsetList["KUBE-LOOP-BACK"].activeEntries
	assert.Equal(t, 1, activeEntries1.Len(), "Expected 1 active entry in KUBE-LOOP-BACK")
	assert.Equal(t, true, activeEntries1.Has("10.0.1.1,tcp:80,10.0.1.1"), "Expected activeEntries to reference first (local) pod")
	virtualServers1, vsErr1 := ipvs.GetVirtualServers()
	assert.Nil(t, vsErr1, "Expected no error getting virtual servers")
	assert.Len(t, virtualServers1, 1, "Expected 1 virtual server")
	realServers1, rsErr1 := ipvs.GetRealServers(virtualServers1[0])
	assert.Nil(t, rsErr1, "Expected no error getting real servers")
	assert.Len(t, realServers1, 3, "Expected 3 real servers")
	assert.Equal(t, realServers1[0].String(), "10.0.1.1:80")
	assert.Equal(t, realServers1[1].String(), "10.0.1.2:80")
	assert.Equal(t, realServers1[2].String(), "10.0.1.3:80")

	fp.OnEndpointSliceDelete(endpointSlice)
	fp.syncProxyRules()

	// Ensure that Proxier updates ipvs appropriately after EndpointSlice delete
	assert.NotNil(t, fp.ipsetList["KUBE-LOOP-BACK"])
	activeEntries2 := fp.ipsetList["KUBE-LOOP-BACK"].activeEntries
	assert.Equal(t, 0, activeEntries2.Len(), "Expected 0 active entries in KUBE-LOOP-BACK")
	virtualServers2, vsErr2 := ipvs.GetVirtualServers()
	assert.Nil(t, vsErr2, "Expected no error getting virtual servers")
	assert.Len(t, virtualServers2, 1, "Expected 1 virtual server")
	realServers2, rsErr2 := ipvs.GetRealServers(virtualServers2[0])
	assert.Nil(t, rsErr2, "Expected no error getting real servers")
	assert.Len(t, realServers2, 0, "Expected 0 real servers")
}

func TestHealthCheckNodePortE2E(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, v1.IPv4Protocol)
	fp.servicesSynced = true
	fp.endpointSlicesSynced = true

	// Add initial service
	serviceName := "svc1"
	namespaceName := "ns1"

	svc := v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: serviceName, Namespace: namespaceName},
		Spec: v1.ServiceSpec{
			ClusterIP:             "172.20.1.1",
			Selector:              map[string]string{"foo": "bar"},
			Ports:                 []v1.ServicePort{{Name: "", TargetPort: intstr.FromInt(80), Protocol: v1.ProtocolTCP}},
			Type:                  "LoadBalancer",
			HealthCheckNodePort:   30000,
			ExternalTrafficPolicy: v1.ServiceExternalTrafficPolicyTypeLocal,
		},
	}
	fp.OnServiceAdd(&svc)
	fp.syncProxyRules()

	// Ensure that Proxier updates ipvs appropriately after service's being created
	assert.NotNil(t, fp.ipsetList["KUBE-HEALTH-CHECK-NODE-PORT"])
	activeEntries1 := fp.ipsetList["KUBE-HEALTH-CHECK-NODE-PORT"].activeEntries
	assert.Equal(t, 1, activeEntries1.Len(), "Expected 1 active entry in KUBE-HEALTH-CHECK-NODE-PORT")
	assert.Equal(t, true, activeEntries1.Has("30000"), "Expected activeEntries to reference hc node port in spec")

	// Update health check node port in the spec
	newSvc := svc
	newSvc.Spec.HealthCheckNodePort = 30001
	fp.OnServiceUpdate(&svc, &newSvc)
	fp.syncProxyRules()

	// Ensure that Proxier updates ipvs appropriately after service's being updated
	assert.NotNil(t, fp.ipsetList["KUBE-HEALTH-CHECK-NODE-PORT"])
	activeEntries2 := fp.ipsetList["KUBE-HEALTH-CHECK-NODE-PORT"].activeEntries
	assert.Equal(t, 1, activeEntries2.Len(), "Expected 1 active entry in KUBE-HEALTH-CHECK-NODE-PORT")
	assert.Equal(t, true, activeEntries2.Has("30001"), "Expected activeEntries to reference updated hc node port in spec")

	fp.OnServiceDelete(&svc)
	fp.syncProxyRules()

	// Ensure that Proxier updates ipvs appropriately after EndpointSlice delete
	assert.NotNil(t, fp.ipsetList["KUBE-HEALTH-CHECK-NODE-PORT"])
	activeEntries3 := fp.ipsetList["KUBE-HEALTH-CHECK-NODE-PORT"].activeEntries
	assert.Equal(t, 0, activeEntries3.Len(), "Expected 0 active entries in KUBE-HEALTH-CHECK-NODE-PORT")
}

// Test_HealthCheckNodePortWhenTerminating tests that health check node ports are not enabled when all local endpoints are terminating
func Test_HealthCheckNodePortWhenTerminating(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, v1.IPv4Protocol)
	fp.servicesSynced = true
	// fp.endpointsSynced = true
	fp.endpointSlicesSynced = true

	serviceName := "svc1"
	namespaceName := "ns1"

	fp.OnServiceAdd(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: serviceName, Namespace: namespaceName},
		Spec: v1.ServiceSpec{
			ClusterIP: "172.20.1.1",
			Selector:  map[string]string{"foo": "bar"},
			Ports:     []v1.ServicePort{{Name: "", TargetPort: intstr.FromInt(80), Protocol: v1.ProtocolTCP}},
		},
	})

	tcpProtocol := v1.ProtocolTCP
	endpointSlice := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-1", serviceName),
			Namespace: namespaceName,
			Labels:    map[string]string{discovery.LabelServiceName: serviceName},
		},
		Ports: []discovery.EndpointPort{{
			Name:     pointer.String(""),
			Port:     pointer.Int32(80),
			Protocol: &tcpProtocol,
		}},
		AddressType: discovery.AddressTypeIPv4,
		Endpoints: []discovery.Endpoint{{
			Addresses:  []string{"10.0.1.1"},
			Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			NodeName:   pointer.String(testHostname),
		}, {
			Addresses:  []string{"10.0.1.2"},
			Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
			NodeName:   pointer.String(testHostname),
		}, {
			Addresses:  []string{"10.0.1.3"},
			Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
		}, { // not ready endpoints should be ignored
			Addresses:  []string{"10.0.1.4"},
			Conditions: discovery.EndpointConditions{Ready: pointer.Bool(false)},
			NodeName:   pointer.String(testHostname),
		}},
	}

	fp.OnEndpointSliceAdd(endpointSlice)
	result := fp.endpointsMap.Update(fp.endpointsChanges)
	if len(result.HCEndpointsLocalIPSize) != 1 {
		t.Errorf("unexpected number of health check node ports, expected 1 but got: %d", len(result.HCEndpointsLocalIPSize))
	}

	// set all endpoints to terminating
	endpointSliceTerminating := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-1", serviceName),
			Namespace: namespaceName,
			Labels:    map[string]string{discovery.LabelServiceName: serviceName},
		},
		Ports: []discovery.EndpointPort{{
			Name:     pointer.String(""),
			Port:     pointer.Int32(80),
			Protocol: &tcpProtocol,
		}},
		AddressType: discovery.AddressTypeIPv4,
		Endpoints: []discovery.Endpoint{{
			Addresses: []string{"10.0.1.1"},
			Conditions: discovery.EndpointConditions{
				Ready:       pointer.Bool(false),
				Serving:     pointer.Bool(true),
				Terminating: pointer.Bool(false),
			},
			NodeName: pointer.String(testHostname),
		}, {
			Addresses: []string{"10.0.1.2"},
			Conditions: discovery.EndpointConditions{
				Ready:       pointer.Bool(false),
				Serving:     pointer.Bool(true),
				Terminating: pointer.Bool(true),
			},
			NodeName: pointer.String(testHostname),
		}, {
			Addresses: []string{"10.0.1.3"},
			Conditions: discovery.EndpointConditions{
				Ready:       pointer.Bool(false),
				Serving:     pointer.Bool(true),
				Terminating: pointer.Bool(true),
			},
			NodeName: pointer.String(testHostname),
		}, { // not ready endpoints should be ignored
			Addresses: []string{"10.0.1.4"},
			Conditions: discovery.EndpointConditions{
				Ready:       pointer.Bool(false),
				Serving:     pointer.Bool(false),
				Terminating: pointer.Bool(true),
			},
			NodeName: pointer.String(testHostname),
		}},
	}

	fp.OnEndpointSliceUpdate(endpointSlice, endpointSliceTerminating)
	result = fp.endpointsMap.Update(fp.endpointsChanges)
	if len(result.HCEndpointsLocalIPSize) != 0 {
		t.Errorf("unexpected number of health check node ports, expected 0 but got: %d", len(result.HCEndpointsLocalIPSize))
	}
}

func TestFilterCIDRs(t *testing.T) {
	var cidrList []string
	var cidrs []string
	var expected []string
	cidrs = filterCIDRs(true, []string{})
	if len(cidrs) > 0 {
		t.Errorf("An empty list produces a non-empty return %v", cidrs)
	}

	cidrList = []string{"1000::/64", "10.0.0.0/16", "11.0.0.0/16", "2000::/64"}
	expected = []string{"1000::/64", "2000::/64"}
	cidrs = filterCIDRs(true, cidrList)
	if !reflect.DeepEqual(cidrs, expected) {
		t.Errorf("cidrs %v is not expected %v", cidrs, expected)
	}

	expected = []string{"10.0.0.0/16", "11.0.0.0/16"}
	cidrs = filterCIDRs(false, cidrList)
	if !reflect.DeepEqual(cidrs, expected) {
		t.Errorf("cidrs %v is not expected %v", cidrs, expected)
	}

	cidrList = []string{"1000::/64", "2000::/64"}
	expected = []string{}
	cidrs = filterCIDRs(false, cidrList)
	if len(cidrs) > 0 {
		t.Errorf("cidrs %v is not expected %v", cidrs, expected)
	}
}

func TestCreateAndLinkKubeChain(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, v1.IPv4Protocol)
	fp.createAndLinkKubeChain()
	expectedNATChains := `:KUBE-SERVICES - [0:0]
:KUBE-POSTROUTING - [0:0]
:KUBE-NODE-PORT - [0:0]
:KUBE-LOAD-BALANCER - [0:0]
:KUBE-MARK-MASQ - [0:0]
`
	expectedFilterChains := `:KUBE-FORWARD - [0:0]
:KUBE-NODE-PORT - [0:0]
:KUBE-PROXY-FIREWALL - [0:0]
:KUBE-SOURCE-RANGES-FIREWALL - [0:0]
`
	assert.Equal(t, expectedNATChains, string(fp.natChains.Bytes()))
	assert.Equal(t, expectedFilterChains, string(fp.filterChains.Bytes()))
}

// This test ensures that the iptables proxier supports translating Endpoints to
// iptables output when internalTrafficPolicy is specified
func TestTestInternalTrafficPolicyE2E(t *testing.T) {
	type endpoint struct {
		ip       string
		hostname string
	}

	cluster := v1.ServiceInternalTrafficPolicyCluster
	local := v1.ServiceInternalTrafficPolicyLocal

	testCases := []struct {
		name                     string
		internalTrafficPolicy    *v1.ServiceInternalTrafficPolicyType
		endpoints                []endpoint
		expectVirtualServer      bool
		expectLocalEntries       bool
		expectLocalRealServerNum int
		expectLocalRealServers   []string
	}{
		{
			name:                  "internalTrafficPolicy is cluster with non-zero local endpoints",
			internalTrafficPolicy: &cluster,
			endpoints: []endpoint{
				{"10.0.1.1", testHostname},
				{"10.0.1.2", "host1"},
				{"10.0.1.3", "host2"},
			},
			expectVirtualServer:      true,
			expectLocalEntries:       true,
			expectLocalRealServerNum: 3,
			expectLocalRealServers: []string{
				"10.0.1.1:80",
				"10.0.1.2:80",
				"10.0.1.3:80",
			},
		},
		{
			name:                  "internalTrafficPolicy is cluster with zero local endpoints",
			internalTrafficPolicy: &cluster,
			endpoints: []endpoint{
				{"10.0.1.1", "host0"},
				{"10.0.1.2", "host1"},
				{"10.0.1.3", "host2"},
			},
			expectVirtualServer:      false,
			expectLocalEntries:       false,
			expectLocalRealServerNum: 3,
			expectLocalRealServers: []string{
				"10.0.1.1:80",
				"10.0.1.2:80",
				"10.0.1.3:80",
			},
		},
		{
			name:                  "internalTrafficPolicy is local with non-zero local endpoints",
			internalTrafficPolicy: &local,
			endpoints: []endpoint{
				{"10.0.1.1", testHostname},
				{"10.0.1.2", "host1"},
				{"10.0.1.3", "host2"},
			},
			expectVirtualServer:      true,
			expectLocalEntries:       true,
			expectLocalRealServerNum: 1,
			expectLocalRealServers: []string{
				"10.0.1.1:80",
			},
		},
		{
			name:                  "internalTrafficPolicy is local with zero local endpoints",
			internalTrafficPolicy: &local,
			endpoints: []endpoint{
				{"10.0.1.1", "host0"},
				{"10.0.1.2", "host1"},
				{"10.0.1.3", "host2"},
			},
			expectVirtualServer:      false,
			expectLocalEntries:       false,
			expectLocalRealServerNum: 0,
			expectLocalRealServers:   []string{},
		},
	}
	for _, tc := range testCases {
		defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ServiceInternalTrafficPolicy, true)()

		ipt := iptablestest.NewFake()
		ipvs := ipvstest.NewFake()
		ipset := ipsettest.NewFake(testIPSetVersion)
		fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, v1.IPv4Protocol)
		fp.servicesSynced = true
		// fp.endpointsSynced = true
		fp.endpointSlicesSynced = true

		// Add initial service
		serviceName := "svc1"
		namespaceName := "ns1"

		svc := &v1.Service{
			ObjectMeta: metav1.ObjectMeta{Name: serviceName, Namespace: namespaceName},
			Spec: v1.ServiceSpec{
				ClusterIP: "172.20.1.1",
				Selector:  map[string]string{"foo": "bar"},
				Ports:     []v1.ServicePort{{Name: "", TargetPort: intstr.FromInt(80), Protocol: v1.ProtocolTCP}},
			},
		}
		if tc.internalTrafficPolicy != nil {
			svc.Spec.InternalTrafficPolicy = tc.internalTrafficPolicy
		}

		fp.OnServiceAdd(svc)

		// Add initial endpoint slice
		tcpProtocol := v1.ProtocolTCP
		endpointSlice := &discovery.EndpointSlice{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("%s-1", serviceName),
				Namespace: namespaceName,
				Labels:    map[string]string{discovery.LabelServiceName: serviceName},
			},
			Ports: []discovery.EndpointPort{{
				Name:     pointer.String(""),
				Port:     pointer.Int32(80),
				Protocol: &tcpProtocol,
			}},
			AddressType: discovery.AddressTypeIPv4,
		}

		for _, ep := range tc.endpoints {
			endpointSlice.Endpoints = append(endpointSlice.Endpoints, discovery.Endpoint{
				Addresses:  []string{ep.ip},
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
				NodeName:   pointer.String(ep.hostname),
			})
		}

		fp.OnEndpointSliceAdd(endpointSlice)
		fp.syncProxyRules()

		// Ensure that Proxier updates ipvs appropriately after EndpointSlice update
		assert.NotNil(t, fp.ipsetList["KUBE-LOOP-BACK"])

		activeEntries1 := fp.ipsetList["KUBE-LOOP-BACK"].activeEntries

		if tc.expectLocalEntries {
			assert.Equal(t, 1, activeEntries1.Len(), "Expected 1 active entry in KUBE-LOOP-BACK")
		} else {
			assert.Equal(t, 0, activeEntries1.Len(), "Expected no active entry in KUBE-LOOP-BACK")
		}

		if tc.expectVirtualServer {
			virtualServers1, vsErr1 := ipvs.GetVirtualServers()
			assert.Nil(t, vsErr1, "Expected no error getting virtual servers")

			assert.Len(t, virtualServers1, 1, "Expected 1 virtual server")
			realServers1, rsErr1 := ipvs.GetRealServers(virtualServers1[0])
			assert.Nil(t, rsErr1, "Expected no error getting real servers")

			assert.Len(t, realServers1, tc.expectLocalRealServerNum, fmt.Sprintf("Expected %d real servers", tc.expectLocalRealServerNum))
			for i := 0; i < tc.expectLocalRealServerNum; i++ {
				assert.Equal(t, realServers1[i].String(), tc.expectLocalRealServers[i])
			}
		}

		fp.OnEndpointSliceDelete(endpointSlice)
		fp.syncProxyRules()

		// Ensure that Proxier updates ipvs appropriately after EndpointSlice delete
		assert.NotNil(t, fp.ipsetList["KUBE-LOOP-BACK"])
		activeEntries3 := fp.ipsetList["KUBE-LOOP-BACK"].activeEntries
		assert.Equal(t, 0, activeEntries3.Len(), "Expected 0 active entries in KUBE-LOOP-BACK")
		virtualServers2, vsErr2 := ipvs.GetVirtualServers()
		assert.Nil(t, vsErr2, "Expected no error getting virtual servers")
		assert.Len(t, virtualServers2, 1, "Expected 1 virtual server")
		realServers2, rsErr2 := ipvs.GetRealServers(virtualServers2[0])
		assert.Nil(t, rsErr2, "Expected no error getting real servers")
		assert.Len(t, realServers2, 0, "Expected 0 real servers")
	}
}

// Test_EndpointSliceReadyAndTerminatingCluster tests that when there are ready and ready + terminating
// endpoints and the traffic policy is "Cluster", only the ready endpoints are used.
func Test_EndpointSliceReadyAndTerminatingCluster(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ProxyTerminatingEndpoints, true)()

	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, v1.IPv4Protocol)
	fp.servicesSynced = true
	// fp.endpointsSynced = true
	fp.endpointSlicesSynced = true

	clusterInternalTrafficPolicy := v1.ServiceInternalTrafficPolicyCluster

	serviceName := "svc1"
	// Add initial service
	namespaceName := "ns1"
	fp.OnServiceAdd(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: serviceName, Namespace: namespaceName},
		Spec: v1.ServiceSpec{
			ClusterIP:             "172.20.1.1",
			Selector:              map[string]string{"foo": "bar"},
			Type:                  v1.ServiceTypeNodePort,
			ExternalTrafficPolicy: v1.ServiceExternalTrafficPolicyTypeCluster,
			InternalTrafficPolicy: &clusterInternalTrafficPolicy,
			ExternalIPs: []string{
				"1.2.3.4",
			},
			Ports: []v1.ServicePort{
				{
					Name:       "",
					Port:       80,
					TargetPort: intstr.FromInt(80),
					Protocol:   v1.ProtocolTCP,
				},
			},
		},
	})

	// Add initial endpoint slice
	tcpProtocol := v1.ProtocolTCP
	endpointSlice := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-1", serviceName),
			Namespace: namespaceName,
			Labels:    map[string]string{discovery.LabelServiceName: serviceName},
		},
		Ports: []discovery.EndpointPort{{
			Name:     pointer.String(""),
			Port:     pointer.Int32(80),
			Protocol: &tcpProtocol,
		}},
		AddressType: discovery.AddressTypeIPv4,
		Endpoints: []discovery.Endpoint{
			{
				Addresses: []string{"10.0.1.1"},
				Conditions: discovery.EndpointConditions{
					Ready:       pointer.Bool(true),
					Serving:     pointer.Bool(true),
					Terminating: pointer.Bool(false),
				},
				NodeName: pointer.String(testHostname),
			},
			{
				Addresses: []string{"10.0.1.2"},
				Conditions: discovery.EndpointConditions{
					Ready:       pointer.Bool(true),
					Serving:     pointer.Bool(true),
					Terminating: pointer.Bool(false),
				},
				NodeName: pointer.String(testHostname),
			},
			{
				Addresses: []string{"10.0.1.3"},
				Conditions: discovery.EndpointConditions{
					Ready:       pointer.Bool(false),
					Serving:     pointer.Bool(true),
					Terminating: pointer.Bool(true),
				},
				NodeName: pointer.String(testHostname),
			},
			{
				Addresses: []string{"10.0.1.4"},
				Conditions: discovery.EndpointConditions{
					Ready:       pointer.Bool(false),
					Serving:     pointer.Bool(false),
					Terminating: pointer.Bool(true),
				},
				NodeName: pointer.String(testHostname),
			},
			{
				Addresses: []string{"10.0.1.5"},
				Conditions: discovery.EndpointConditions{
					Ready:       pointer.Bool(true),
					Serving:     pointer.Bool(true),
					Terminating: pointer.Bool(false),
				},
				NodeName: pointer.String("another-host"),
			},
		},
	}

	fp.OnEndpointSliceAdd(endpointSlice)
	fp.syncProxyRules()

	// Ensure that Proxier updates ipvs appropriately after EndpointSlice update
	assert.NotNil(t, fp.ipsetList["KUBE-LOOP-BACK"])
	activeEntries1 := fp.ipsetList["KUBE-LOOP-BACK"].activeEntries
	assert.Equal(t, 4, activeEntries1.Len(), "Expected 4 active entry in KUBE-LOOP-BACK")
	assert.Equal(t, true, activeEntries1.Has("10.0.1.1,tcp:80,10.0.1.1"), "Expected activeEntries to reference first pod")
	assert.Equal(t, true, activeEntries1.Has("10.0.1.2,tcp:80,10.0.1.2"), "Expected activeEntries to reference second pod")
	assert.Equal(t, true, activeEntries1.Has("10.0.1.3,tcp:80,10.0.1.3"), "Expected activeEntries to reference third pod")
	assert.Equal(t, true, activeEntries1.Has("10.0.1.4,tcp:80,10.0.1.4"), "Expected activeEntries to reference fourth pod")

	virtualServers, vsErr := ipvs.GetVirtualServers()
	assert.Nil(t, vsErr, "Expected no error getting virtual servers")
	assert.Len(t, virtualServers, 2, "Expected 2 virtual server")

	var clusterIPServer, externalIPServer *utilipvs.VirtualServer
	for _, virtualServer := range virtualServers {
		if virtualServer.Address.String() == "172.20.1.1" {
			clusterIPServer = virtualServer
		}

		if virtualServer.Address.String() == "1.2.3.4" {
			externalIPServer = virtualServer
		}
	}

	// clusterIP should route to cluster-wide ready endpoints
	realServers1, rsErr1 := ipvs.GetRealServers(clusterIPServer)
	assert.Nil(t, rsErr1, "Expected no error getting real servers")
	assert.Len(t, realServers1, 3, "Expected 3 real servers")
	assert.Equal(t, realServers1[0].String(), "10.0.1.1:80")
	assert.Equal(t, realServers1[1].String(), "10.0.1.2:80")
	assert.Equal(t, realServers1[2].String(), "10.0.1.5:80")

	// externalIP should route to cluster-wide ready endpoints
	realServers2, rsErr2 := ipvs.GetRealServers(externalIPServer)
	assert.Nil(t, rsErr2, "Expected no error getting real servers")
	assert.Len(t, realServers2, 3, "Expected 3 real servers")
	assert.Equal(t, realServers2[0].String(), "10.0.1.1:80")
	assert.Equal(t, realServers2[1].String(), "10.0.1.2:80")
	assert.Equal(t, realServers1[2].String(), "10.0.1.5:80")

	fp.OnEndpointSliceDelete(endpointSlice)
	fp.syncProxyRules()

	// Ensure that Proxier updates ipvs appropriately after EndpointSlice delete
	assert.NotNil(t, fp.ipsetList["KUBE-LOOP-BACK"])
	activeEntries2 := fp.ipsetList["KUBE-LOOP-BACK"].activeEntries
	assert.Equal(t, 0, activeEntries2.Len(), "Expected 0 active entries in KUBE-LOOP-BACK")

	virtualServers, vsErr = ipvs.GetVirtualServers()
	assert.Nil(t, vsErr, "Expected no error getting virtual servers")
	assert.Len(t, virtualServers, 2, "Expected 2 virtual server")

	for _, virtualServer := range virtualServers {
		if virtualServer.Address.String() == "172.20.1.1" {
			clusterIPServer = virtualServer
		}

		if virtualServer.Address.String() == "1.2.3.4" {
			externalIPServer = virtualServer
		}
	}

	realServers1, rsErr1 = ipvs.GetRealServers(clusterIPServer)
	assert.Nil(t, rsErr1, "Expected no error getting real servers")
	assert.Len(t, realServers1, 0, "Expected 0 real servers")

	realServers2, rsErr2 = ipvs.GetRealServers(externalIPServer)
	assert.Nil(t, rsErr2, "Expected no error getting real servers")
	assert.Len(t, realServers2, 0, "Expected 0 real servers")
}

// Test_EndpointSliceReadyAndTerminatingLocal tests that when there are local ready and ready + terminating
// endpoints, only the ready endpoints are used.
func Test_EndpointSliceReadyAndTerminatingLocal(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ProxyTerminatingEndpoints, true)()

	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, v1.IPv4Protocol)
	fp.servicesSynced = true
	// fp.endpointsSynced = true
	fp.endpointSlicesSynced = true

	clusterInternalTrafficPolicy := v1.ServiceInternalTrafficPolicyCluster

	serviceName := "svc1"
	// Add initial service
	namespaceName := "ns1"
	fp.OnServiceAdd(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: serviceName, Namespace: namespaceName},
		Spec: v1.ServiceSpec{
			ClusterIP:             "172.20.1.1",
			Selector:              map[string]string{"foo": "bar"},
			Type:                  v1.ServiceTypeNodePort,
			ExternalTrafficPolicy: v1.ServiceExternalTrafficPolicyTypeLocal,
			InternalTrafficPolicy: &clusterInternalTrafficPolicy,
			ExternalIPs: []string{
				"1.2.3.4",
			},
			Ports: []v1.ServicePort{
				{
					Name:       "",
					Port:       80,
					TargetPort: intstr.FromInt(80),
					Protocol:   v1.ProtocolTCP,
				},
			},
		},
	})

	// Add initial endpoint slice
	tcpProtocol := v1.ProtocolTCP
	endpointSlice := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-1", serviceName),
			Namespace: namespaceName,
			Labels:    map[string]string{discovery.LabelServiceName: serviceName},
		},
		Ports: []discovery.EndpointPort{{
			Name:     pointer.String(""),
			Port:     pointer.Int32(80),
			Protocol: &tcpProtocol,
		}},
		AddressType: discovery.AddressTypeIPv4,
		Endpoints: []discovery.Endpoint{
			{
				Addresses: []string{"10.0.1.1"},
				Conditions: discovery.EndpointConditions{
					Ready:       pointer.Bool(true),
					Serving:     pointer.Bool(true),
					Terminating: pointer.Bool(false),
				},
				NodeName: pointer.String(testHostname),
			},
			{
				Addresses: []string{"10.0.1.2"},
				Conditions: discovery.EndpointConditions{
					Ready:       pointer.Bool(true),
					Serving:     pointer.Bool(true),
					Terminating: pointer.Bool(false),
				},
				NodeName: pointer.String(testHostname),
			},
			{
				Addresses: []string{"10.0.1.3"},
				Conditions: discovery.EndpointConditions{
					Ready:       pointer.Bool(false),
					Serving:     pointer.Bool(true),
					Terminating: pointer.Bool(true),
				},
				NodeName: pointer.String(testHostname),
			},
			{
				Addresses: []string{"10.0.1.4"},
				Conditions: discovery.EndpointConditions{
					Ready:       pointer.Bool(false),
					Serving:     pointer.Bool(false),
					Terminating: pointer.Bool(true),
				},
				NodeName: pointer.String(testHostname),
			},
			{
				Addresses: []string{"10.0.1.5"},
				Conditions: discovery.EndpointConditions{
					Ready:       pointer.Bool(true),
					Serving:     pointer.Bool(true),
					Terminating: pointer.Bool(false),
				},
				NodeName: pointer.String("another-host"),
			},
		},
	}

	fp.OnEndpointSliceAdd(endpointSlice)
	fp.syncProxyRules()

	// Ensure that Proxier updates ipvs appropriately after EndpointSlice update
	assert.NotNil(t, fp.ipsetList["KUBE-LOOP-BACK"])
	activeEntries1 := fp.ipsetList["KUBE-LOOP-BACK"].activeEntries
	assert.Equal(t, 4, activeEntries1.Len(), "Expected 3 active entry in KUBE-LOOP-BACK")
	assert.Equal(t, true, activeEntries1.Has("10.0.1.1,tcp:80,10.0.1.1"), "Expected activeEntries to reference first (local) pod")
	assert.Equal(t, true, activeEntries1.Has("10.0.1.2,tcp:80,10.0.1.2"), "Expected activeEntries to reference second (local) pod")
	assert.Equal(t, true, activeEntries1.Has("10.0.1.3,tcp:80,10.0.1.3"), "Expected activeEntries to reference second (local) pod")
	assert.Equal(t, true, activeEntries1.Has("10.0.1.4,tcp:80,10.0.1.4"), "Expected activeEntries to reference second (local) pod")

	virtualServers, vsErr := ipvs.GetVirtualServers()
	assert.Nil(t, vsErr, "Expected no error getting virtual servers")
	assert.Len(t, virtualServers, 2, "Expected 2 virtual server")

	var clusterIPServer, externalIPServer *utilipvs.VirtualServer
	for _, virtualServer := range virtualServers {
		if virtualServer.Address.String() == "172.20.1.1" {
			clusterIPServer = virtualServer
		}

		if virtualServer.Address.String() == "1.2.3.4" {
			externalIPServer = virtualServer
		}
	}

	// clusterIP should route to cluster-wide ready endpoints
	realServers1, rsErr1 := ipvs.GetRealServers(clusterIPServer)
	assert.Nil(t, rsErr1, "Expected no error getting real servers")
	assert.Len(t, realServers1, 3, "Expected 3 real servers")
	assert.Equal(t, realServers1[0].String(), "10.0.1.1:80")
	assert.Equal(t, realServers1[1].String(), "10.0.1.2:80")
	assert.Equal(t, realServers1[2].String(), "10.0.1.5:80")

	// externalIP should route to local ready + non-terminating endpoints if they exist
	realServers2, rsErr2 := ipvs.GetRealServers(externalIPServer)
	assert.Nil(t, rsErr2, "Expected no error getting real servers")
	assert.Len(t, realServers2, 2, "Expected 2 real servers")
	assert.Equal(t, realServers2[0].String(), "10.0.1.1:80")
	assert.Equal(t, realServers2[1].String(), "10.0.1.2:80")

	fp.OnEndpointSliceDelete(endpointSlice)
	fp.syncProxyRules()

	// Ensure that Proxier updates ipvs appropriately after EndpointSlice delete
	assert.NotNil(t, fp.ipsetList["KUBE-LOOP-BACK"])
	activeEntries2 := fp.ipsetList["KUBE-LOOP-BACK"].activeEntries
	assert.Equal(t, 0, activeEntries2.Len(), "Expected 0 active entries in KUBE-LOOP-BACK")

	virtualServers, vsErr = ipvs.GetVirtualServers()
	assert.Nil(t, vsErr, "Expected no error getting virtual servers")
	assert.Len(t, virtualServers, 2, "Expected 2 virtual server")

	for _, virtualServer := range virtualServers {
		if virtualServer.Address.String() == "172.20.1.1" {
			clusterIPServer = virtualServer
		}

		if virtualServer.Address.String() == "1.2.3.4" {
			externalIPServer = virtualServer
		}
	}

	realServers1, rsErr1 = ipvs.GetRealServers(clusterIPServer)
	assert.Nil(t, rsErr1, "Expected no error getting real servers")
	assert.Len(t, realServers1, 0, "Expected 0 real servers")

	realServers2, rsErr2 = ipvs.GetRealServers(externalIPServer)
	assert.Nil(t, rsErr2, "Expected no error getting real servers")
	assert.Len(t, realServers2, 0, "Expected 0 real servers")
}

// Test_EndpointSliceOnlyReadyTerminatingCluster tests that when there are only ready terminating
// endpoints and the traffic policy is "Cluster",  we fall back to terminating endpoints.
func Test_EndpointSliceOnlyReadyAndTerminatingCluster(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ProxyTerminatingEndpoints, true)()

	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, v1.IPv4Protocol)
	fp.servicesSynced = true
	// fp.endpointsSynced = true
	fp.endpointSlicesSynced = true

	clusterInternalTrafficPolicy := v1.ServiceInternalTrafficPolicyCluster

	// Add initial service
	serviceName := "svc1"
	namespaceName := "ns1"
	fp.OnServiceAdd(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: serviceName, Namespace: namespaceName},
		Spec: v1.ServiceSpec{
			ClusterIP:             "172.20.1.1",
			Selector:              map[string]string{"foo": "bar"},
			Type:                  v1.ServiceTypeNodePort,
			ExternalTrafficPolicy: v1.ServiceExternalTrafficPolicyTypeCluster,
			InternalTrafficPolicy: &clusterInternalTrafficPolicy,
			ExternalIPs: []string{
				"1.2.3.4",
			},
			Ports: []v1.ServicePort{
				{
					Name:       "",
					Port:       80,
					TargetPort: intstr.FromInt(80),
					Protocol:   v1.ProtocolTCP,
				},
			},
		},
	})

	// Add initial endpoint slice
	tcpProtocol := v1.ProtocolTCP
	endpointSlice := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-1", serviceName),
			Namespace: namespaceName,
			Labels:    map[string]string{discovery.LabelServiceName: serviceName},
		},
		Ports: []discovery.EndpointPort{{
			Name:     pointer.String(""),
			Port:     pointer.Int32(80),
			Protocol: &tcpProtocol,
		}},
		AddressType: discovery.AddressTypeIPv4,
		Endpoints: []discovery.Endpoint{
			{
				Addresses: []string{"10.0.1.1"},
				Conditions: discovery.EndpointConditions{
					Ready:       pointer.Bool(false),
					Serving:     pointer.Bool(true),
					Terminating: pointer.Bool(true),
				},
				NodeName: pointer.String(testHostname),
			},
			{
				Addresses: []string{"10.0.1.2"},
				Conditions: discovery.EndpointConditions{
					Ready:       pointer.Bool(false),
					Serving:     pointer.Bool(true),
					Terminating: pointer.Bool(true),
				},
				NodeName: pointer.String(testHostname),
			},
			{
				Addresses: []string{"10.0.1.3"},
				Conditions: discovery.EndpointConditions{
					Ready:       pointer.Bool(false),
					Serving:     pointer.Bool(false),
					Terminating: pointer.Bool(true),
				},
				NodeName: pointer.String(testHostname),
			},
			{
				Addresses: []string{"10.0.1.4"},
				Conditions: discovery.EndpointConditions{
					Ready:       pointer.Bool(false),
					Serving:     pointer.Bool(true),
					Terminating: pointer.Bool(true),
				},
				NodeName: pointer.String("another-host"),
			},
			{
				Addresses: []string{"10.0.1.5"},
				Conditions: discovery.EndpointConditions{
					Ready:       pointer.Bool(false),
					Serving:     pointer.Bool(false),
					Terminating: pointer.Bool(false),
				},
				NodeName: pointer.String("another-host"),
			},
		},
	}

	fp.OnEndpointSliceAdd(endpointSlice)
	fp.syncProxyRules()

	// Ensure that Proxier updates ipvs appropriately after EndpointSlice update
	assert.NotNil(t, fp.ipsetList["KUBE-LOOP-BACK"])
	activeEntries1 := fp.ipsetList["KUBE-LOOP-BACK"].activeEntries
	assert.Equal(t, 3, activeEntries1.Len(), "Expected 3 active entry in KUBE-LOOP-BACK")
	assert.Equal(t, true, activeEntries1.Has("10.0.1.1,tcp:80,10.0.1.1"), "Expected activeEntries to reference first (local) pod")
	assert.Equal(t, true, activeEntries1.Has("10.0.1.2,tcp:80,10.0.1.2"), "Expected activeEntries to reference second (local) pod")
	assert.Equal(t, true, activeEntries1.Has("10.0.1.3,tcp:80,10.0.1.3"), "Expected activeEntries to reference second (local) pod")

	virtualServers, vsErr := ipvs.GetVirtualServers()
	assert.Nil(t, vsErr, "Expected no error getting virtual servers")
	assert.Len(t, virtualServers, 2, "Expected 2 virtual server")

	var clusterIPServer, externalIPServer *utilipvs.VirtualServer
	for _, virtualServer := range virtualServers {
		if virtualServer.Address.String() == "172.20.1.1" {
			clusterIPServer = virtualServer
		}

		if virtualServer.Address.String() == "1.2.3.4" {
			externalIPServer = virtualServer
		}
	}

	// clusterIP should fall back to cluster-wide ready + terminating endpoints
	realServers1, rsErr1 := ipvs.GetRealServers(clusterIPServer)
	assert.Nil(t, rsErr1, "Expected no error getting real servers")
	assert.Len(t, realServers1, 3, "Expected 1 real servers")
	assert.Equal(t, realServers1[0].String(), "10.0.1.1:80")
	assert.Equal(t, realServers1[1].String(), "10.0.1.2:80")
	assert.Equal(t, realServers1[2].String(), "10.0.1.4:80")

	// externalIP should fall back to ready + terminating endpoints
	realServers2, rsErr2 := ipvs.GetRealServers(externalIPServer)
	assert.Nil(t, rsErr2, "Expected no error getting real servers")
	assert.Len(t, realServers2, 3, "Expected 2 real servers")
	assert.Equal(t, realServers2[0].String(), "10.0.1.1:80")
	assert.Equal(t, realServers2[1].String(), "10.0.1.2:80")
	assert.Equal(t, realServers2[2].String(), "10.0.1.4:80")

	fp.OnEndpointSliceDelete(endpointSlice)
	fp.syncProxyRules()

	// Ensure that Proxier updates ipvs appropriately after EndpointSlice delete
	assert.NotNil(t, fp.ipsetList["KUBE-LOOP-BACK"])
	activeEntries2 := fp.ipsetList["KUBE-LOOP-BACK"].activeEntries
	assert.Equal(t, 0, activeEntries2.Len(), "Expected 0 active entries in KUBE-LOOP-BACK")

	virtualServers, vsErr = ipvs.GetVirtualServers()
	assert.Nil(t, vsErr, "Expected no error getting virtual servers")
	assert.Len(t, virtualServers, 2, "Expected 2 virtual server")

	for _, virtualServer := range virtualServers {
		if virtualServer.Address.String() == "172.20.1.1" {
			clusterIPServer = virtualServer
		}

		if virtualServer.Address.String() == "1.2.3.4" {
			externalIPServer = virtualServer
		}
	}

	realServers1, rsErr1 = ipvs.GetRealServers(clusterIPServer)
	assert.Nil(t, rsErr1, "Expected no error getting real servers")
	assert.Len(t, realServers1, 0, "Expected 0 real servers")

	realServers2, rsErr2 = ipvs.GetRealServers(externalIPServer)
	assert.Nil(t, rsErr2, "Expected no error getting real servers")
	assert.Len(t, realServers2, 0, "Expected 0 real servers")
}

// Test_EndpointSliceOnlyReadyTerminatingLocal tests that when there are only local ready terminating
// endpoints, we fall back to those endpoints.
func Test_EndpointSliceOnlyReadyAndTerminatingLocal(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ProxyTerminatingEndpoints, true)()

	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, v1.IPv4Protocol)
	fp.servicesSynced = true
	// fp.endpointsSynced = true
	fp.endpointSlicesSynced = true

	clusterInternalTrafficPolicy := v1.ServiceInternalTrafficPolicyCluster

	// Add initial service
	serviceName := "svc1"
	namespaceName := "ns1"
	fp.OnServiceAdd(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: serviceName, Namespace: namespaceName},
		Spec: v1.ServiceSpec{
			ClusterIP:             "172.20.1.1",
			Selector:              map[string]string{"foo": "bar"},
			Type:                  v1.ServiceTypeNodePort,
			ExternalTrafficPolicy: v1.ServiceExternalTrafficPolicyTypeLocal,
			InternalTrafficPolicy: &clusterInternalTrafficPolicy,
			ExternalIPs: []string{
				"1.2.3.4",
			},
			Ports: []v1.ServicePort{
				{
					Name:       "",
					Port:       80,
					TargetPort: intstr.FromInt(80),
					Protocol:   v1.ProtocolTCP,
				},
			},
		},
	})

	// Add initial endpoint slice
	tcpProtocol := v1.ProtocolTCP
	endpointSlice := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-1", serviceName),
			Namespace: namespaceName,
			Labels:    map[string]string{discovery.LabelServiceName: serviceName},
		},
		Ports: []discovery.EndpointPort{{
			Name:     pointer.String(""),
			Port:     pointer.Int32(80),
			Protocol: &tcpProtocol,
		}},
		AddressType: discovery.AddressTypeIPv4,
		Endpoints: []discovery.Endpoint{
			{
				Addresses: []string{"10.0.1.1"},
				Conditions: discovery.EndpointConditions{
					Ready:       pointer.Bool(false),
					Serving:     pointer.Bool(true),
					Terminating: pointer.Bool(true),
				},
				NodeName: pointer.String(testHostname),
			},
			{
				Addresses: []string{"10.0.1.2"},
				Conditions: discovery.EndpointConditions{
					Ready:       pointer.Bool(false),
					Serving:     pointer.Bool(true),
					Terminating: pointer.Bool(true),
				},
				NodeName: pointer.String(testHostname),
			},
			{
				Addresses: []string{"10.0.1.3"},
				Conditions: discovery.EndpointConditions{
					Ready:       pointer.Bool(false),
					Serving:     pointer.Bool(false),
					Terminating: pointer.Bool(true),
				},
				NodeName: pointer.String(testHostname),
			},
			{
				Addresses: []string{"10.0.1.4"},
				Conditions: discovery.EndpointConditions{
					Ready:       pointer.Bool(false),
					Serving:     pointer.Bool(true),
					Terminating: pointer.Bool(true),
				},
				NodeName: pointer.String("another-host"),
			},
			{
				Addresses: []string{"10.0.1.5"},
				Conditions: discovery.EndpointConditions{
					Ready:       pointer.Bool(true),
					Serving:     pointer.Bool(true),
					Terminating: pointer.Bool(false),
				},
				NodeName: pointer.String("another-host"),
			},
		},
	}

	fp.OnEndpointSliceAdd(endpointSlice)
	fp.syncProxyRules()

	// Ensure that Proxier updates ipvs appropriately after EndpointSlice update
	assert.NotNil(t, fp.ipsetList["KUBE-LOOP-BACK"])
	activeEntries1 := fp.ipsetList["KUBE-LOOP-BACK"].activeEntries
	assert.Equal(t, 3, activeEntries1.Len(), "Expected 3 active entry in KUBE-LOOP-BACK")
	assert.Equal(t, true, activeEntries1.Has("10.0.1.1,tcp:80,10.0.1.1"), "Expected activeEntries to reference first (local) pod")
	assert.Equal(t, true, activeEntries1.Has("10.0.1.2,tcp:80,10.0.1.2"), "Expected activeEntries to reference second (local) pod")
	assert.Equal(t, true, activeEntries1.Has("10.0.1.3,tcp:80,10.0.1.3"), "Expected activeEntries to reference second (local) pod")

	virtualServers, vsErr := ipvs.GetVirtualServers()
	assert.Nil(t, vsErr, "Expected no error getting virtual servers")
	assert.Len(t, virtualServers, 2, "Expected 2 virtual server")

	var clusterIPServer, externalIPServer *utilipvs.VirtualServer
	for _, virtualServer := range virtualServers {
		if virtualServer.Address.String() == "172.20.1.1" {
			clusterIPServer = virtualServer
		}

		if virtualServer.Address.String() == "1.2.3.4" {
			externalIPServer = virtualServer
		}
	}

	// clusterIP should route to cluster-wide Ready endpoints
	realServers1, rsErr1 := ipvs.GetRealServers(clusterIPServer)
	assert.Nil(t, rsErr1, "Expected no error getting real servers")
	assert.Len(t, realServers1, 1, "Expected 1 real servers")
	assert.Equal(t, realServers1[0].String(), "10.0.1.5:80")

	// externalIP should fall back to local ready + terminating endpoints
	realServers2, rsErr2 := ipvs.GetRealServers(externalIPServer)
	assert.Nil(t, rsErr2, "Expected no error getting real servers")
	assert.Len(t, realServers2, 2, "Expected 2 real servers")
	assert.Equal(t, realServers2[0].String(), "10.0.1.1:80")
	assert.Equal(t, realServers2[1].String(), "10.0.1.2:80")

	fp.OnEndpointSliceDelete(endpointSlice)
	fp.syncProxyRules()

	// Ensure that Proxier updates ipvs appropriately after EndpointSlice delete
	assert.NotNil(t, fp.ipsetList["KUBE-LOOP-BACK"])
	activeEntries2 := fp.ipsetList["KUBE-LOOP-BACK"].activeEntries
	assert.Equal(t, 0, activeEntries2.Len(), "Expected 0 active entries in KUBE-LOOP-BACK")

	virtualServers, vsErr = ipvs.GetVirtualServers()
	assert.Nil(t, vsErr, "Expected no error getting virtual servers")
	assert.Len(t, virtualServers, 2, "Expected 2 virtual server")

	for _, virtualServer := range virtualServers {
		if virtualServer.Address.String() == "172.20.1.1" {
			clusterIPServer = virtualServer
		}

		if virtualServer.Address.String() == "1.2.3.4" {
			externalIPServer = virtualServer
		}
	}

	realServers1, rsErr1 = ipvs.GetRealServers(clusterIPServer)
	assert.Nil(t, rsErr1, "Expected no error getting real servers")
	assert.Len(t, realServers1, 0, "Expected 0 real servers")

	realServers2, rsErr2 = ipvs.GetRealServers(externalIPServer)
	assert.Nil(t, rsErr2, "Expected no error getting real servers")
	assert.Len(t, realServers2, 0, "Expected 0 real servers")
}

// Test_EndpointSliceOnlyReadyTerminatingLocalWithFeatureGateDisabled tests that when there are only local ready terminating
// endpoints, we fall back to those endpoints.
func Test_EndpointSliceOnlyReadyAndTerminatingLocalWithFeatureGateDisabled(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ProxyTerminatingEndpoints, false)()

	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, v1.IPv4Protocol)
	fp.servicesSynced = true
	// fp.endpointsSynced = true
	fp.endpointSlicesSynced = true

	clusterInternalTrafficPolicy := v1.ServiceInternalTrafficPolicyCluster

	// Add initial service
	serviceName := "svc1"
	namespaceName := "ns1"
	fp.OnServiceAdd(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: serviceName, Namespace: namespaceName},
		Spec: v1.ServiceSpec{
			ClusterIP:             "172.20.1.1",
			Selector:              map[string]string{"foo": "bar"},
			Type:                  v1.ServiceTypeNodePort,
			ExternalTrafficPolicy: v1.ServiceExternalTrafficPolicyTypeLocal,
			InternalTrafficPolicy: &clusterInternalTrafficPolicy,
			ExternalIPs: []string{
				"1.2.3.4",
			},
			Ports: []v1.ServicePort{
				{
					Name:       "",
					Port:       80,
					TargetPort: intstr.FromInt(80),
					Protocol:   v1.ProtocolTCP,
				},
			},
		},
	})

	// Add initial endpoint slice
	tcpProtocol := v1.ProtocolTCP
	endpointSlice := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-1", serviceName),
			Namespace: namespaceName,
			Labels:    map[string]string{discovery.LabelServiceName: serviceName},
		},
		Ports: []discovery.EndpointPort{{
			Name:     pointer.String(""),
			Port:     pointer.Int32(80),
			Protocol: &tcpProtocol,
		}},
		AddressType: discovery.AddressTypeIPv4,
		Endpoints: []discovery.Endpoint{
			{
				Addresses: []string{"10.0.1.1"},
				Conditions: discovery.EndpointConditions{
					Ready:       pointer.Bool(false),
					Serving:     pointer.Bool(true),
					Terminating: pointer.Bool(true),
				},
				NodeName: pointer.String(testHostname),
			},
			{
				Addresses: []string{"10.0.1.2"},
				Conditions: discovery.EndpointConditions{
					Ready:       pointer.Bool(false),
					Serving:     pointer.Bool(true),
					Terminating: pointer.Bool(true),
				},
				NodeName: pointer.String(testHostname),
			},
			{
				Addresses: []string{"10.0.1.3"},
				Conditions: discovery.EndpointConditions{
					Ready:       pointer.Bool(false),
					Serving:     pointer.Bool(false),
					Terminating: pointer.Bool(true),
				},
				NodeName: pointer.String(testHostname),
			},
			{
				Addresses: []string{"10.0.1.4"},
				Conditions: discovery.EndpointConditions{
					Ready:       pointer.Bool(false),
					Serving:     pointer.Bool(true),
					Terminating: pointer.Bool(true),
				},
				NodeName: pointer.String("another-host"),
			},
			{
				Addresses: []string{"10.0.1.5"},
				Conditions: discovery.EndpointConditions{
					Ready:       pointer.Bool(true),
					Serving:     pointer.Bool(true),
					Terminating: pointer.Bool(false),
				},
				NodeName: pointer.String("another-host"),
			},
		},
	}

	fp.OnEndpointSliceAdd(endpointSlice)
	fp.syncProxyRules()

	// Ensure that Proxier updates ipvs appropriately after EndpointSlice update
	assert.NotNil(t, fp.ipsetList["KUBE-LOOP-BACK"])
	activeEntries1 := fp.ipsetList["KUBE-LOOP-BACK"].activeEntries
	assert.Equal(t, 3, activeEntries1.Len(), "Expected 3 active entry in KUBE-LOOP-BACK")
	assert.Equal(t, true, activeEntries1.Has("10.0.1.1,tcp:80,10.0.1.1"), "Expected activeEntries to reference first (local) pod")
	assert.Equal(t, true, activeEntries1.Has("10.0.1.2,tcp:80,10.0.1.2"), "Expected activeEntries to reference second (local) pod")
	assert.Equal(t, true, activeEntries1.Has("10.0.1.3,tcp:80,10.0.1.3"), "Expected activeEntries to reference second (local) pod")

	virtualServers, vsErr := ipvs.GetVirtualServers()
	assert.Nil(t, vsErr, "Expected no error getting virtual servers")
	assert.Len(t, virtualServers, 2, "Expected 2 virtual server")

	var clusterIPServer, externalIPServer *utilipvs.VirtualServer
	for _, virtualServer := range virtualServers {
		if virtualServer.Address.String() == "172.20.1.1" {
			clusterIPServer = virtualServer
		}

		if virtualServer.Address.String() == "1.2.3.4" {
			externalIPServer = virtualServer
		}
	}

	// clusterIP should route to cluster-wide Ready endpoints
	realServers1, rsErr1 := ipvs.GetRealServers(clusterIPServer)
	assert.Nil(t, rsErr1, "Expected no error getting real servers")
	assert.Len(t, realServers1, 1, "Expected 1 real servers")
	assert.Equal(t, realServers1[0].String(), "10.0.1.5:80")

	// externalIP should have 1 (remote) endpoint since the feature gate is disabled.
	realServers2, rsErr2 := ipvs.GetRealServers(externalIPServer)
	assert.Nil(t, rsErr2, "Expected no error getting real servers")
	assert.Len(t, realServers2, 1, "Expected 0 real servers")
	assert.Equal(t, realServers2[0].String(), "10.0.1.5:80")

	fp.OnEndpointSliceDelete(endpointSlice)
	fp.syncProxyRules()

	// Ensure that Proxier updates ipvs appropriately after EndpointSlice delete
	assert.NotNil(t, fp.ipsetList["KUBE-LOOP-BACK"])
	activeEntries2 := fp.ipsetList["KUBE-LOOP-BACK"].activeEntries
	assert.Equal(t, 0, activeEntries2.Len(), "Expected 0 active entries in KUBE-LOOP-BACK")

	virtualServers, vsErr = ipvs.GetVirtualServers()
	assert.Nil(t, vsErr, "Expected no error getting virtual servers")
	assert.Len(t, virtualServers, 2, "Expected 2 virtual server")

	for _, virtualServer := range virtualServers {
		if virtualServer.Address.String() == "172.20.1.1" {
			clusterIPServer = virtualServer
		}

		if virtualServer.Address.String() == "1.2.3.4" {
			externalIPServer = virtualServer
		}
	}

	realServers1, rsErr1 = ipvs.GetRealServers(clusterIPServer)
	assert.Nil(t, rsErr1, "Expected no error getting real servers")
	assert.Len(t, realServers1, 0, "Expected 0 real servers")

	realServers2, rsErr2 = ipvs.GetRealServers(externalIPServer)
	assert.Nil(t, rsErr2, "Expected no error getting real servers")
	assert.Len(t, realServers2, 0, "Expected 0 real servers")
}

func TestIpIsValidForSet(t *testing.T) {
	testCases := []struct {
		isIPv6 bool
		ip     string
		res    bool
	}{
		{
			false,
			"127.0.0.1",
			false,
		},
		{
			false,
			"127.0.0.0",
			false,
		},
		{
			false,
			"127.6.7.8",
			false,
		},
		{
			false,
			"8.8.8.8",
			true,
		},
		{
			false,
			"192.168.0.1",
			true,
		},
		{
			false,
			"169.254.0.0",
			true,
		},
		{
			false,
			"::ffff:169.254.0.0", // IPv6 mapped IPv4
			true,
		},
		{
			false,
			"1000::",
			false,
		},
		// IPv6
		{
			true,
			"::1",
			false,
		},
		{
			true,
			"1000::",
			true,
		},
		{
			true,
			"fe80::200:ff:fe01:1",
			false,
		},
		{
			true,
			"8.8.8.8",
			false,
		},
		{
			true,
			"::ffff:8.8.8.8",
			false,
		},
	}

	for _, tc := range testCases {
		v := &netlinkHandle{}
		v.isIPv6 = tc.isIPv6
		ip := netutils.ParseIPSloppy(tc.ip)
		if ip == nil {
			t.Errorf("Parse error: %s", tc.ip)
		}
		if v.isValidForSet(ip) != tc.res {
			if tc.isIPv6 {
				t.Errorf("IPv6: %s", tc.ip)
			} else {
				t.Errorf("IPv4: %s", tc.ip)
			}
		}
	}
}

func TestNoEndpointsMetric(t *testing.T) {
	type endpoint struct {
		ip       string
		hostname string
	}

	internalTrafficPolicyLocal := v1.ServiceInternalTrafficPolicyLocal
	externalTrafficPolicyLocal := v1.ServiceExternalTrafficPolicyTypeLocal
	metrics.RegisterMetrics()

	testCases := []struct {
		name                                                string
		internalTrafficPolicy                               *v1.ServiceInternalTrafficPolicyType
		externalTrafficPolicy                               v1.ServiceExternalTrafficPolicyType
		endpoints                                           []endpoint
		expectedSyncProxyRulesNoLocalEndpointsTotalInternal int
		expectedSyncProxyRulesNoLocalEndpointsTotalExternal int
	}{
		{
			name:                  "internalTrafficPolicy is set and there are local endpoints",
			internalTrafficPolicy: &internalTrafficPolicyLocal,
			endpoints: []endpoint{
				{"10.0.1.1", testHostname},
				{"10.0.1.2", "host1"},
				{"10.0.1.3", "host2"},
			},
		},
		{
			name:                  "externalTrafficPolicy is set and there are local endpoints",
			externalTrafficPolicy: externalTrafficPolicyLocal,
			endpoints: []endpoint{
				{"10.0.1.1", testHostname},
				{"10.0.1.2", "host1"},
				{"10.0.1.3", "host2"},
			},
		},
		{
			name:                  "both policies are set and there are local endpoints",
			internalTrafficPolicy: &internalTrafficPolicyLocal,
			externalTrafficPolicy: externalTrafficPolicyLocal,
			endpoints: []endpoint{
				{"10.0.1.1", testHostname},
				{"10.0.1.2", "host1"},
				{"10.0.1.3", "host2"},
			},
		},
		{
			name:                  "internalTrafficPolicy is set and there are no local endpoints",
			internalTrafficPolicy: &internalTrafficPolicyLocal,
			endpoints: []endpoint{
				{"10.0.1.1", "host0"},
				{"10.0.1.2", "host1"},
				{"10.0.1.3", "host2"},
			},
			expectedSyncProxyRulesNoLocalEndpointsTotalInternal: 1,
		},
		{
			name:                  "externalTrafficPolicy is set and there are no local endpoints",
			externalTrafficPolicy: externalTrafficPolicyLocal,
			endpoints: []endpoint{
				{"10.0.1.1", "host0"},
				{"10.0.1.2", "host1"},
				{"10.0.1.3", "host2"},
			},
			expectedSyncProxyRulesNoLocalEndpointsTotalExternal: 1,
		},
		{
			name:                  "Both policies are set and there are no local endpoints",
			internalTrafficPolicy: &internalTrafficPolicyLocal,
			externalTrafficPolicy: externalTrafficPolicyLocal,
			endpoints: []endpoint{
				{"10.0.1.1", "host0"},
				{"10.0.1.2", "host1"},
				{"10.0.1.3", "host2"},
			},
			expectedSyncProxyRulesNoLocalEndpointsTotalInternal: 1,
			expectedSyncProxyRulesNoLocalEndpointsTotalExternal: 1,
		},
		{
			name:                  "Both policies are set and there are no endpoints at all",
			internalTrafficPolicy: &internalTrafficPolicyLocal,
			externalTrafficPolicy: externalTrafficPolicyLocal,
			endpoints:             []endpoint{},
			expectedSyncProxyRulesNoLocalEndpointsTotalInternal: 0,
			expectedSyncProxyRulesNoLocalEndpointsTotalExternal: 0,
		},
	}
	for _, tc := range testCases {
		defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ServiceInternalTrafficPolicy, true)()

		ipt := iptablestest.NewFake()
		ipvs := ipvstest.NewFake()
		ipset := ipsettest.NewFake(testIPSetVersion)
		fp := NewFakeProxier(ipt, ipvs, ipset, []net.IP{netutils.ParseIPSloppy("10.0.0.1")}, nil, v1.IPv4Protocol)
		fp.servicesSynced = true
		// fp.endpointsSynced = true
		fp.endpointSlicesSynced = true

		// Add initial service
		serviceName := "svc1"
		namespaceName := "ns1"

		svc := &v1.Service{
			ObjectMeta: metav1.ObjectMeta{Name: serviceName, Namespace: namespaceName},
			Spec: v1.ServiceSpec{
				ClusterIP: "172.20.1.1",
				Selector:  map[string]string{"foo": "bar"},
				Ports:     []v1.ServicePort{{Name: "p80", Port: 80, TargetPort: intstr.FromInt(80), Protocol: v1.ProtocolTCP, NodePort: 30000}},
			},
		}
		if tc.internalTrafficPolicy != nil {
			svc.Spec.InternalTrafficPolicy = tc.internalTrafficPolicy
		}
		if tc.externalTrafficPolicy != "" {
			svc.Spec.Type = v1.ServiceTypeNodePort
			svc.Spec.ExternalTrafficPolicy = tc.externalTrafficPolicy
		}

		fp.OnServiceAdd(svc)

		// Add initial endpoint slice
		tcpProtocol := v1.ProtocolTCP
		endpointSlice := &discovery.EndpointSlice{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("%s-1", serviceName),
				Namespace: namespaceName,
				Labels:    map[string]string{discovery.LabelServiceName: serviceName},
			},
			Ports: []discovery.EndpointPort{{
				Name:     pointer.String("p80"),
				Port:     pointer.Int32(80),
				Protocol: &tcpProtocol,
			}},
			AddressType: discovery.AddressTypeIPv4,
		}

		for _, ep := range tc.endpoints {
			endpointSlice.Endpoints = append(endpointSlice.Endpoints, discovery.Endpoint{
				Addresses:  []string{ep.ip},
				Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
				NodeName:   pointer.String(ep.hostname),
			})
		}

		fp.OnEndpointSliceAdd(endpointSlice)
		fp.syncProxyRules()

		syncProxyRulesNoLocalEndpointsTotalInternal, err := testutil.GetGaugeMetricValue(metrics.SyncProxyRulesNoLocalEndpointsTotal.WithLabelValues("internal"))
		if err != nil {
			t.Errorf("failed to get %s value(internal), err: %v", metrics.SyncProxyRulesNoLocalEndpointsTotal.Name, err)
		}

		if tc.expectedSyncProxyRulesNoLocalEndpointsTotalInternal != int(syncProxyRulesNoLocalEndpointsTotalInternal) {
			t.Errorf("sync_proxy_rules_no_endpoints_total metric mismatch(internal): got=%d, expected %d", int(syncProxyRulesNoLocalEndpointsTotalInternal), tc.expectedSyncProxyRulesNoLocalEndpointsTotalInternal)
		}

		syncProxyRulesNoLocalEndpointsTotalExternal, err := testutil.GetGaugeMetricValue(metrics.SyncProxyRulesNoLocalEndpointsTotal.WithLabelValues("external"))
		if err != nil {
			t.Errorf("failed to get %s value(external), err: %v", metrics.SyncProxyRulesNoLocalEndpointsTotal.Name, err)
		}

		if tc.expectedSyncProxyRulesNoLocalEndpointsTotalExternal != int(syncProxyRulesNoLocalEndpointsTotalExternal) {
			t.Errorf("sync_proxy_rules_no_endpoints_total metric mismatch(external): got=%d, expected %d", int(syncProxyRulesNoLocalEndpointsTotalExternal), tc.expectedSyncProxyRulesNoLocalEndpointsTotalExternal)
		}
	}
}
