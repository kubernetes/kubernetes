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
	discovery "k8s.io/api/discovery/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	netlinktest "k8s.io/kubernetes/pkg/proxy/ipvs/testing"
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
	utilpointer "k8s.io/utils/pointer"
)

const testHostname = "test-hostname"

type fakeIPGetter struct {
	nodeIPs []net.IP
}

func (f *fakeIPGetter) NodeIPs() ([]net.IP, error) {
	return f.nodeIPs, nil
}

// fakePortOpener implements portOpener.
type fakePortOpener struct {
	openPorts []*utilproxy.LocalPort
}

// OpenLocalPort fakes out the listen() and bind() used by syncProxyRules
// to lock a local port.
func (f *fakePortOpener) OpenLocalPort(lp *utilproxy.LocalPort, isIPv6 bool) (utilproxy.Closeable, error) {
	f.openPorts = append(f.openPorts, lp)
	return nil, nil
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

func NewFakeProxier(ipt utiliptables.Interface, ipvs utilipvs.Interface, ipset utilipset.Interface, nodeIPs []net.IP, excludeCIDRs []*net.IPNet, endpointSlicesEnabled bool) *Proxier {
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
		serviceChanges:        proxy.NewServiceChangeTracker(newServiceInfo, nil, nil),
		endpointsMap:          make(proxy.EndpointsMap),
		endpointsChanges:      proxy.NewEndpointChangeTracker(testHostname, nil, nil, nil, endpointSlicesEnabled),
		excludeCIDRs:          excludeCIDRs,
		iptables:              ipt,
		ipvs:                  ipvs,
		ipset:                 ipset,
		strictARP:             false,
		localDetector:         proxyutiliptables.NewNoOpLocalDetector(),
		hostname:              testHostname,
		portsMap:              make(map[utilproxy.LocalPort]utilproxy.Closeable),
		portMapper:            &fakePortOpener{[]*utilproxy.LocalPort{}},
		serviceHealthServer:   healthcheck.NewFakeServiceHealthServer(),
		ipvsScheduler:         DefaultScheduler,
		ipGetter:              &fakeIPGetter{nodeIPs: nodeIPs},
		iptablesData:          bytes.NewBuffer(nil),
		filterChainsData:      bytes.NewBuffer(nil),
		natChains:             bytes.NewBuffer(nil),
		natRules:              bytes.NewBuffer(nil),
		filterChains:          bytes.NewBuffer(nil),
		filterRules:           bytes.NewBuffer(nil),
		netlinkHandle:         netlinktest.NewFakeNetlinkHandle(),
		ipsetList:             ipsetList,
		nodePortAddresses:     make([]string, 0),
		networkInterfacer:     proxyutiltest.NewFakeNetwork(),
		gracefuldeleteManager: NewGracefulTerminationManager(ipvs),
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

func makeEndpointsMap(proxier *Proxier, allEndpoints ...*v1.Endpoints) {
	for i := range allEndpoints {
		proxier.OnEndpointsAdd(allEndpoints[i])
	}

	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	proxier.endpointsSynced = true
}

func makeTestEndpoints(namespace, name string, eptFunc func(*v1.Endpoints)) *v1.Endpoints {
	ept := &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
	}
	eptFunc(ept)
	return ept
}

func TestCleanupLeftovers(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, false)
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
	makeEndpointsMap(fp,
		makeTestEndpoints(svcPortName.Namespace, svcPortName.Name, func(ept *v1.Endpoints) {
			ept.Subsets = []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{
					IP: epIP,
				}},
				Ports: []v1.EndpointPort{{
					Name: svcPortName.Port,
					Port: int32(svcPort),
				}},
			}}
		}),
	)

	fp.syncProxyRules()

	// test cleanup left over
	if CleanupLeftovers(ipvs, ipt, ipset, true) {
		t.Errorf("Cleanup leftovers failed")
	}
}

func TestCanUseIPVSProxier(t *testing.T) {
	testCases := []struct {
		mods          []string
		kernelVersion string
		kernelErr     error
		ipsetVersion  string
		ipsetErr      error
		ok            bool
	}{
		// case 0, kernel error
		{
			mods:          []string{"foo", "bar", "baz"},
			kernelVersion: "4.19",
			kernelErr:     fmt.Errorf("oops"),
			ipsetVersion:  "0.0",
			ok:            false,
		},
		// case 1, ipset error
		{
			mods:          []string{"foo", "bar", "baz"},
			kernelVersion: "4.19",
			ipsetVersion:  MinIPSetCheckVersion,
			ipsetErr:      fmt.Errorf("oops"),
			ok:            false,
		},
		// case 2, missing required kernel modules and ipset version too low
		{
			mods:          []string{"foo", "bar", "baz"},
			kernelVersion: "4.19",
			ipsetVersion:  "1.1",
			ok:            false,
		},
		// case 3, missing required ip_vs_* kernel modules
		{
			mods:          []string{"ip_vs", "a", "bc", "def"},
			kernelVersion: "4.19",
			ipsetVersion:  MinIPSetCheckVersion,
			ok:            false,
		},
		// case 4, ipset version too low
		{
			mods:          []string{"ip_vs", "ip_vs_rr", "ip_vs_wrr", "ip_vs_sh", "nf_conntrack"},
			kernelVersion: "4.19",
			ipsetVersion:  "4.3.0",
			ok:            false,
		},
		// case 5, ok for linux kernel 4.19
		{
			mods:          []string{"ip_vs", "ip_vs_rr", "ip_vs_wrr", "ip_vs_sh", "nf_conntrack"},
			kernelVersion: "4.19",
			ipsetVersion:  MinIPSetCheckVersion,
			ok:            true,
		},
		// case 6, ok for linux kernel 4.18
		{
			mods:          []string{"ip_vs", "ip_vs_rr", "ip_vs_wrr", "ip_vs_sh", "nf_conntrack_ipv4"},
			kernelVersion: "4.18",
			ipsetVersion:  MinIPSetCheckVersion,
			ok:            true,
		},
		// case 7. ok when module list has extra modules
		{
			mods:          []string{"foo", "ip_vs", "ip_vs_rr", "ip_vs_wrr", "ip_vs_sh", "nf_conntrack", "bar"},
			kernelVersion: "4.19",
			ipsetVersion:  "6.19",
			ok:            true,
		},
	}

	for i := range testCases {
		handle := &fakeKernelHandler{modules: testCases[i].mods, kernelVersion: testCases[i].kernelVersion}
		versioner := &fakeIPSetVersioner{version: testCases[i].ipsetVersion, err: testCases[i].ipsetErr}
		ok, err := CanUseIPVSProxier(handle, versioner)
		if ok != testCases[i].ok {
			t.Errorf("Case [%d], expect %v, got %v: err: %v", i, testCases[i].ok, ok, err)
		}
	}
}

func TestGetNodeIPs(t *testing.T) {
	testCases := []struct {
		devAddresses map[string][]string
		expectIPs    []string
	}{
		// case 0
		{
			devAddresses: map[string][]string{"eth0": {"1.2.3.4"}, "lo": {"127.0.0.1"}},
			expectIPs:    []string{"1.2.3.4", "127.0.0.1"},
		},
		// case 1
		{
			devAddresses: map[string][]string{"lo": {"127.0.0.1"}},
			expectIPs:    []string{"127.0.0.1"},
		},
		// case 2
		{
			devAddresses: map[string][]string{},
			expectIPs:    []string{},
		},
		// case 3
		{
			devAddresses: map[string][]string{"encap0": {"10.20.30.40"}, "lo": {"127.0.0.1"}, "docker0": {"172.17.0.1"}},
			expectIPs:    []string{"10.20.30.40", "127.0.0.1", "172.17.0.1"},
		},
		// case 4
		{
			devAddresses: map[string][]string{"encaps9": {"10.20.30.40"}, "lo": {"127.0.0.1"}, "encap7": {"10.20.30.31"}},
			expectIPs:    []string{"10.20.30.40", "127.0.0.1", "10.20.30.31"},
		},
		// case 5
		{
			devAddresses: map[string][]string{"kube-ipvs0": {"1.2.3.4"}, "lo": {"127.0.0.1"}, "encap7": {"10.20.30.31"}},
			expectIPs:    []string{"127.0.0.1", "10.20.30.31"},
		},
		// case 6
		{
			devAddresses: map[string][]string{"kube-ipvs0": {"1.2.3.4", "2.3.4.5"}, "lo": {"127.0.0.1"}},
			expectIPs:    []string{"127.0.0.1"},
		},
		// case 7
		{
			devAddresses: map[string][]string{"kube-ipvs0": {"1.2.3.4", "2.3.4.5"}},
			expectIPs:    []string{},
		},
		// case 8
		{
			devAddresses: map[string][]string{"kube-ipvs0": {"1.2.3.4", "2.3.4.5"}, "eth5": {"3.4.5.6"}, "lo": {"127.0.0.1"}},
			expectIPs:    []string{"127.0.0.1", "3.4.5.6"},
		},
		// case 9
		{
			devAddresses: map[string][]string{"ipvs0": {"1.2.3.4"}, "lo": {"127.0.0.1"}, "encap7": {"10.20.30.31"}},
			expectIPs:    []string{"127.0.0.1", "10.20.30.31", "1.2.3.4"},
		},
	}

	for i := range testCases {
		fake := netlinktest.NewFakeNetlinkHandle()
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

func TestNodePort(t *testing.T) {
	tests := []struct {
		name                   string
		services               []*v1.Service
		endpoints              []*v1.Endpoints
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
			endpoints: []*v1.Endpoints{
				makeTestEndpoints("ns1", "svc1", func(ept *v1.Endpoints) {
					ept.Subsets = []v1.EndpointSubset{{
						Addresses: []v1.EndpointAddress{{
							IP: "10.180.0.1",
						}, {
							IP: "1002:ab8::2:10",
						}},
						Ports: []v1.EndpointPort{{
							Name:     "p80",
							Port:     int32(80),
							Protocol: v1.ProtocolTCP,
						}},
					}}
				}),
			},
			nodeIPs: []net.IP{
				net.ParseIP("100.101.102.103"),
				net.ParseIP("2001:db8::1:1"),
			},
			nodePortAddresses: []string{},
			expectedIPVS: &ipvstest.FakeIPVS{
				Services: map[ipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "10.20.30.41",
						Port:     80,
						Protocol: "TCP",
					}: {
						Address:   net.ParseIP("10.20.30.41"),
						Protocol:  "TCP",
						Port:      uint16(80),
						Scheduler: "rr",
					},
					{
						IP:       "100.101.102.103",
						Port:     3001,
						Protocol: "TCP",
					}: {
						Address:   net.ParseIP("100.101.102.103"),
						Protocol:  "TCP",
						Port:      uint16(3001),
						Scheduler: "rr",
					},
					{
						IP:       "2001:db8::1:1",
						Port:     3001,
						Protocol: "TCP",
					}: {
						Address:   net.ParseIP("2001:db8::1:1"),
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
							Address: net.ParseIP("10.180.0.1"),
							Port:    uint16(80),
							Weight:  1,
						},
						{
							Address: net.ParseIP("1002:ab8::2:10"),
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
							Address: net.ParseIP("10.180.0.1"),
							Port:    uint16(80),
							Weight:  1,
						},
						{
							Address: net.ParseIP("1002:ab8::2:10"),
							Port:    uint16(80),
							Weight:  1,
						},
					},
					{
						IP:       "2001:db8::1:1",
						Port:     3001,
						Protocol: "TCP",
					}: {
						{
							Address: net.ParseIP("10.180.0.1"),
							Port:    uint16(80),
							Weight:  1,
						},
						{
							Address: net.ParseIP("1002:ab8::2:10"),
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
			endpoints: []*v1.Endpoints{
				makeTestEndpoints("ns1", "svc1", func(ept *v1.Endpoints) {
					ept.Subsets = []v1.EndpointSubset{{
						Addresses: []v1.EndpointAddress{{
							IP: "10.180.0.1",
						}},
						Ports: []v1.EndpointPort{{
							Name:     "p80",
							Port:     int32(80),
							Protocol: v1.ProtocolUDP,
						}},
					}}
				}),
			},
			nodeIPs: []net.IP{
				net.ParseIP("100.101.102.103"),
			},
			nodePortAddresses: []string{"0.0.0.0/0"},
			expectedIPVS: &ipvstest.FakeIPVS{
				Services: map[ipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "10.20.30.41",
						Port:     80,
						Protocol: "UDP",
					}: {
						Address:   net.ParseIP("10.20.30.41"),
						Protocol:  "UDP",
						Port:      uint16(80),
						Scheduler: "rr",
					},
					{
						IP:       "100.101.102.103",
						Port:     3001,
						Protocol: "UDP",
					}: {
						Address:   net.ParseIP("100.101.102.103"),
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
							Address: net.ParseIP("10.180.0.1"),
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
							Address: net.ParseIP("10.180.0.1"),
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
				string(KubeNodePortChain): {{
					JumpChain: string(KubeMarkMasqChain), MatchSet: kubeNodePortSetUDP,
				}},
				string(kubeServicesChain): {{
					JumpChain: string(KubeNodePortChain), MatchSet: "",
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
			endpoints: []*v1.Endpoints{},
			nodeIPs: []net.IP{
				net.ParseIP("100.101.102.103"),
			},
			nodePortAddresses: []string{},
			expectedIPVS: &ipvstest.FakeIPVS{
				Services: map[ipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "10.20.30.41",
						Port:     80,
						Protocol: "TCP",
					}: {
						Address:   net.ParseIP("10.20.30.41"),
						Protocol:  "TCP",
						Port:      uint16(80),
						Scheduler: "rr",
					},
					{
						IP:       "100.101.102.103",
						Port:     3001,
						Protocol: "TCP",
					}: {
						Address:   net.ParseIP("100.101.102.103"),
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
			endpoints: []*v1.Endpoints{
				makeTestEndpoints("ns1", "svc1", func(ept *v1.Endpoints) {
					ept.Subsets = []v1.EndpointSubset{{
						Addresses: []v1.EndpointAddress{{
							IP: "10.180.0.1",
						}},
						Ports: []v1.EndpointPort{{
							Name:     "p80",
							Port:     int32(80),
							Protocol: v1.ProtocolSCTP,
						}},
					}}
				}),
			},
			nodeIPs: []net.IP{
				net.ParseIP("100.101.102.103"),
				net.ParseIP("100.101.102.104"),
				net.ParseIP("100.101.102.105"),
				net.ParseIP("2001:db8::1:1"),
				net.ParseIP("2001:db8::1:2"),
				net.ParseIP("2001:db8::1:3"),
			},
			nodePortAddresses: []string{},
			expectedIPVS: &ipvstest.FakeIPVS{
				Services: map[ipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "10.20.30.41",
						Port:     80,
						Protocol: "SCTP",
					}: {
						Address:   net.ParseIP("10.20.30.41"),
						Protocol:  "SCTP",
						Port:      uint16(80),
						Scheduler: "rr",
					},
					{
						IP:       "100.101.102.103",
						Port:     3001,
						Protocol: "SCTP",
					}: {
						Address:   net.ParseIP("100.101.102.103"),
						Protocol:  "SCTP",
						Port:      uint16(3001),
						Scheduler: "rr",
					},
					{
						IP:       "100.101.102.104",
						Port:     3001,
						Protocol: "SCTP",
					}: {
						Address:   net.ParseIP("100.101.102.104"),
						Protocol:  "SCTP",
						Port:      uint16(3001),
						Scheduler: "rr",
					},
					{
						IP:       "100.101.102.105",
						Port:     3001,
						Protocol: "SCTP",
					}: {
						Address:   net.ParseIP("100.101.102.105"),
						Protocol:  "SCTP",
						Port:      uint16(3001),
						Scheduler: "rr",
					},
					{
						IP:       "2001:db8::1:1",
						Port:     3001,
						Protocol: "SCTP",
					}: {
						Address:   net.ParseIP("2001:db8::1:1"),
						Protocol:  "SCTP",
						Port:      uint16(3001),
						Scheduler: "rr",
					},
					{
						IP:       "2001:db8::1:2",
						Port:     3001,
						Protocol: "SCTP",
					}: {
						Address:   net.ParseIP("2001:db8::1:2"),
						Protocol:  "SCTP",
						Port:      uint16(3001),
						Scheduler: "rr",
					},
					{
						IP:       "2001:db8::1:3",
						Port:     3001,
						Protocol: "SCTP",
					}: {
						Address:   net.ParseIP("2001:db8::1:3"),
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
							Address: net.ParseIP("10.180.0.1"),
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
							Address: net.ParseIP("10.180.0.1"),
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
							Address: net.ParseIP("10.180.0.1"),
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
							Address: net.ParseIP("10.180.0.1"),
							Port:    uint16(80),
							Weight:  1,
						},
					},
					{
						IP:       "2001:db8::1:1",
						Port:     3001,
						Protocol: "SCTP",
					}: {
						{
							Address: net.ParseIP("10.180.0.1"),
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
							Address: net.ParseIP("10.180.0.1"),
							Port:    uint16(80),
							Weight:  1,
						},
					},
					{
						IP:       "2001:db8::1:3",
						Port:     3001,
						Protocol: "SCTP",
					}: {
						{
							Address: net.ParseIP("10.180.0.1"),
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
					{
						IP:       "2001:db8::1:3",
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
			fp := NewFakeProxier(ipt, ipvs, ipset, test.nodeIPs, nil, false)
			fp.nodePortAddresses = test.nodePortAddresses

			makeServiceMap(fp, test.services...)
			makeEndpointsMap(fp, test.endpoints...)

			fp.syncProxyRules()

			if !reflect.DeepEqual(ipvs, test.expectedIPVS) {
				t.Logf("actual ipvs state: %v", ipvs)
				t.Logf("expected ipvs state: %v", test.expectedIPVS)
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

func TestClusterIP(t *testing.T) {
	tests := []struct {
		name         string
		services     []*v1.Service
		endpoints    []*v1.Endpoints
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
			endpoints: []*v1.Endpoints{
				makeTestEndpoints("ns1", "svc1", func(ept *v1.Endpoints) {
					ept.Subsets = []v1.EndpointSubset{{
						Addresses: []v1.EndpointAddress{{
							IP: "10.180.0.1",
						}},
						Ports: []v1.EndpointPort{{
							Name:     "p80",
							Port:     int32(80),
							Protocol: v1.ProtocolTCP,
						}},
					}}
				}),
				makeTestEndpoints("ns2", "svc2", func(ept *v1.Endpoints) {
					ept.Subsets = []v1.EndpointSubset{{
						Addresses: []v1.EndpointAddress{{
							IP: "1009:ab8::5:6",
						}},
						Ports: []v1.EndpointPort{{
							Name:     "p8080",
							Port:     int32(8080),
							Protocol: v1.ProtocolTCP,
						}},
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
						Address:   net.ParseIP("10.20.30.41"),
						Protocol:  "TCP",
						Port:      uint16(80),
						Scheduler: "rr",
					},
					{
						IP:       "1002:ab8::2:1",
						Port:     8080,
						Protocol: "TCP",
					}: {
						Address:   net.ParseIP("1002:ab8::2:1"),
						Protocol:  "TCP",
						Port:      uint16(8080),
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
							Address: net.ParseIP("10.180.0.1"),
							Port:    uint16(80),
							Weight:  1,
						},
					},
					{
						IP:       "1002:ab8::2:1",
						Port:     8080,
						Protocol: "TCP",
					}: {
						{
							Address: net.ParseIP("1009:ab8::5:6"),
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
					svc.Spec.ClusterIP = "10.20.30.41"
					svc.Spec.Ports = []v1.ServicePort{{
						Name:     "p80",
						Port:     int32(80),
						Protocol: v1.ProtocolTCP,
					}}
				}),
			},
			endpoints: []*v1.Endpoints{},
			expectedIPVS: &ipvstest.FakeIPVS{
				Services: map[ipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "10.20.30.41",
						Port:     80,
						Protocol: "TCP",
					}: {
						Address:   net.ParseIP("10.20.30.41"),
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
			fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, false)

			makeServiceMap(fp, test.services...)
			makeEndpointsMap(fp, test.endpoints...)

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
		fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, false)
		makeServiceMap(fp)
		makeEndpointsMap(fp)
		fp.syncProxyRules()

		postRoutingRules := ipt.GetRules(string(kubePostroutingChain))
		if !hasJump(postRoutingRules, "MASQUERADE", "") {
			t.Errorf("Failed to find -j MASQUERADE in %s chain", kubePostroutingChain)
		}
		if hasMasqRandomFully(postRoutingRules) != testcase {
			probs := map[bool]string{false: "found", true: "did not find"}
			t.Errorf("%s --random-fully in -j MASQUERADE rule in %s chain for HasRandomFully()=%v", probs[testcase], kubePostroutingChain, testcase)
		}
	}
}

func TestExternalIPsNoEndpoint(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, false)
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

	makeEndpointsMap(fp)

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
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, false)
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
	makeEndpointsMap(fp,
		makeTestEndpoints(svcPortName.Namespace, svcPortName.Name, func(ept *v1.Endpoints) {
			ept.Subsets = []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{
					IP: epIP,
				}},
				Ports: []v1.EndpointPort{{
					Name: svcPortName.Port,
					Port: int32(svcPort),
				}},
			}}
		}),
	)

	fp.syncProxyRules()

	// check ipvs service and destinations
	services, err := ipvs.GetVirtualServers()
	if err != nil {
		t.Errorf("Failed to get ipvs services, err: %v", err)
	}
	if len(services) != 4 {
		t.Errorf("Expect 4 ipvs services, got %d", len(services))
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
	// TODO(freehan): remove this in k8s 1.19
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ExternalPolicyForExternalIP, true)()

	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	ipset := ipsettest.NewFake(testIPSetVersion)
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, false)
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
	makeEndpointsMap(fp,
		makeTestEndpoints(svcPortName.Namespace, svcPortName.Name, func(ept *v1.Endpoints) {
			ept.Subsets = []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{
					IP:       epIP,
					NodeName: utilpointer.StringPtr(thisHostname),
				},
					{
						IP:       epIP1,
						NodeName: utilpointer.StringPtr(otherHostname),
					},
				},
				Ports: []v1.EndpointPort{{
					Name:     svcPortName.Port,
					Port:     int32(svcPort),
					Protocol: v1.ProtocolTCP,
				}},
			}}
		}),
	)

	fp.syncProxyRules()

	// check ipvs service and destinations
	services, err := ipvs.GetVirtualServers()
	if err != nil {
		t.Errorf("Failed to get ipvs services, err: %v", err)
	}
	if len(services) != 4 {
		t.Errorf("Expect 4 ipvs services, got %d", len(services))
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
	makeEndpointsMap(fp,
		makeTestEndpoints(svcPortName.Namespace, svcPortName.Name, func(ept *v1.Endpoints) {
			ept.Subsets = []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{
					IP: epIP,
				}},
				Ports: []v1.EndpointPort{{
					Name: svcPortName.Port,
					Port: int32(svcPort),
				}},
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
			JumpChain: string(KubeLoadBalancerChain), MatchSet: kubeLoadBalancerSet,
		}},
		string(kubeLoadBalancerSet): {{
			JumpChain: string(KubeMarkMasqChain), MatchSet: "",
		}},
	}
	checkIptables(t, ipt, epIpt)
}

func TestOnlyLocalNodePorts(t *testing.T) {
	nodeIP := net.ParseIP("100.101.102.103")
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

	makeEndpointsMap(fp,
		makeTestEndpoints(svcPortName.Namespace, svcPortName.Name, func(ept *v1.Endpoints) {
			ept.Subsets = []v1.EndpointSubset{
				{ // **local** endpoint address, should be added as RS
					Addresses: []v1.EndpointAddress{{
						IP:       epIP,
						NodeName: &thisHostname,
					}},
					Ports: []v1.EndpointPort{{
						Name:     svcPortName.Port,
						Port:     int32(svcPort),
						Protocol: v1.ProtocolTCP,
					}}},
				{ // **remote** endpoint address, should not be added as RS
					Addresses: []v1.EndpointAddress{{
						IP:       epIP1,
						NodeName: &otherHostname,
					}},
					Ports: []v1.EndpointPort{{
						Name:     svcPortName.Port,
						Port:     int32(svcPort),
						Protocol: v1.ProtocolTCP,
					}},
				}}
		}),
	)

	itf := net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0}
	addrs := []net.Addr{proxyutiltest.AddrStruct{Val: "100.101.102.103/24"}}
	itf1 := net.Interface{Index: 1, MTU: 0, Name: "eth1", HardwareAddr: nil, Flags: 0}
	addrs1 := []net.Addr{proxyutiltest.AddrStruct{Val: "2001:db8::0/64"}}
	fp.networkInterfacer.(*proxyutiltest.FakeNetwork).AddInterfaceAddr(&itf, addrs)
	fp.networkInterfacer.(*proxyutiltest.FakeNetwork).AddInterfaceAddr(&itf1, addrs1)
	fp.nodePortAddresses = []string{"100.101.102.0/24", "2001:db8::0/64"}

	fp.syncProxyRules()

	// Expect 3 services and 1 destination
	epVS := &netlinktest.ExpectedVirtualServer{
		VSNum: 3, IP: nodeIP.String(), Port: uint16(svcNodePort), Protocol: string(v1.ProtocolTCP),
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
			JumpChain: string(KubeNodePortChain), MatchSet: "",
		}},
		string(KubeNodePortChain): {{
			JumpChain: "RETURN", MatchSet: kubeNodePortLocalSetTCP,
		}, {
			JumpChain: string(KubeMarkMasqChain), MatchSet: kubeNodePortSetTCP,
		}},
	}
	checkIptables(t, ipt, epIpt)
}

func TestLoadBalanceSourceRanges(t *testing.T) {
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
	makeEndpointsMap(fp,
		makeTestEndpoints(svcPortName.Namespace, svcPortName.Name, func(ept *v1.Endpoints) {
			ept.Subsets = []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{
					IP:       epIP,
					NodeName: nil,
				}},
				Ports: []v1.EndpointPort{{
					Name:     svcPortName.Port,
					Port:     int32(svcPort),
					Protocol: v1.ProtocolTCP,
				}},
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
		kubeLoadbalancerFWSet: {{
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
			JumpChain: string(KubeLoadBalancerChain), MatchSet: kubeLoadBalancerSet,
		}},
		string(KubeLoadBalancerChain): {{
			JumpChain: string(KubeFireWallChain), MatchSet: kubeLoadbalancerFWSet,
		}, {
			JumpChain: string(KubeMarkMasqChain), MatchSet: "",
		}},
		string(KubeFireWallChain): {{
			JumpChain: "RETURN", MatchSet: kubeLoadBalancerSourceCIDRSet,
		}, {
			JumpChain: string(KubeMarkDropChain), MatchSet: "",
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

		makeEndpointsMap(fp,
			makeTestEndpoints("ns1", "p80", func(ept *v1.Endpoints) {
				ept.Subsets = []v1.EndpointSubset{{
					Addresses: []v1.EndpointAddress{{
						IP: svcInfo.epIP,
					}},
					Ports: []v1.EndpointPort{{
						Name: "p80",
						Port: 80,
					}},
				}}
			}),
		)
	}
	fp.syncProxyRules()

	// Check iptables chain and rules
	epIpt := netlinktest.ExpectedIptablesChain{
		string(kubeServicesChain): {
			{JumpChain: "ACCEPT", MatchSet: kubeClusterIPSet},
			{JumpChain: "ACCEPT", MatchSet: kubeLoadBalancerSet},
			{JumpChain: "ACCEPT", MatchSet: kubeExternalIPSet},
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

	makeEndpointsMap(fp,
		makeTestEndpoints(svcPortName.Namespace, svcPortName.Name, func(ept *v1.Endpoints) {
			ept.Subsets = []v1.EndpointSubset{
				{ // **local** endpoint address, should be added as RS
					Addresses: []v1.EndpointAddress{{
						IP:       epIP,
						NodeName: &thisHostname,
					}},
					Ports: []v1.EndpointPort{{
						Name:     svcPortName.Port,
						Port:     int32(svcPort),
						Protocol: v1.ProtocolTCP,
					}}},
				{ // **remote** endpoint address, should not be added as RS
					Addresses: []v1.EndpointAddress{{
						IP:       epIP1,
						NodeName: &otherHostname,
					}},
					Ports: []v1.EndpointPort{{
						Name:     svcPortName.Port,
						Port:     int32(svcPort),
						Protocol: v1.ProtocolTCP,
					}},
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
			JumpChain: string(KubeLoadBalancerChain), MatchSet: kubeLoadBalancerSet,
		}},
		string(KubeLoadBalancerChain): {{
			JumpChain: "RETURN", MatchSet: kubeLoadBalancerLocalSet,
		}, {
			JumpChain: string(KubeMarkMasqChain), MatchSet: "",
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
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, false)

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
	result := proxy.UpdateServiceMap(fp.serviceMap, fp.serviceChanges)
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

	result = proxy.UpdateServiceMap(fp.serviceMap, fp.serviceChanges)
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
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, false)

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
	result := proxy.UpdateServiceMap(fp.serviceMap, fp.serviceChanges)
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
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, false)

	makeServiceMap(fp,
		makeTestService("somewhere-else", "external-name", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeExternalName
			svc.Spec.ClusterIP = "172.16.55.4" // Should be ignored
			svc.Spec.ExternalName = "foo2.bar.com"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "blah", "UDP", 1235, 5321, 0)
		}),
	)

	result := proxy.UpdateServiceMap(fp.serviceMap, fp.serviceChanges)
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
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, false)

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

	result := proxy.UpdateServiceMap(fp.serviceMap, fp.serviceChanges)
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
	result = proxy.UpdateServiceMap(fp.serviceMap, fp.serviceChanges)
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
	result = proxy.UpdateServiceMap(fp.serviceMap, fp.serviceChanges)
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
	result = proxy.UpdateServiceMap(fp.serviceMap, fp.serviceChanges)
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
	nodeIP := net.ParseIP("100.101.102.103")
	fp := NewFakeProxier(ipt, ipvs, ipset, []net.IP{nodeIP}, nil, false)
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
	makeEndpointsMap(fp)

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

	emptyEndpoint := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{}
	}
	unnamedPort := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []v1.EndpointPort{{
				Port:     11,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	unnamedPortLocal := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP:       "1.1.1.1",
				NodeName: &nodeName,
			}},
			Ports: []v1.EndpointPort{{
				Port:     11,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	namedPortLocal := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP:       "1.1.1.1",
				NodeName: &nodeName,
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p11",
				Port:     11,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	namedPort := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p11",
				Port:     11,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	namedPortRenamed := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p11-2",
				Port:     11,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	namedPortRenumbered := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p11",
				Port:     22,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	namedPortsLocalNoLocal := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.1",
			}, {
				IP:       "1.1.1.2",
				NodeName: &nodeName,
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p11",
				Port:     11,
				Protocol: v1.ProtocolUDP,
			}, {
				Name:     "p12",
				Port:     12,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	multipleSubsets := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p11",
				Port:     11,
				Protocol: v1.ProtocolUDP,
			}},
		}, {
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.2",
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p12",
				Port:     12,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	multipleSubsetsWithLocal := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p11",
				Port:     11,
				Protocol: v1.ProtocolUDP,
			}},
		}, {
			Addresses: []v1.EndpointAddress{{
				IP:       "1.1.1.2",
				NodeName: &nodeName,
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p12",
				Port:     12,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	multipleSubsetsMultiplePortsLocal := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP:       "1.1.1.1",
				NodeName: &nodeName,
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p11",
				Port:     11,
				Protocol: v1.ProtocolUDP,
			}, {
				Name:     "p12",
				Port:     12,
				Protocol: v1.ProtocolUDP,
			}},
		}, {
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.3",
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p13",
				Port:     13,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	multipleSubsetsIPsPorts1 := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.1",
			}, {
				IP:       "1.1.1.2",
				NodeName: &nodeName,
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p11",
				Port:     11,
				Protocol: v1.ProtocolUDP,
			}, {
				Name:     "p12",
				Port:     12,
				Protocol: v1.ProtocolUDP,
			}},
		}, {
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.3",
			}, {
				IP:       "1.1.1.4",
				NodeName: &nodeName,
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p13",
				Port:     13,
				Protocol: v1.ProtocolUDP,
			}, {
				Name:     "p14",
				Port:     14,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	multipleSubsetsIPsPorts2 := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP: "2.2.2.1",
			}, {
				IP:       "2.2.2.2",
				NodeName: &nodeName,
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p21",
				Port:     21,
				Protocol: v1.ProtocolUDP,
			}, {
				Name:     "p22",
				Port:     22,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	complexBefore1 := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p11",
				Port:     11,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	complexBefore2 := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP:       "2.2.2.2",
				NodeName: &nodeName,
			}, {
				IP:       "2.2.2.22",
				NodeName: &nodeName,
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p22",
				Port:     22,
				Protocol: v1.ProtocolUDP,
			}},
		}, {
			Addresses: []v1.EndpointAddress{{
				IP:       "2.2.2.3",
				NodeName: &nodeName,
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p23",
				Port:     23,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	complexBefore4 := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP:       "4.4.4.4",
				NodeName: &nodeName,
			}, {
				IP:       "4.4.4.5",
				NodeName: &nodeName,
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p44",
				Port:     44,
				Protocol: v1.ProtocolUDP,
			}},
		}, {
			Addresses: []v1.EndpointAddress{{
				IP:       "4.4.4.6",
				NodeName: &nodeName,
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p45",
				Port:     45,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	complexAfter1 := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.1",
			}, {
				IP: "1.1.1.11",
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p11",
				Port:     11,
				Protocol: v1.ProtocolUDP,
			}},
		}, {
			Addresses: []v1.EndpointAddress{{
				IP: "1.1.1.2",
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p12",
				Port:     12,
				Protocol: v1.ProtocolUDP,
			}, {
				Name:     "p122",
				Port:     122,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	complexAfter3 := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP: "3.3.3.3",
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p33",
				Port:     33,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}
	complexAfter4 := func(ept *v1.Endpoints) {
		ept.Subsets = []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP:       "4.4.4.4",
				NodeName: &nodeName,
			}},
			Ports: []v1.EndpointPort{{
				Name:     "p44",
				Port:     44,
				Protocol: v1.ProtocolUDP,
			}},
		}}
	}

	testCases := []struct {
		// previousEndpoints and currentEndpoints are used to call appropriate
		// handlers OnEndpoints* (based on whether corresponding values are nil
		// or non-nil) and must be of equal length.
		previousEndpoints         []*v1.Endpoints
		currentEndpoints          []*v1.Endpoints
		oldEndpoints              map[proxy.ServicePortName][]*proxy.BaseEndpointInfo
		expectedResult            map[proxy.ServicePortName][]*proxy.BaseEndpointInfo
		expectedStaleEndpoints    []proxy.ServiceEndpoint
		expectedStaleServiceNames map[proxy.ServicePortName]bool
		expectedHealthchecks      map[types.NamespacedName]int
	}{{
		// Case[0]: nothing
		oldEndpoints:              map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{},
		expectedResult:            map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{},
		expectedStaleEndpoints:    []proxy.ServiceEndpoint{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[1]: no change, unnamed port
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", unnamedPort),
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", unnamedPort),
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
		},
		expectedStaleEndpoints:    []proxy.ServiceEndpoint{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[2]: no change, named port, local
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortLocal),
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortLocal),
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: true},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: true},
			},
		},
		expectedStaleEndpoints:    []proxy.ServiceEndpoint{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[3]: no change, multiple subsets
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsets),
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsets),
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.2:12", IsLocal: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.2:12", IsLocal: false},
			},
		},
		expectedStaleEndpoints:    []proxy.ServiceEndpoint{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[4]: no change, multiple subsets, multiple ports, local
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsetsMultiplePortsLocal),
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsetsMultiplePortsLocal),
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:12", IsLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p13", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.3:13", IsLocal: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:12", IsLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p13", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.3:13", IsLocal: false},
			},
		},
		expectedStaleEndpoints:    []proxy.ServiceEndpoint{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[5]: no change, multiple endpoints, subsets, IPs, and ports
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsetsIPsPorts1),
			makeTestEndpoints("ns2", "ep2", multipleSubsetsIPsPorts2),
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsetsIPsPorts1),
			makeTestEndpoints("ns2", "ep2", multipleSubsetsIPsPorts2),
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
				{Endpoint: "1.1.1.2:11", IsLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:12", IsLocal: false},
				{Endpoint: "1.1.1.2:12", IsLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p13", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.3:13", IsLocal: false},
				{Endpoint: "1.1.1.4:13", IsLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p14", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.3:14", IsLocal: false},
				{Endpoint: "1.1.1.4:14", IsLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p21", v1.ProtocolUDP): {
				{Endpoint: "2.2.2.1:21", IsLocal: false},
				{Endpoint: "2.2.2.2:21", IsLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p22", v1.ProtocolUDP): {
				{Endpoint: "2.2.2.1:22", IsLocal: false},
				{Endpoint: "2.2.2.2:22", IsLocal: true},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
				{Endpoint: "1.1.1.2:11", IsLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:12", IsLocal: false},
				{Endpoint: "1.1.1.2:12", IsLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p13", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.3:13", IsLocal: false},
				{Endpoint: "1.1.1.4:13", IsLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p14", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.3:14", IsLocal: false},
				{Endpoint: "1.1.1.4:14", IsLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p21", v1.ProtocolUDP): {
				{Endpoint: "2.2.2.1:21", IsLocal: false},
				{Endpoint: "2.2.2.2:21", IsLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p22", v1.ProtocolUDP): {
				{Endpoint: "2.2.2.1:22", IsLocal: false},
				{Endpoint: "2.2.2.2:22", IsLocal: true},
			},
		},
		expectedStaleEndpoints:    []proxy.ServiceEndpoint{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 2,
			makeNSN("ns2", "ep2"): 1,
		},
	}, {
		// Case[6]: add an Endpoints
		previousEndpoints: []*v1.Endpoints{
			nil,
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", unnamedPortLocal),
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: true},
			},
		},
		expectedStaleEndpoints: []proxy.ServiceEndpoint{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "", v1.ProtocolUDP): true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[7]: remove an Endpoints
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", unnamedPortLocal),
		},
		currentEndpoints: []*v1.Endpoints{
			nil,
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: true},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{},
		expectedStaleEndpoints: []proxy.ServiceEndpoint{{
			Endpoint:        "1.1.1.1:11",
			ServicePortName: makeServicePortName("ns1", "ep1", "", v1.ProtocolUDP),
		}},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[8]: add an IP and port
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortsLocalNoLocal),
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
				{Endpoint: "1.1.1.2:11", IsLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:12", IsLocal: false},
				{Endpoint: "1.1.1.2:12", IsLocal: true},
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
		// Case[9]: remove an IP and port
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortsLocalNoLocal),
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
				{Endpoint: "1.1.1.2:11", IsLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:12", IsLocal: false},
				{Endpoint: "1.1.1.2:12", IsLocal: true},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
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
		// Case[10]: add a subset
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsetsWithLocal),
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.2:12", IsLocal: true},
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
		// Case[11]: remove a subset
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsets),
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.2:12", IsLocal: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
		},
		expectedStaleEndpoints: []proxy.ServiceEndpoint{{
			Endpoint:        "1.1.1.2:12",
			ServicePortName: makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP),
		}},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[12]: rename a port
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortRenamed),
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11-2", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
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
		// Case[13]: renumber a port
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortRenumbered),
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:22", IsLocal: false},
			},
		},
		expectedStaleEndpoints: []proxy.ServiceEndpoint{{
			Endpoint:        "1.1.1.1:11",
			ServicePortName: makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP),
		}},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[14]: complex add and remove
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", complexBefore1),
			makeTestEndpoints("ns2", "ep2", complexBefore2),
			nil,
			makeTestEndpoints("ns4", "ep4", complexBefore4),
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", complexAfter1),
			nil,
			makeTestEndpoints("ns3", "ep3", complexAfter3),
			makeTestEndpoints("ns4", "ep4", complexAfter4),
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
			makeServicePortName("ns2", "ep2", "p22", v1.ProtocolUDP): {
				{Endpoint: "2.2.2.2:22", IsLocal: true},
				{Endpoint: "2.2.2.22:22", IsLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p23", v1.ProtocolUDP): {
				{Endpoint: "2.2.2.3:23", IsLocal: true},
			},
			makeServicePortName("ns4", "ep4", "p44", v1.ProtocolUDP): {
				{Endpoint: "4.4.4.4:44", IsLocal: true},
				{Endpoint: "4.4.4.5:44", IsLocal: true},
			},
			makeServicePortName("ns4", "ep4", "p45", v1.ProtocolUDP): {
				{Endpoint: "4.4.4.6:45", IsLocal: true},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
				{Endpoint: "1.1.1.11:11", IsLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.2:12", IsLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p122", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.2:122", IsLocal: false},
			},
			makeServicePortName("ns3", "ep3", "p33", v1.ProtocolUDP): {
				{Endpoint: "3.3.3.3:33", IsLocal: false},
			},
			makeServicePortName("ns4", "ep4", "p44", v1.ProtocolUDP): {
				{Endpoint: "4.4.4.4:44", IsLocal: true},
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
		// Case[15]: change from 0 endpoint address to 1 unnamed port
		previousEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", emptyEndpoint),
		},
		currentEndpoints: []*v1.Endpoints{
			makeTestEndpoints("ns1", "ep1", unnamedPort),
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{},
		expectedResult: map[proxy.ServicePortName][]*proxy.BaseEndpointInfo{
			makeServicePortName("ns1", "ep1", "", v1.ProtocolUDP): {
				{Endpoint: "1.1.1.1:11", IsLocal: false},
			},
		},
		expectedStaleEndpoints: []proxy.ServiceEndpoint{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "", v1.ProtocolUDP): true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{},
	},
	}

	for tci, tc := range testCases {
		ipt := iptablestest.NewFake()
		ipvs := ipvstest.NewFake()
		ipset := ipsettest.NewFake(testIPSetVersion)
		fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, false)
		fp.hostname = nodeName

		// First check that after adding all previous versions of endpoints,
		// the fp.oldEndpoints is as we expect.
		for i := range tc.previousEndpoints {
			if tc.previousEndpoints[i] != nil {
				fp.OnEndpointsAdd(tc.previousEndpoints[i])
			}
		}
		fp.endpointsMap.Update(fp.endpointsChanges)
		compareEndpointsMaps(t, tci, fp.endpointsMap, tc.oldEndpoints)

		// Now let's call appropriate handlers to get to state we want to be.
		if len(tc.previousEndpoints) != len(tc.currentEndpoints) {
			t.Fatalf("[%d] different lengths of previous and current endpoints", tci)
			continue
		}

		for i := range tc.previousEndpoints {
			prev, curr := tc.previousEndpoints[i], tc.currentEndpoints[i]
			switch {
			case prev == nil:
				fp.OnEndpointsAdd(curr)
			case curr == nil:
				fp.OnEndpointsDelete(prev)
			default:
				fp.OnEndpointsUpdate(prev, curr)
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
	}{
		{
			// case 0, old virtual server is same as new virtual server
			oldVirtualServer: &utilipvs.VirtualServer{
				Address:   net.ParseIP("1.2.3.4"),
				Protocol:  string(v1.ProtocolTCP),
				Port:      80,
				Scheduler: "rr",
				Flags:     utilipvs.FlagHashed,
			},
			svcName: "foo",
			newVirtualServer: &utilipvs.VirtualServer{
				Address:   net.ParseIP("1.2.3.4"),
				Protocol:  string(v1.ProtocolTCP),
				Port:      80,
				Scheduler: "rr",
				Flags:     utilipvs.FlagHashed,
			},
			bindAddr: false,
		},
		{
			// case 1, old virtual server is different from new virtual server
			oldVirtualServer: &utilipvs.VirtualServer{
				Address:   net.ParseIP("1.2.3.4"),
				Protocol:  string(v1.ProtocolTCP),
				Port:      8080,
				Scheduler: "rr",
				Flags:     utilipvs.FlagHashed,
			},
			svcName: "bar",
			newVirtualServer: &utilipvs.VirtualServer{
				Address:   net.ParseIP("1.2.3.4"),
				Protocol:  string(v1.ProtocolTCP),
				Port:      8080,
				Scheduler: "rr",
				Flags:     utilipvs.FlagPersistent,
			},
			bindAddr: false,
		},
		{
			// case 2, old virtual server is different from new virtual server
			oldVirtualServer: &utilipvs.VirtualServer{
				Address:   net.ParseIP("1.2.3.4"),
				Protocol:  string(v1.ProtocolTCP),
				Port:      8080,
				Scheduler: "rr",
				Flags:     utilipvs.FlagHashed,
			},
			svcName: "bar",
			newVirtualServer: &utilipvs.VirtualServer{
				Address:   net.ParseIP("1.2.3.4"),
				Protocol:  string(v1.ProtocolTCP),
				Port:      8080,
				Scheduler: "wlc",
				Flags:     utilipvs.FlagHashed,
			},
			bindAddr: false,
		},
		{
			// case 3, old virtual server is nil, and create new virtual server
			oldVirtualServer: nil,
			svcName:          "baz",
			newVirtualServer: &utilipvs.VirtualServer{
				Address:   net.ParseIP("1.2.3.4"),
				Protocol:  string(v1.ProtocolUDP),
				Port:      53,
				Scheduler: "rr",
				Flags:     utilipvs.FlagHashed,
			},
			bindAddr: true,
		},
		{
			// case 4, SCTP, old virtual server is same as new virtual server
			oldVirtualServer: &utilipvs.VirtualServer{
				Address:   net.ParseIP("1.2.3.4"),
				Protocol:  string(v1.ProtocolSCTP),
				Port:      80,
				Scheduler: "rr",
				Flags:     utilipvs.FlagHashed,
			},
			svcName: "foo",
			newVirtualServer: &utilipvs.VirtualServer{
				Address:   net.ParseIP("1.2.3.4"),
				Protocol:  string(v1.ProtocolSCTP),
				Port:      80,
				Scheduler: "rr",
				Flags:     utilipvs.FlagHashed,
			},
			bindAddr: false,
		},
		{
			// case 5, old virtual server is different from new virtual server
			oldVirtualServer: &utilipvs.VirtualServer{
				Address:   net.ParseIP("1.2.3.4"),
				Protocol:  string(v1.ProtocolSCTP),
				Port:      8080,
				Scheduler: "rr",
				Flags:     utilipvs.FlagHashed,
			},
			svcName: "bar",
			newVirtualServer: &utilipvs.VirtualServer{
				Address:   net.ParseIP("1.2.3.4"),
				Protocol:  string(v1.ProtocolSCTP),
				Port:      8080,
				Scheduler: "rr",
				Flags:     utilipvs.FlagPersistent,
			},
			bindAddr: false,
		},
		{
			// case 6, old virtual server is different from new virtual server
			oldVirtualServer: &utilipvs.VirtualServer{
				Address:   net.ParseIP("1.2.3.4"),
				Protocol:  string(v1.ProtocolSCTP),
				Port:      8080,
				Scheduler: "rr",
				Flags:     utilipvs.FlagHashed,
			},
			svcName: "bar",
			newVirtualServer: &utilipvs.VirtualServer{
				Address:   net.ParseIP("1.2.3.4"),
				Protocol:  string(v1.ProtocolSCTP),
				Port:      8080,
				Scheduler: "wlc",
				Flags:     utilipvs.FlagHashed,
			},
			bindAddr: false,
		},
		{
			// case 7, old virtual server is nil, and create new virtual server
			oldVirtualServer: nil,
			svcName:          "baz",
			newVirtualServer: &utilipvs.VirtualServer{
				Address:   net.ParseIP("1.2.3.4"),
				Protocol:  string(v1.ProtocolSCTP),
				Port:      53,
				Scheduler: "rr",
				Flags:     utilipvs.FlagHashed,
			},
			bindAddr: true,
		},
	}

	for i := range testCases {
		ipt := iptablestest.NewFake()
		ipvs := ipvstest.NewFake()
		ipset := ipsettest.NewFake(testIPSetVersion)
		proxier := NewFakeProxier(ipt, ipvs, ipset, nil, nil, false)

		proxier.netlinkHandle.EnsureDummyDevice(DefaultDummyDevice)
		if testCases[i].oldVirtualServer != nil {
			if err := proxier.ipvs.AddVirtualServer(testCases[i].oldVirtualServer); err != nil {
				t.Errorf("Case [%d], unexpected add IPVS virtual server error: %v", i, err)
			}
		}
		if err := proxier.syncService(testCases[i].svcName, testCases[i].newVirtualServer, testCases[i].bindAddr); err != nil {
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
	return ipt, NewFakeProxier(ipt, ipvs, ipset, nil, nil, false)
}

func hasJump(rules []iptablestest.Rule, destChain, ipSet string) bool {
	for _, r := range rules {
		if r[iptablestest.Jump] == destChain {
			if ipSet == "" {
				return true
			}
			if strings.Contains(r[iptablestest.MatchSet], ipSet) {
				return true
			}
		}
	}
	return false
}

func hasMasqRandomFully(rules []iptablestest.Rule) bool {
	for _, r := range rules {
		if r[iptablestest.Masquerade] == "--random-fully" {
			return true
		}
	}
	return false
}

// checkIptabless to check expected iptables chain and rules
func checkIptables(t *testing.T, ipt *iptablestest.FakeIPTables, epIpt netlinktest.ExpectedIptablesChain) {
	for epChain, epRules := range epIpt {
		rules := ipt.GetRules(epChain)
		for _, epRule := range epRules {
			if !hasJump(rules, epRule.JumpChain, epRule.MatchSet) {
				t.Errorf("Didn't find jump from chain %v match set %v to %v", epChain, epRule.MatchSet, epRule.JumpChain)
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
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, parseExcludedCIDRs([]string{"3.3.3.0/24", "4.4.4.0/24"}), false)

	// All ipvs services that were processed in the latest sync loop.
	activeServices := map[string]bool{"ipvs0": true, "ipvs1": true}
	// All ipvs services in the system.
	currentServices := map[string]*utilipvs.VirtualServer{
		// Created by kube-proxy.
		"ipvs0": {
			Address:   net.ParseIP("1.1.1.1"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      53,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
		// Created by kube-proxy.
		"ipvs1": {
			Address:   net.ParseIP("2.2.2.2"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      54,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
		// Created by an external party.
		"ipvs2": {
			Address:   net.ParseIP("3.3.3.3"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      55,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
		// Created by an external party.
		"ipvs3": {
			Address:   net.ParseIP("4.4.4.4"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      56,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
		// Created by an external party.
		"ipvs4": {
			Address:   net.ParseIP("5.5.5.5"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      57,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
		// Created by kube-proxy, but now stale.
		"ipvs5": {
			Address:   net.ParseIP("6.6.6.6"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      58,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
	}
	for v := range currentServices {
		fp.ipvs.AddVirtualServer(currentServices[v])
	}

	fp.netlinkHandle.EnsureDummyDevice(DefaultDummyDevice)
	activeBindAddrs := map[string]bool{"1.1.1.1": true, "2.2.2.2": true, "3.3.3.3": true, "4.4.4.4": true}
	// This is ipv4-only so ipv6 addresses should be ignored
	currentBindAddrs := []string{"1.1.1.1", "2.2.2.2", "3.3.3.3", "4.4.4.4", "5.5.5.5", "6.6.6.6", "fd80::1:2:3", "fd80::1:2:4"}
	for i := range currentBindAddrs {
		fp.netlinkHandle.EnsureAddressBind(currentBindAddrs[i], DefaultDummyDevice)
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
	remainingAddrs, _ := fp.netlinkHandle.ListBindAddress(DefaultDummyDevice)
	if len(remainingAddrs) != 6 {
		t.Errorf("Expected number of remaining bound addrs after cleanup to be %v. Got %v", 6, len(remainingAddrs))
	}
	// check that address "1.1.1.1", "2.2.2.2", "3.3.3.3", "4.4.4.4" are bound, ignore ipv6 addresses
	remainingAddrsMap := make(map[string]bool)
	for _, a := range remainingAddrs {
		if net.ParseIP(a).To4() == nil {
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
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, false)

	// all deleted expect ipvs2
	activeServices := map[string]bool{"ipvs2": true}
	// All ipvs services in the system.
	currentServices := map[string]*utilipvs.VirtualServer{
		"ipvs0": { // deleted with real servers
			Address:   net.ParseIP("1.1.1.1"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      53,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
		"ipvs1": { // deleted no real server
			Address:   net.ParseIP("2.2.2.2"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      54,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
		"ipvs2": { // not deleted
			Address:   net.ParseIP("3.3.3.3"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      54,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
	}

	// "ipvs0" has a real server, but it should still be deleted since the Service is deleted
	realServers := map[*utilipvs.VirtualServer]*utilipvs.RealServer{
		{
			Address:   net.ParseIP("1.1.1.1"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      53,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		}: {
			Address: net.ParseIP("10.180.0.1"),
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

	fp.netlinkHandle.EnsureDummyDevice(DefaultDummyDevice)
	activeBindAddrs := map[string]bool{"3.3.3.3": true}
	currentBindAddrs := []string{"1.1.1.1", "2.2.2.2", "3.3.3.3"}
	for i := range currentBindAddrs {
		fp.netlinkHandle.EnsureAddressBind(currentBindAddrs[i], DefaultDummyDevice)
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

	remainingAddrs, _ := fp.netlinkHandle.ListBindAddress(DefaultDummyDevice)
	if len(remainingAddrs) != 1 {
		t.Errorf("Expected number of remaining bound addrs after cleanup to be %v. Got %v", 1, len(remainingAddrs))
	}
	// check that address is "3.3.3.3"
	remainingAddrsMap := make(map[string]bool)
	for _, a := range remainingAddrs {
		if net.ParseIP(a).To4() == nil {
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
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, parseExcludedCIDRs([]string{"4.4.4.4/32"}), false)
	fp.gracefuldeleteManager = gtm

	vs := &utilipvs.VirtualServer{
		Address:   net.ParseIP("4.4.4.4"),
		Protocol:  string(v1.ProtocolUDP),
		Port:      56,
		Scheduler: "rr",
		Flags:     utilipvs.FlagHashed,
	}

	fp.ipvs.AddVirtualServer(vs)

	rss := []*utilipvs.RealServer{
		{
			Address:      net.ParseIP("10.10.10.10"),
			Port:         56,
			ActiveConn:   0,
			InactiveConn: 0,
		},
		{
			Address:      net.ParseIP("11.11.11.11"),
			Port:         56,
			ActiveConn:   0,
			InactiveConn: 0,
		},
	}
	for _, rs := range rss {
		fp.ipvs.AddRealServer(vs, rs)
	}

	fp.netlinkHandle.EnsureDummyDevice(DefaultDummyDevice)

	fp.netlinkHandle.EnsureAddressBind("4.4.4.4", DefaultDummyDevice)

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
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, parseExcludedCIDRs([]string{"3000::/64", "4000::/64"}), false)
	fp.nodeIP = net.ParseIP("::1")

	// All ipvs services that were processed in the latest sync loop.
	activeServices := map[string]bool{"ipvs0": true, "ipvs1": true}
	// All ipvs services in the system.
	currentServices := map[string]*utilipvs.VirtualServer{
		// Created by kube-proxy.
		"ipvs0": {
			Address:   net.ParseIP("1000::1"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      53,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
		// Created by kube-proxy.
		"ipvs1": {
			Address:   net.ParseIP("1000::2"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      54,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
		// Created by an external party.
		"ipvs2": {
			Address:   net.ParseIP("3000::1"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      55,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
		// Created by an external party.
		"ipvs3": {
			Address:   net.ParseIP("4000::1"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      56,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
		// Created by an external party.
		"ipvs4": {
			Address:   net.ParseIP("5000::1"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      57,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
		// Created by kube-proxy, but now stale.
		"ipvs5": {
			Address:   net.ParseIP("1000::6"),
			Protocol:  string(v1.ProtocolUDP),
			Port:      58,
			Scheduler: "rr",
			Flags:     utilipvs.FlagHashed,
		},
	}
	for v := range currentServices {
		fp.ipvs.AddVirtualServer(currentServices[v])
	}

	fp.netlinkHandle.EnsureDummyDevice(DefaultDummyDevice)
	activeBindAddrs := map[string]bool{"1000::1": true, "1000::2": true, "3000::1": true, "4000::1": true}
	// This is ipv6-only so ipv4 addresses should be ignored
	currentBindAddrs := []string{"1000::1", "1000::2", "3000::1", "4000::1", "5000::1", "1000::6", "1.1.1.1", "2.2.2.2"}
	for i := range currentBindAddrs {
		fp.netlinkHandle.EnsureAddressBind(currentBindAddrs[i], DefaultDummyDevice)
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
	remainingAddrs, _ := fp.netlinkHandle.ListBindAddress(DefaultDummyDevice)
	if len(remainingAddrs) != 6 {
		t.Errorf("Expected number of remaining bound addrs after cleanup to be %v. Got %v", 6, len(remainingAddrs))
	}
	// check that address "1000::1", "1000::2", "3000::1", "4000::1" are still bound, ignore ipv4 addresses
	remainingAddrsMap := make(map[string]bool)
	for _, a := range remainingAddrs {
		if net.ParseIP(a).To4() != nil {
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
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, false)

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
	fp.endpointsSynced = true

	// first, add multi-port service1
	fp.OnServiceAdd(service1)
	fp.syncProxyRules()
	remainingAddrs, _ := fp.netlinkHandle.ListBindAddress(DefaultDummyDevice)
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
	remainingAddrs, _ = fp.netlinkHandle.ListBindAddress(DefaultDummyDevice)
	// should still only remain address "172.16.55.4"
	if len(remainingAddrs) != 1 {
		t.Errorf("Expected number of remaining bound addrs after cleanup to be %v. Got %v", 1, len(remainingAddrs))
	} else if remainingAddrs[0] != "172.16.55.4" {
		t.Errorf("Expected remaining address should be %s, got %s", "172.16.55.4", remainingAddrs[0])
	}

	// update single-port service2 to multi-port service3
	fp.OnServiceUpdate(service2, service3)
	fp.syncProxyRules()
	remainingAddrs, _ = fp.netlinkHandle.ListBindAddress(DefaultDummyDevice)
	// should still only remain address "172.16.55.4"
	if len(remainingAddrs) != 1 {
		t.Errorf("Expected number of remaining bound addrs after cleanup to be %v. Got %v", 1, len(remainingAddrs))
	} else if remainingAddrs[0] != "172.16.55.4" {
		t.Errorf("Expected remaining address should be %s, got %s", "172.16.55.4", remainingAddrs[0])
	}

	// delete multi-port service3
	fp.OnServiceDelete(service3)
	fp.syncProxyRules()
	remainingAddrs, _ = fp.netlinkHandle.ListBindAddress(DefaultDummyDevice)
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
	fp := NewFakeProxier(ipt, ipvs, ipset, nil, nil, true)
	fp.servicesSynced = true
	fp.endpointsSynced = true
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
			Name:     utilpointer.StringPtr(""),
			Port:     utilpointer.Int32Ptr(80),
			Protocol: &tcpProtocol,
		}},
		AddressType: discovery.AddressTypeIPv4,
		Endpoints: []discovery.Endpoint{{
			Addresses:  []string{"10.0.1.1"},
			Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
			Topology:   map[string]string{"kubernetes.io/hostname": testHostname},
		}, {
			Addresses:  []string{"10.0.1.2"},
			Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
			Topology:   map[string]string{"kubernetes.io/hostname": "node2"},
		}, {
			Addresses:  []string{"10.0.1.3"},
			Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
			Topology:   map[string]string{"kubernetes.io/hostname": "node3"},
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
