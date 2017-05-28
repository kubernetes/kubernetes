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
	_ "fmt"
	"net"
	"reflect"
	"syscall"
	"testing"

	"github.com/docker/libnetwork/ipvs"
	"k8s.io/kubernetes/pkg/util/dbus"
	"k8s.io/kubernetes/pkg/util/exec"
)

func TestAlias(t *testing.T) {
	execer := exec.New()
	dbus := dbus.New()
	run := New(execer, dbus)
	defer run.Destroy()
	err := run.CreateAliasDevice(AliasDevice)
	if err != nil {
		t.Errorf("expected create aliasdevice success, got %v", err)
	}
	// Create again
	err = run.CreateAliasDevice(AliasDevice)
	if err != nil {
		t.Errorf("expected create aliasdevice again success, got %v", err)
	}

	err = run.CheckAliasDevice(AliasDevice)
	if err != nil {
		t.Errorf("expected check aliasdevice success, got %v", err)
	}
	svc := Service{
		Address:  net.ParseIP("10.20.30.41"),
		Port:     uint16(12345),
		Protocol: string("TCP"),
	}
	err = run.SetAlias(&svc)
	if err != nil {
		t.Errorf("expected setalias success, got %v", err)
	}
	//SetAlias again
	err = run.SetAlias(&svc)
	if err != nil {
		t.Errorf("expected setalias again success, got %v", err)
	}

	err = run.UnSetAlias(&svc)
	if err != nil {
		t.Errorf("expected unsetalias success, got %v", err)
	}
	//UnSetAlias again
	err = run.UnSetAlias(&svc)
	if err != nil {
		t.Errorf("expected unsetalias again success, got %v", err)
	}

	err = run.DeleteAliasDevice(AliasDevice)
	if err != nil {
		t.Errorf("expected delete aliasdevice success, got %v", err)
	}
	err = run.DeleteAliasDevice(AliasDevice)
	if err != nil {
		t.Errorf("expected delete aliasdevice again success, got %v", err)
	}
}

func deleteAliasDevice(t *testing.T) {
	execer := exec.New()
	dbus := dbus.New()
	run := New(execer, dbus)
	defer run.Destroy()
	err := run.DeleteAliasDevice(AliasDevice)
	if err != nil {
		t.Errorf("expected delete aliasdevice success, got %v", err)
	}
}

var ServiceTests = []struct {
	ipvsService ipvs.Service
	service     Service
}{
	{
		ipvs.Service{
			Protocol:      syscall.IPPROTO_TCP,
			Port:          80,
			FWMark:        0,
			SchedName:     "",
			Flags:         0,
			Timeout:       0,
			Netmask:       0xffffffff,
			AddressFamily: syscall.AF_INET,
			Address:       nil,
			PEName:        "",
		},
		Service{
			Address:   net.ParseIP("0.0.0.0"),
			Protocol:  "TCP",
			Port:      80,
			Scheduler: "",
			Flags:     0,
			Timeout:   0,
		},
	},
	{
		ipvs.Service{
			Protocol:      syscall.IPPROTO_UDP,
			Port:          33434,
			FWMark:        0,
			SchedName:     "wlc",
			Flags:         1234,
			Timeout:       100,
			Netmask:       128,
			AddressFamily: syscall.AF_INET6,
			Address:       net.ParseIP("2012::beef"),
			PEName:        "",
		},
		Service{
			Address:   net.ParseIP("2012::beef"),
			Protocol:  "UDP",
			Port:      33434,
			Scheduler: "wlc",
			Flags:     1234,
			Timeout:   100,
		},
	},
	{
		ipvs.Service{
			Protocol:      0,
			Port:          0,
			FWMark:        0,
			SchedName:     "lc",
			Flags:         0,
			Timeout:       0,
			Netmask:       0xffffffff,
			AddressFamily: syscall.AF_INET,
			Address:       net.ParseIP("1.2.3.4"),
			PEName:        "",
		},
		Service{
			Address:   net.ParseIP("1.2.3.4"),
			Protocol:  "",
			Port:      0,
			Scheduler: "lc",
			Flags:     0,
			Timeout:   0,
		},
	},
	{
		ipvs.Service{
			Protocol:      0,
			Port:          0,
			FWMark:        0,
			SchedName:     "wrr",
			Flags:         0,
			Timeout:       0,
			Netmask:       128,
			AddressFamily: syscall.AF_INET6,
			Address:       nil,
			PEName:        "",
		},
		Service{
			Address:   net.ParseIP("::0"),
			Protocol:  "",
			Port:      0,
			Scheduler: "wrr",
			Flags:     0,
			Timeout:   0,
		},
	},
}

func TestStringToProtocolNumber(t *testing.T) {
	for _, test := range ServiceTests {
		got := ToProtocolNumber(test.service.Protocol)
		if got != test.ipvsService.Protocol {
			t.Errorf("ToProtocolNumber() failed - got %#v, want %#v",
				got, test.service.Protocol)
		}
	}
}

func TestProtocolNumberToString(t *testing.T) {
	for _, test := range ServiceTests {
		got := String(IPProto(test.ipvsService.Protocol))
		if got != test.service.Protocol {
			t.Errorf("ProtocolNumberToString() failed - got %#v, want %#v",
				got, test.ipvsService.Protocol)
		}
	}
}

func TestIPVSServiceToService(t *testing.T) {
	for _, test := range ServiceTests {
		got, err := toService(&test.ipvsService)
		if err != nil {
			t.Errorf("expected ipvsService to service success, got %v", err)
		}

		if !reflect.DeepEqual(*got, test.service) {
			t.Errorf("toService() failed - got %#v, want %#v",
				*got, test.service)
		}
	}
}

var IPVSServiceTests = []struct {
	ipvsService ipvs.Service
	service     Service
}{
	{
		ipvs.Service{
			Protocol:      syscall.IPPROTO_TCP,
			Port:          80,
			FWMark:        0,
			SchedName:     "",
			Flags:         0,
			Timeout:       0,
			Netmask:       0xffffffff,
			AddressFamily: syscall.AF_INET,
			Address:       net.ParseIP("0.0.0.0"),
			PEName:        "",
		},
		Service{
			Address:   net.ParseIP("0.0.0.0"),
			Protocol:  "TCP",
			Port:      80,
			Scheduler: "",
			Flags:     0,
			Timeout:   0,
		},
	},
	{
		ipvs.Service{
			Protocol:      syscall.IPPROTO_UDP,
			Port:          33434,
			FWMark:        0,
			SchedName:     "wlc",
			Flags:         1234,
			Timeout:       100,
			Netmask:       128,
			AddressFamily: syscall.AF_INET6,
			Address:       net.ParseIP("2012::beef"),
			PEName:        "",
		},
		Service{
			Address:   net.ParseIP("2012::beef"),
			Protocol:  "UDP",
			Port:      33434,
			Scheduler: "wlc",
			Flags:     1234,
			Timeout:   100,
		},
	},
	{
		ipvs.Service{
			Protocol:      0,
			Port:          0,
			FWMark:        0,
			SchedName:     "lc",
			Flags:         0,
			Timeout:       0,
			Netmask:       0xffffffff,
			AddressFamily: syscall.AF_INET,
			Address:       net.ParseIP("1.2.3.4"),
			PEName:        "",
		},
		Service{
			Address:   net.ParseIP("1.2.3.4"),
			Protocol:  "",
			Port:      0,
			Scheduler: "lc",
			Flags:     0,
			Timeout:   0,
		},
	},
	{
		ipvs.Service{
			Protocol:      0,
			Port:          0,
			FWMark:        0,
			SchedName:     "wrr",
			Flags:         0,
			Timeout:       0,
			Netmask:       128,
			AddressFamily: syscall.AF_INET6,
			Address:       net.ParseIP("::0"),
			PEName:        "",
		},
		Service{
			Address:   net.ParseIP("::0"),
			Protocol:  "",
			Port:      0,
			Scheduler: "wrr",
			Flags:     0,
			Timeout:   0,
		},
	},
}

func TestServiceToIPVSService(t *testing.T) {
	for _, test := range IPVSServiceTests {
		got := NewIpvsService(&test.service)

		if !reflect.DeepEqual(*got, test.ipvsService) {
			t.Errorf("NewIpvsService() failed - got %#v, want %#v",
				*got, test.ipvsService)
		}
	}
}

var ServiceEqualtest = []struct {
	svcA Service
	svcB Service
}{
	{
		Service{
			Address:   net.ParseIP("2012::beef"),
			Protocol:  "",
			Port:      0,
			Scheduler: "wrr",
			Flags:     0,
			Timeout:   0,
		},
		Service{
			Address:   net.ParseIP("2012::beef"),
			Protocol:  "",
			Port:      0,
			Scheduler: "wrr",
			Flags:     0,
			Timeout:   0,
		},
	},
	{
		Service{
			Address:   net.ParseIP("2012::beef"),
			Protocol:  "",
			Port:      0,
			Scheduler: "wrr",
			Flags:     0,
			Timeout:   0,
		},
		Service{
			Address:   net.ParseIP("2012::beee"),
			Protocol:  "",
			Port:      0,
			Scheduler: "wrr",
			Flags:     0,
			Timeout:   0,
		},
	},
	{
		Service{
			Address:   net.ParseIP("2012::beef"),
			Protocol:  "",
			Port:      0,
			Scheduler: "wrr",
			Flags:     0,
			Timeout:   0,
		},
		Service{
			Address:   net.ParseIP("2012::beef"),
			Protocol:  "TCP",
			Port:      0,
			Scheduler: "wrr",
			Flags:     0,
			Timeout:   0,
		},
	},
	{
		Service{
			Address:   net.ParseIP("1.2.3.4"),
			Protocol:  "TCP",
			Port:      80,
			Scheduler: "wlc",
			Flags:     0,
			Timeout:   0,
		},
		Service{
			Address:   net.ParseIP("1.2.3.4"),
			Protocol:  "TCP",
			Port:      8080,
			Scheduler: "wlc",
			Flags:     0,
			Timeout:   0,
		},
	},
	{
		Service{
			Address:   net.ParseIP("2012::beef"),
			Protocol:  "",
			Port:      0,
			Scheduler: "wrr",
			Flags:     0,
			Timeout:   0,
		},
		Service{
			Address:   net.ParseIP("2012::beef"),
			Protocol:  "",
			Port:      0,
			Scheduler: "rr",
			Flags:     0,
			Timeout:   0,
		},
	},
	{
		Service{
			Address:   net.ParseIP("2012::beef"),
			Protocol:  "",
			Port:      0,
			Scheduler: "wrr",
			Flags:     0,
			Timeout:   0,
		},
		Service{
			Address:   net.ParseIP("2012::beef"),
			Protocol:  "",
			Port:      0,
			Scheduler: "wrr",
			Flags:     0,
			Timeout:   10800,
		},
	},
}

func TestEqual(t *testing.T) {
	equal := ServiceEqualtest[0].svcA.Equal(&ServiceEqualtest[0].svcB)
	if !equal {
		t.Errorf("expect the two services same")
	}
	equal = ServiceEqualtest[1].svcA.Equal(&ServiceEqualtest[1].svcB)
	if equal {
		t.Errorf("Did not expect the two services same")
	}
}

var DestinationTests = []struct {
	ipvsDestination ipvs.Destination
	destination     Destination
}{
	{
		ipvs.Destination{
			Port:            54321,
			ConnectionFlags: 0,
			Weight:          1,
			Address:         net.ParseIP("1.2.3.4"),
		},
		Destination{
			Address: net.ParseIP("1.2.3.4"),
			Port:    54321,
			Weight:  1,
		},
	},
	{
		ipvs.Destination{
			Port:            53,
			ConnectionFlags: 0,
			Weight:          1,
			Address:         net.ParseIP("2002::cafe"),
		},
		Destination{
			Address: net.ParseIP("2002::cafe"),
			Port:    53,
			Weight:  1,
		},
	},
}

func TestIPVSDestinationToDestination(t *testing.T) {
	for _, test := range DestinationTests {
		got, err := toDestination(&test.ipvsDestination)
		if err != nil {
			t.Errorf("expected ipvsDestination to destination success, got %v", err)
		}
		if !reflect.DeepEqual(*got, test.destination) {
			t.Errorf("toDestination() failed - got %#v, want %#v",
				*got, test.destination)
		}
	}
}

func TestDestinationToIPVSDestination(t *testing.T) {
	for _, test := range DestinationTests {
		got := NewIPVSDestination(&test.destination)
		if !reflect.DeepEqual(*got, test.ipvsDestination) {
			t.Errorf("NewIPVSDestination() failed - got %#v, want %#v",
				*got, test.ipvsDestination)
		}
	}
}

var IPToIntTests = []struct {
	ip  net.IP
	num uint32
}{
	{
		ip:  net.ParseIP("1.2.3.4"),
		num: 16909060,
	},
	{
		ip:  net.ParseIP("2002::cafe"),
		num: 51966,
	},
}

func TestIPToInt(t *testing.T) {
	got := IPtoInt(IPToIntTests[0].ip)
	if got != IPToIntTests[0].num {
		t.Errorf("IPtoInt() failed - got %#v, want %#v",
			got, IPToIntTests[0].num)
	}
	got = IPtoInt(IPToIntTests[0].ip.To4())
	if got != IPToIntTests[0].num {
		t.Errorf("IPtoInt() failed - got %#v, want %#v",
			got, IPToIntTests[0].num)
	}
	got = IPtoInt(IPToIntTests[1].ip)
	if got != IPToIntTests[1].num {
		t.Errorf("IPtoInt() failed - got %#v, want %#v",
			got, IPToIntTests[1].num)
	}
}

var ServiceFuncTests = []Service{
	Service{
		Address:   net.ParseIP("10.109.22.11"),
		Protocol:  "TCP",
		Port:      13242,
		Scheduler: "lc",
		Flags:     0,
		Timeout:   0,
	},
	Service{
		Address:   net.ParseIP("2012::beef"),
		Protocol:  "UDP",
		Port:      33434,
		Scheduler: "wlc",
		Flags:     1,
		Timeout:   100,
	},
	Service{
		Address:   net.ParseIP("10.108.23.44"),
		Protocol:  "UDP",
		Port:      12345,
		Scheduler: "",
		Flags:     1,
		Timeout:   10800,
	},
}

func compareService(svc *Service, got *Service) bool {
	if svc.Scheduler == "" {
		svc.Scheduler = DefaultIpvsScheduler
	}
	return svc.Address.Equal(got.Address) &&
		svc.Protocol == got.Protocol &&
		svc.Port == got.Port &&
		svc.Scheduler == got.Scheduler &&
		svc.Timeout == got.Timeout
}

func checkservice(t *testing.T, i Interface, s *Service, check bool) {
	svcs, err := i.GetServices()
	if err != nil {
		t.Errorf("expected get all services success, got %v", err)
	}

	var found bool = false
	for _, svc := range svcs {
		if compareService(s, svc) {
			found = true
			break
		}
	}
	switch check {
	case true:
		if !found {
			t.Errorf("Did not find the service %s in ipvs output")
		}
	case false:
		if found {
			t.Errorf("Did not expect the service %s in ipvs output")
		}
	}
}

func TestService(t *testing.T) {
	execer := exec.New()
	dbus := dbus.New()
	run := New(execer, dbus)
	err := run.InitIpvsInterface()
	if err != nil {
		t.Errorf("expected init ipvs interface success, got %v", err)
	}
	defer run.Destroy()
	for _, svc := range ServiceFuncTests {
		err := run.AddService(&svc)
		if err != nil {
			t.Errorf("expected add service success, got %v", err)
		}
		_, err = run.GetService(&svc)
		if err != nil {
			t.Errorf("expected get service success, got %v", err)
		}
	}

	for _, svc := range ServiceFuncTests {
		checkservice(t, run, &svc, true)

		err := run.DeleteService(&svc)
		if err != nil {
			t.Errorf("expected delete service success, got %v", err)
		}
		checkservice(t, run, &svc, false)
	}
	deleteAliasDevice(t)
}

func TestFlush(t *testing.T) {
	execer := exec.New()
	dbus := dbus.New()
	run := New(execer, dbus)
	err := run.InitIpvsInterface()
	if err != nil {
		t.Errorf("expected init ipvs interface success, got %v", err)
	}
	defer run.Destroy()
	for _, svc := range ServiceFuncTests {
		err := run.AddService(&svc)
		if err != nil {
			t.Errorf("expected add service success, got %v", err)
		}
		_, err = run.GetService(&svc)
		if err != nil {
			t.Errorf("expected get service success, got %v", err)
		}
	}

	err = run.Flush()
	if err != nil {
		t.Errorf("expected delete all services success, got %v", err)
	}
	for _, svc := range ServiceFuncTests {
		checkservice(t, run, &svc, false)
	}
	deleteAliasDevice(t)
}

var DestinationFuncTests = []Destination{
	Destination{
		Address: net.ParseIP("10.232.12.32"),
		Port:    32144,
		Weight:  1,
	},
	Destination{
		Address: net.ParseIP("10.233.23.211"),
		Port:    23123,
		Weight:  1,
	},
}

func compareDestination(dest *Destination, got *Destination) bool {
	return dest.Address.Equal(got.Address) &&
		dest.Port == got.Port &&
		dest.Weight == got.Weight
}

func checkdestination(t *testing.T, i Interface, s *Service, d *Destination, check bool) {
	dests, err := i.GetDestinations(nil)
	if err == nil {
		t.Errorf("expected get destinations failed, got %v", err)
	}

	dests, err = i.GetDestinations(s)
	if err != nil {
		t.Errorf("expected get all services success, got %v", err)
	}

	var found bool = false
	for _, dest := range dests {
		if compareDestination(d, dest) {
			found = true
			break
		}
	}
	switch check {
	case true:
		if !found {
			t.Errorf("Did not find the destination %s in ipvs output")
		}
	case false:
		if found {
			t.Errorf("Did not expect the destination %s in ipvs output")
		}
	}
}

func TestDestination(t *testing.T) {
	execer := exec.New()
	dbus := dbus.New()
	run := New(execer, dbus)
	err := run.InitIpvsInterface()
	if err != nil {
		t.Errorf("expected init ipvs interface success, got %v", err)
	}
	defer run.Destroy()
	s := ServiceFuncTests[0]

	err = run.AddService(&s)
	if err != nil {
		t.Errorf("expected add service success, got %v", err)
	}
	checkservice(t, run, &s, true)
	for _, dest := range DestinationFuncTests {
		err := run.AddDestination(nil, &dest)
		if err == nil {
			t.Errorf("expected add destination failed, got %v", err)
		}
		err = run.AddDestination(&s, &dest)
		if err != nil {
			t.Errorf("expected add destination success, got %v", err)
		}
		checkdestination(t, run, &s, &dest, true)

		err = run.DeleteDestination(nil, &dest)
		if err == nil {
			t.Errorf("expected delete destination failed, got %v", err)
		}
		err = run.DeleteDestination(&s, &dest)
		if err != nil {
			t.Errorf("expected delete destination success, got %v", err)
		}
		checkdestination(t, run, &s, &dest, false)
	}
	err = run.DeleteService(&s)
	if err != nil {
		t.Errorf("expected delete service success, got %v", err)
	}
	checkservice(t, run, &s, false)
	deleteAliasDevice(t)
}
