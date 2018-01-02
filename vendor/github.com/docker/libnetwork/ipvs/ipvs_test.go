// +build linux

package ipvs

import (
	"net"
	"syscall"
	"testing"

	"github.com/docker/libnetwork/testutils"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/vishvananda/netlink"
	"github.com/vishvananda/netlink/nl"
)

var (
	schedMethods = []string{
		RoundRobin,
		LeastConnection,
		DestinationHashing,
		SourceHashing,
	}

	protocols = []string{
		"TCP",
		"UDP",
		"FWM",
	}

	fwdMethods = []uint32{
		ConnectionFlagMasq,
		ConnectionFlagTunnel,
		ConnectionFlagDirectRoute,
	}

	fwdMethodStrings = []string{
		"Masq",
		"Tunnel",
		"Route",
	}
)

func lookupFwMethod(fwMethod uint32) string {

	switch fwMethod {
	case ConnectionFlagMasq:
		return fwdMethodStrings[0]
	case ConnectionFlagTunnel:
		return fwdMethodStrings[1]
	case ConnectionFlagDirectRoute:
		return fwdMethodStrings[2]
	}
	return ""
}

func checkDestination(t *testing.T, i *Handle, s *Service, d *Destination, checkPresent bool) {
	var dstFound bool

	dstArray, err := i.GetDestinations(s)
	require.NoError(t, err)

	for _, dst := range dstArray {
		if dst.Address.Equal(d.Address) && dst.Port == d.Port && lookupFwMethod(dst.ConnectionFlags) == lookupFwMethod(d.ConnectionFlags) {
			dstFound = true
			break
		}
	}

	switch checkPresent {
	case true: //The test expects the service to be present
		if !dstFound {

			t.Fatalf("Did not find the service %s in ipvs output", d.Address.String())
		}
	case false: //The test expects that the service should not be present
		if dstFound {
			t.Fatalf("Did not find the destination %s fwdMethod %s in ipvs output", d.Address.String(), lookupFwMethod(d.ConnectionFlags))
		}
	}

}

func checkService(t *testing.T, i *Handle, s *Service, checkPresent bool) {

	svcArray, err := i.GetServices()
	require.NoError(t, err)

	var svcFound bool

	for _, svc := range svcArray {

		if svc.Protocol == s.Protocol && svc.Address.String() == s.Address.String() && svc.Port == s.Port {
			svcFound = true
			break
		}
	}

	switch checkPresent {
	case true: //The test expects the service to be present
		if !svcFound {

			t.Fatalf("Did not find the service %s in ipvs output", s.Address.String())
		}
	case false: //The test expects that the service should not be present
		if svcFound {
			t.Fatalf("Did not expect the service %s in ipvs output", s.Address.String())
		}
	}

}

func TestGetFamily(t *testing.T) {
	if testutils.RunningOnCircleCI() {
		t.Skip("Skipping as not supported on CIRCLE CI kernel")
	}

	id, err := getIPVSFamily()
	require.NoError(t, err)
	assert.NotEqual(t, 0, id)
}

func TestService(t *testing.T) {
	if testutils.RunningOnCircleCI() {
		t.Skip("Skipping as not supported on CIRCLE CI kernel")
	}

	defer testutils.SetupTestOSContext(t)()

	i, err := New("")
	require.NoError(t, err)

	for _, protocol := range protocols {
		for _, schedMethod := range schedMethods {

			s := Service{
				AddressFamily: nl.FAMILY_V4,
				SchedName:     schedMethod,
			}

			switch protocol {
			case "FWM":
				s.FWMark = 1234
			case "TCP":
				s.Protocol = syscall.IPPROTO_TCP
				s.Port = 80
				s.Address = net.ParseIP("1.2.3.4")
				s.Netmask = 0xFFFFFFFF
			case "UDP":
				s.Protocol = syscall.IPPROTO_UDP
				s.Port = 53
				s.Address = net.ParseIP("2.3.4.5")
			}

			err := i.NewService(&s)
			assert.NoError(t, err)
			checkService(t, i, &s, true)
			for _, updateSchedMethod := range schedMethods {
				if updateSchedMethod == schedMethod {
					continue
				}

				s.SchedName = updateSchedMethod
				err = i.UpdateService(&s)
				assert.NoError(t, err)
				checkService(t, i, &s, true)

				scopy, err := i.GetService(&s)
				assert.NoError(t, err)
				assert.Equal(t, (*scopy).Address.String(), s.Address.String())
				assert.Equal(t, (*scopy).Port, s.Port)
				assert.Equal(t, (*scopy).Protocol, s.Protocol)
			}

			err = i.DelService(&s)
			assert.NoError(t, err)
			checkService(t, i, &s, false)
		}
	}

	svcs := []Service{
		{
			AddressFamily: nl.FAMILY_V4,
			SchedName:     RoundRobin,
			Protocol:      syscall.IPPROTO_TCP,
			Port:          80,
			Address:       net.ParseIP("10.20.30.40"),
			Netmask:       0xFFFFFFFF,
		},
		{
			AddressFamily: nl.FAMILY_V4,
			SchedName:     LeastConnection,
			Protocol:      syscall.IPPROTO_UDP,
			Port:          8080,
			Address:       net.ParseIP("10.20.30.41"),
			Netmask:       0xFFFFFFFF,
		},
	}
	// Create services for testing flush
	for _, svc := range svcs {
		if !i.IsServicePresent(&svc) {
			err = i.NewService(&svc)
			assert.NoError(t, err)
			checkService(t, i, &svc, true)
		} else {
			t.Errorf("svc: %v exists", svc)
		}
	}
	err = i.Flush()
	assert.NoError(t, err)
	got, err := i.GetServices()
	assert.NoError(t, err)
	if len(got) != 0 {
		t.Errorf("Unexpected services after flush")
	}
}

func createDummyInterface(t *testing.T) {
	if testutils.RunningOnCircleCI() {
		t.Skip("Skipping as not supported on CIRCLE CI kernel")
	}

	dummy := &netlink.Dummy{
		LinkAttrs: netlink.LinkAttrs{
			Name: "dummy",
		},
	}

	err := netlink.LinkAdd(dummy)
	require.NoError(t, err)

	dummyLink, err := netlink.LinkByName("dummy")
	require.NoError(t, err)

	ip, ipNet, err := net.ParseCIDR("10.1.1.1/24")
	require.NoError(t, err)

	ipNet.IP = ip

	ipAddr := &netlink.Addr{IPNet: ipNet, Label: ""}
	err = netlink.AddrAdd(dummyLink, ipAddr)
	require.NoError(t, err)
}

func TestDestination(t *testing.T) {
	defer testutils.SetupTestOSContext(t)()

	createDummyInterface(t)
	i, err := New("")
	require.NoError(t, err)

	for _, protocol := range protocols {

		s := Service{
			AddressFamily: nl.FAMILY_V4,
			SchedName:     RoundRobin,
		}

		switch protocol {
		case "FWM":
			s.FWMark = 1234
		case "TCP":
			s.Protocol = syscall.IPPROTO_TCP
			s.Port = 80
			s.Address = net.ParseIP("1.2.3.4")
			s.Netmask = 0xFFFFFFFF
		case "UDP":
			s.Protocol = syscall.IPPROTO_UDP
			s.Port = 53
			s.Address = net.ParseIP("2.3.4.5")
		}

		err := i.NewService(&s)
		assert.NoError(t, err)
		checkService(t, i, &s, true)

		s.SchedName = ""
		for _, fwdMethod := range fwdMethods {
			d1 := Destination{
				AddressFamily:   nl.FAMILY_V4,
				Address:         net.ParseIP("10.1.1.2"),
				Port:            5000,
				Weight:          1,
				ConnectionFlags: fwdMethod,
			}

			err := i.NewDestination(&s, &d1)
			assert.NoError(t, err)
			checkDestination(t, i, &s, &d1, true)
			d2 := Destination{
				AddressFamily:   nl.FAMILY_V4,
				Address:         net.ParseIP("10.1.1.3"),
				Port:            5000,
				Weight:          1,
				ConnectionFlags: fwdMethod,
			}

			err = i.NewDestination(&s, &d2)
			assert.NoError(t, err)
			checkDestination(t, i, &s, &d2, true)

			d3 := Destination{
				AddressFamily:   nl.FAMILY_V4,
				Address:         net.ParseIP("10.1.1.4"),
				Port:            5000,
				Weight:          1,
				ConnectionFlags: fwdMethod,
			}

			err = i.NewDestination(&s, &d3)
			assert.NoError(t, err)
			checkDestination(t, i, &s, &d3, true)

			for _, updateFwdMethod := range fwdMethods {
				if updateFwdMethod == fwdMethod {
					continue
				}
				d1.ConnectionFlags = updateFwdMethod
				err = i.UpdateDestination(&s, &d1)
				assert.NoError(t, err)
				checkDestination(t, i, &s, &d1, true)

				d2.ConnectionFlags = updateFwdMethod
				err = i.UpdateDestination(&s, &d2)
				assert.NoError(t, err)
				checkDestination(t, i, &s, &d2, true)

				d3.ConnectionFlags = updateFwdMethod
				err = i.UpdateDestination(&s, &d3)
				assert.NoError(t, err)
				checkDestination(t, i, &s, &d3, true)
			}

			err = i.DelDestination(&s, &d1)
			assert.NoError(t, err)
			err = i.DelDestination(&s, &d2)
			assert.NoError(t, err)
			err = i.DelDestination(&s, &d3)
			assert.NoError(t, err)
			checkDestination(t, i, &s, &d3, false)

		}
	}
}
