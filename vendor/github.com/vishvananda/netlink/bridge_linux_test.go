package netlink

import (
	"fmt"
	"io/ioutil"
	"testing"
)

func TestBridgeVlan(t *testing.T) {
	minKernelRequired(t, 3, 10)

	tearDown := setUpNetlinkTest(t)
	defer tearDown()
	if err := remountSysfs(); err != nil {
		t.Fatal(err)
	}
	bridgeName := "foo"
	bridge := &Bridge{LinkAttrs: LinkAttrs{Name: bridgeName}}
	if err := LinkAdd(bridge); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(fmt.Sprintf("/sys/devices/virtual/net/%s/bridge/vlan_filtering", bridgeName), []byte("1"), 0644); err != nil {
		t.Fatal(err)
	}
	if vlanMap, err := BridgeVlanList(); err != nil {
		t.Fatal(err)
	} else {
		if len(vlanMap) != 1 {
			t.Fatal()
		}
		if vInfo, ok := vlanMap[int32(bridge.Index)]; !ok {
			t.Fatal("vlanMap should include foo port vlan info")
		} else {
			if len(vInfo) != 1 {
				t.Fatal()
			} else {
				if !vInfo[0].EngressUntag() || !vInfo[0].PortVID() || vInfo[0].Vid != 1 {
					t.Fatalf("bridge vlan show get wrong return %s", vInfo[0].String())
				}
			}
		}
	}
	dummy := &Dummy{LinkAttrs: LinkAttrs{Name: "dum1"}}
	if err := LinkAdd(dummy); err != nil {
		t.Fatal(err)
	}
	if err := LinkSetMaster(dummy, bridge); err != nil {
		t.Fatal(err)
	}
	if err := BridgeVlanAdd(dummy, 2, false, false, false, false); err != nil {
		t.Fatal(err)
	}
	if err := BridgeVlanAdd(dummy, 3, true, true, false, false); err != nil {
		t.Fatal(err)
	}
	if vlanMap, err := BridgeVlanList(); err != nil {
		t.Fatal(err)
	} else {
		if len(vlanMap) != 2 {
			t.Fatal()
		}
		if vInfo, ok := vlanMap[int32(bridge.Index)]; !ok {
			t.Fatal("vlanMap should include foo port vlan info")
		} else {
			if "[{Flags:6 Vid:1}]" != fmt.Sprintf("%v", vInfo) {
				t.Fatalf("unexpected result %v", vInfo)
			}
		}
		if vInfo, ok := vlanMap[int32(dummy.Index)]; !ok {
			t.Fatal("vlanMap should include dum1 port vlan info")
		} else {
			if "[{Flags:4 Vid:1} {Flags:0 Vid:2} {Flags:6 Vid:3}]" != fmt.Sprintf("%v", vInfo) {
				t.Fatalf("unexpected result %v", vInfo)
			}
		}
	}
}
