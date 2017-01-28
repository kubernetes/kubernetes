package libipvs

import (
	"bytes"
	"encoding/hex"
	"net"
	"syscall"
	"testing"

	"github.com/hkwi/nlgo"
)

var testVersion = []struct {
	raw uint32
	str string
}{
	{0x00010203, "1.2.3"},
}

func TestVersion(t *testing.T) {
	for _, test := range testVersion {
		ver := Version(test.raw)
		str := ver.String()

		if str != test.str {
			t.Errorf("fail %08x: %s != %s", test.raw, str, test.str)
		}
	}
}

func TestInfoUnpack(t *testing.T) {
	testAttrs := nlgo.AttrMap{Policy: ipvs_info_policy, AttrSlice: nlgo.AttrSlice{
		{Header: syscall.NlAttr{Type: IPVS_INFO_ATTR_VERSION}, Value: nlgo.U32(0x00010203)},
		{Header: syscall.NlAttr{Type: IPVS_INFO_ATTR_CONN_TAB_SIZE}, Value: nlgo.U32(4096)},
	}}

	if info, err := unpackInfo(testAttrs); err != nil {
		t.Errorf("error Info.unpack(): %s", err)
	} else {
		if info.Version.String() != "1.2.3" {
			t.Errorf("fail Info.Version: %s != 1.2.3", info.Version.String())
		}

		if info.ConnTabSize != 4096 {
			t.Errorf("fail Info.ConnTabSize: %s != 4096", info.ConnTabSize)
		}
	}
}

func testServiceEquals(t *testing.T, testService Service, service Service) {
	if service.AddressFamily != testService.AddressFamily {
		t.Errorf("fail Service.Af: %s", service.AddressFamily)
	}
	if service.Protocol != testService.Protocol {
		t.Errorf("fail Service.Protocol: %s", service.Protocol)
	}
	if service.Address.String() != testService.Address.String() {
		t.Errorf("fail Service.Addr: %s", service.Address.String())
	}
	if service.Port != testService.Port {
		t.Errorf("fail Service.Port: %s", service.Port)
	}
	if service.SchedName != testService.SchedName {
		t.Errorf("fail Service.SchedName: %s", service.SchedName)
	}
	if service.Flags.Flags != testService.Flags.Flags || service.Flags.Mask != testService.Flags.Mask {
		t.Errorf("fail Service.Flags: %+v", service.Flags)
	}
	if service.Timeout != testService.Timeout {
		t.Errorf("fail Service.Timeout: %s", service.Timeout)
	}
	if service.Netmask != testService.Netmask {
		t.Errorf("fail Service.Netmask: %s", service.Netmask)
	}
}

func TestServiceUnpack(t *testing.T) {
	testService := Service{
		AddressFamily:        syscall.AF_INET,     // 2
		Protocol:  syscall.IPPROTO_TCP, // 6
		Address:      net.ParseIP("10.107.107.0"),
		Port:      1337,
		SchedName: "wlc",
		Flags:     Flags{0, 0},
		Timeout:   0,
		Netmask:   0x00000000,
	}
	testBytes := []byte{
		0x06, 0x00, 0x01, 0x00, // IPVS_SVC_ATTR_AF
		0x02, 0x00, 0x00, 0x00, // 2
		0x06, 0x00, 0x02, 0x00, // IPVS_SVC_ATTR_PROTOCOL
		0x06, 0x00, 0x00, 0x00, // 6
		0x08, 0x00, 0x03, 0x00, 0x0a, 0x6b, 0x6b, 0x00, // IPVS_SVC_ATTR_ADDR       10.107.107.0
		0x06, 0x00, 0x04, 0x00, 0x05, 0x39, 0x00, 0x00, // IPVS_SVC_ATTR_PORT       1337
		0x08, 0x00, 0x06, 0x00, 'w', 'l', 'c', 0x00, // IPVS_SVC_ATTR_SCHED_NAME wlc
		0x0c, 0x00, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // IPVS_SVC_ATTR_FLAGS 0:0
		0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, // IPVS_SVC_ATTR_TIMEOUT    0
		0x08, 0x00, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, // IPVS_SVC_ATTR_NETMASK    0
	}

	// pack
	packAttrs := testService.attrs(true)
	packBytes := packAttrs.Bytes()

	if !bytes.Equal(packBytes, testBytes) {
		t.Errorf("fail Dest.attrs(): \n%s", hex.Dump(packBytes))
	}

	// unpack
	if unpackedAttrs, err := ipvs_service_policy.Parse(packBytes); err != nil {
		t.Fatalf("error ipvs_service_policy.Parse: %s", err)
	} else if unpackedService, err := unpackService(unpackedAttrs.(nlgo.AttrMap)); err != nil {
		t.Fatalf("error unpackService: %s", err)
	} else {
		testServiceEquals(t, testService, unpackedService)
	}
}

func testDestEquals(t *testing.T, testDest Destination, dest Destination) {
	if dest.Address.String() != testDest.Address.String() {
		t.Errorf("fail testDest.unpack(): Addr %v", dest.Address.String())
	}
	if dest.Port != testDest.Port {
		t.Errorf("fail testDest.unpack(): Port %v", dest.Port)
	}
	if dest.FwdMethod != testDest.FwdMethod {
		t.Errorf("fail testDest.unpack(): FwdMethod %v", dest.FwdMethod)
	}
	if dest.Weight != testDest.Weight {
		t.Errorf("fail testDest.unpack(): Weight %v", dest.Weight)
	}
	if dest.UThresh != testDest.UThresh {
		t.Errorf("fail testDest.unpack(): UThresh %v", dest.UThresh)
	}
	if dest.LThresh != testDest.LThresh {
		t.Errorf("fail testDest.unpack(): LThresh %v", dest.LThresh)
	}
}

func TestDest(t *testing.T) {
	testService := Service{
		AddressFamily: syscall.AF_INET6,
	}
	testDest := Destination{
		Address: net.ParseIP("2001:db8:6b:6b::0"),
		Port: 1337,

		FwdMethod: IP_VS_CONN_F_TUNNEL,
		Weight:    10,
		UThresh:   1000,
		LThresh:   0,
	}
	testAttrs := nlgo.AttrSlice{
		nlattr(IPVS_DEST_ATTR_ADDR, nlgo.Binary([]byte{0x20, 0x01, 0x0d, 0xb8, 0x00, 0x6b, 0x00, 0x6b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00})),
		nlattr(IPVS_DEST_ATTR_PORT, nlgo.U16(0x3905)),
		nlattr(IPVS_DEST_ATTR_FWD_METHOD, nlgo.U32(IP_VS_CONN_F_TUNNEL)),
		nlattr(IPVS_DEST_ATTR_WEIGHT, nlgo.U32(10)),
		nlattr(IPVS_DEST_ATTR_U_THRESH, nlgo.U32(1000)),
		nlattr(IPVS_DEST_ATTR_L_THRESH, nlgo.U32(0)),
	}

	// pack
	packAttrs := testDest.attrs(&testService, true)
	packBytes := packAttrs.Bytes()

	if !bytes.Equal(packBytes, testAttrs.Bytes()) {
		t.Errorf("fail Dest.attrs(): \n%s", hex.Dump(packBytes))
	}

	// unpack
	if unpackedAttrs, err := ipvs_dest_policy.Parse(packBytes); err != nil {
		t.Fatalf("error ipvs_dest_policy.Parse: %s", err)
	} else if unpackedDest, err := unpackDest(testService, unpackedAttrs.(nlgo.AttrMap)); err != nil {
		t.Fatalf("error unpackDest: %s", err)
	} else {
		testDestEquals(t, testDest, unpackedDest)
	}
}
