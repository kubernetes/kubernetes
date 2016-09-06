// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package netlink

import (
	"bytes"
	"net"
	"reflect"
	"syscall"
	"testing"
)

type ipvsInfo struct {
	Version       uint32 `netlink:"attr:1"`
	ConnTableSize uint32 `netlink:"attr:2"`
}

type ipvsStats struct {
	Conns      uint32 `netlink:"attr:1"`
	PacketsIn  uint32 `netlink:"attr:2"`
	PacketsOut uint32 `netlink:"attr:3"`
	BytesIn    uint64 `netlink:"attr:4"`
	BytesOut   uint64 `netlink:"attr:5"`
	CPS        uint32 `netlink:"attr:6"`
	PPSIn      uint32 `netlink:"attr:7"`
	PPSOut     uint32 `netlink:"attr:8"`
	BPSIn      uint32 `netlink:"attr:9"`
	BPSOut     uint32 `netlink:"attr:10"`
}

type ipvsServiceStats struct {
	Ignore bool
	ipvsStats
}

type ipvsService struct {
	AddrFamily        uint16            `netlink:"attr:1"`
	Protocol          uint16            `netlink:"attr:2,omitempty,optional"`
	Address           net.IP            `netlink:"attr:3,omitempty,optional"`
	Port              uint16            `netlink:"attr:4,network,omitempty,optional"`
	FirewallMark      uint32            `netlink:"attr:5,omitempty,optional"`
	Scheduler         string            `netlink:"attr:6"`
	Flags             [8]byte           `netlink:"attr:7"`
	Timeout           uint32            `netlink:"attr:8"`
	Netmask           uint32            `netlink:"attr:9"`
	Stats             *ipvsServiceStats `netlink:"attr:10,optional"`
	PersistenceEngine string            `netlink:"attr:11,omitempty,optional"`
}

type ipvsCommand struct {
	Service *ipvsService `netlink:"attr:1"`
}

type ipvsIPAddr struct {
	IP net.IP `netlink:"attr:1"`
}

var (
	nlmIPVSInfo = []byte{
		0x24, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00,
		0xc6, 0x20, 0xc8, 0x55, 0x33, 0x28, 0x00, 0x00,
		0x0e, 0x01, 0x00, 0x00, 0x08, 0x00, 0x01, 0x00,
		0x01, 0x02, 0x01, 0x00, 0x08, 0x00, 0x02, 0x00,
		0x00, 0x10, 0x00, 0x00,
	}

	nlmIPVSAddService = []byte{
		0x68, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x01, 0x01, 0x00, 0x00, 0x54, 0x00, 0x01, 0x00,
		0x06, 0x00, 0x01, 0x00, 0x02, 0x00, 0x00, 0x00,
		0x06, 0x00, 0x02, 0x00, 0x06, 0x00, 0x00, 0x00,
		0x14, 0x00, 0x03, 0x00, 0x01, 0x01, 0x01, 0x01,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x04, 0x00,
		0x50, 0x00, 0x00, 0x00, 0x08, 0x00, 0x06, 0x00,
		0x77, 0x6c, 0x63, 0x00, 0x0c, 0x00, 0x07, 0x00,
		0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff,
		0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x08, 0x00, 0x09, 0x00, 0xff, 0xff, 0xff, 0xff,
	}

	nlmIPVSService = []byte{
		0xc4, 0x00, 0x00, 0x00, 0x16, 0x00, 0x02, 0x00,
		0xb0, 0xb3, 0xc8, 0x55, 0x79, 0x02, 0x00, 0x00,
		0x01, 0x01, 0x00, 0x00, 0xb0, 0x00, 0x01, 0x00,
		0x06, 0x00, 0x01, 0x00, 0x02, 0x00, 0x00, 0x00,
		0x06, 0x00, 0x02, 0x00, 0x11, 0x00, 0x00, 0x00,
		0x14, 0x00, 0x03, 0x00, 0x01, 0x02, 0x03, 0x04,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x04, 0x00,
		0x35, 0x00, 0x00, 0x00, 0x07, 0x00, 0x06, 0x00,
		0x72, 0x72, 0x00, 0x00, 0x0c, 0x00, 0x07, 0x00,
		0x02, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff,
		0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x08, 0x00, 0x09, 0x00, 0xff, 0xff, 0xff, 0xff,
		0x5c, 0x00, 0x0a, 0x00, 0x08, 0x00, 0x01, 0x00,
		0x03, 0x00, 0x00, 0x00, 0x08, 0x00, 0x02, 0x00,
		0x0c, 0x00, 0x00, 0x00, 0x08, 0x00, 0x03, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x04, 0x00,
		0x1a, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x0c, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x06, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x07, 0x00,
		0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x08, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x09, 0x00,
		0x54, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0a, 0x00,
		0x00, 0x00, 0x00, 0x00,
	}

	nlmIPVSIPv4 = []byte{
		0x28, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x01, 0x01, 0x00, 0x00, 0x14, 0x00, 0x01, 0x00,
		0x01, 0x02, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	}
	nlmIPVSIPv6 = []byte{
		0x28, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x01, 0x01, 0x00, 0x00, 0x14, 0x00, 0x01, 0x00,
		0x20, 0x15, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xca, 0xfe,
	}

	ipvsIPTests = []struct {
		ip         net.IP
		nlm        []byte
		bytesEqual bool
	}{
		{net.IPv4(1, 2, 3, 4), nlmIPVSIPv4, true},
		{net.IP{1, 2, 3, 4}, nlmIPVSIPv4, false},
		{net.ParseIP("2015::cafe"), nlmIPVSIPv6, true},
	}
)

func TestMessageBytes(t *testing.T) {
	m, err := NewMessageFromBytes(nlmIPVSInfo)
	if err != nil {
		t.Fatalf("Failed to make netlink message: %v", err)
	}
	defer m.Free()

	got, err := m.Bytes()
	if err != nil {
		t.Fatalf("Failed to get netlink message bytes: %v", err)
	}
	if want := nlmIPVSInfo; !bytes.Equal(got, want) {
		t.Errorf("Got netlink bytes %#v, want %#v", got, want)
	}
}

func TestMessageMarshal(t *testing.T) {
	m, err := NewMessage(1, 25, 0)
	if err != nil {
		t.Fatalf("Failed to make netlink message: %v", err)
	}
	defer m.Free()

	service := &ipvsService{
		AddrFamily:   syscall.AF_INET,
		Protocol:     syscall.IPPROTO_TCP,
		Address:      net.IPv4(1, 1, 1, 1),
		Port:         0x5000,
		FirewallMark: 0x0,
		Scheduler:    "wlc",
		Flags: [8]uint8{
			0x0, 0x0, 0x0, 0x0, 0xff, 0xff, 0xff, 0xff,
		},
		Timeout:           0x0,
		Netmask:           0xffffffff,
		Stats:             nil,
		PersistenceEngine: "",
	}
	ic := &ipvsCommand{Service: service}
	if err := m.Marshal(ic); err != nil {
		t.Fatalf("Failed to marshal: %v", err)
	}

	got, err := m.Bytes()
	if err != nil {
		t.Fatalf("Failed to get message bytes: %v", err)
	}
	if want := nlmIPVSAddService; !bytes.Equal(got, want) {
		t.Fatalf("Got netlink bytes %#v, want %#v", got, want)
	}
}

func TestMessageMarshalIP(t *testing.T) {
	for _, test := range ipvsIPTests {
		func() {
			m, err := NewMessage(1, 25, 0)
			if err != nil {
				t.Fatalf("Failed to make netlink message: %v", err)
			}
			defer m.Free()
			if err := m.Marshal(&ipvsIPAddr{test.ip}); err != nil {
				t.Errorf("Failed to marshal: %v", err)
				return
			}
			got, err := m.Bytes()
			if err != nil {
				t.Errorf("Failed to get message bytes: %v", err)
				return
			}
			if want := test.nlm; !bytes.Equal(got, want) {
				t.Errorf("Got netlink bytes %#v, want %#v", got, want)
			}
		}()
	}
}

func TestMessageMarshalUnexported(t *testing.T) {
	m, err := NewMessage(1, 25, 0)
	if err != nil {
		t.Fatalf("Failed to make netlink message: %v", err)
	}
	defer m.Free()

	nes := &struct {
		internal uint32 `netlink:"attr:x"`
	}{}
	if err := m.Marshal(nes); err == nil {
		t.Fatal("Marshal message succeeded with unexported field")
	}
}

func TestMesageMarshalNonStruct(t *testing.T) {
	var info *ipvsInfo
	var u int32
	var s string

	tests := []struct {
		v interface{}
	}{
		{nil},
		{info},
		{u},
		{&u},
		{s},
		{&s},
	}

	m, err := NewMessage(1, 25, 0)
	if err != nil {
		t.Fatalf("Failed to make netlink message: %v", err)
	}
	defer m.Free()

	for _, test := range tests {
		if err := m.Marshal(test.v); err == nil {
			t.Errorf("Marshal message succeeded with %v", reflect.TypeOf(test.v))
		}
	}
}

func TestMessageUnmarshal(t *testing.T) {
	m, err := NewMessageFromBytes(nlmIPVSInfo)
	if err != nil {
		t.Fatalf("Failed to make netlink message: %v", err)
	}
	defer m.Free()

	got := &ipvsInfo{}
	if err := m.Unmarshal(got); err != nil {
		t.Fatalf("Failed to unmarshal message: %v", err)
	}

	want := &ipvsInfo{Version: 0x10201, ConnTableSize: 0x1000}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Got IPVS info %#v, want %#v", got, want)
	}
}

func TestMessageUnmarshalIP(t *testing.T) {
	for _, test := range ipvsIPTests {
		func() {
			m, err := NewMessageFromBytes(test.nlm)
			if err != nil {
				t.Fatalf("Failed to make netlink message: %v", err)
			}
			defer m.Free()
			ipa := &ipvsIPAddr{}
			if err := m.Unmarshal(ipa); err != nil {
				t.Errorf("Failed to unmarshal: %v", err)
				return
			}
			if test.bytesEqual && !bytes.Equal(test.ip, ipa.IP) {
				t.Errorf("IP address bytes differ - got %#v, want %#v", ipa.IP, test.ip)
				return
			}
			if !test.ip.Equal(ipa.IP) {
				t.Errorf("IP addresses differ - got %#v, want %#v", ipa.IP, test.ip)
			}
		}()
	}
}

func TestMessageUnmarshalUnexported(t *testing.T) {
	m, err := NewMessageFromBytes(nlmIPVSInfo)
	if err != nil {
		t.Fatalf("Failed to make netlink message: %v", err)
	}
	defer m.Free()

	nes := &struct {
		internal uint32 `netlink:"attr:1"`
	}{}
	if err := m.Unmarshal(nes); err == nil {
		t.Error("Unmarshal message succeeded with unexported field")
	}
}

func TestMesageUnmarshalNonStructPtr(t *testing.T) {
	var info *ipvsInfo
	var u int32
	var s string

	tests := []struct {
		v interface{}
	}{
		{nil},
		{ipvsInfo{}},
		{info},
		{u},
		{&u},
		{s},
		{&s},
	}

	m, err := NewMessageFromBytes(nlmIPVSInfo)
	if err != nil {
		t.Fatalf("Failed to make netlink message: %v", err)
	}
	defer m.Free()

	for _, test := range tests {
		if err := m.Unmarshal(test.v); err == nil {
			t.Errorf("Unmarshal message succeeded with %v", reflect.TypeOf(test.v))
		}
	}
}

func TestMessageUnmarshalNested(t *testing.T) {
	m, err := NewMessageFromBytes(nlmIPVSService)
	if err != nil {
		t.Fatalf("Failed to make netlink message: %v", err)
	}
	defer m.Free()

	got := &ipvsCommand{}
	if err := m.Unmarshal(got); err != nil {
		t.Errorf("Failed to unmarshal message: %v", err)
	}

	want := &ipvsService{
		AddrFamily:   0x2,
		Protocol:     0x11,
		Address:      net.IPv4(1, 2, 3, 4),
		Port:         0x3500,
		FirewallMark: 0x0,
		Scheduler:    "rr",
		Flags: [8]uint8{
			0x02, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff,
		},
		Timeout: 0x0,
		Netmask: 0xffffffff,
		Stats: &ipvsServiceStats{
			ipvsStats: ipvsStats{
				Conns:      0x3,
				PacketsIn:  0xc,
				PacketsOut: 0x0,
				BytesIn:    0x41a,
				BytesOut:   0x0,
				CPS:        0x0,
				PPSIn:      0x1,
				PPSOut:     0x0,
				BPSIn:      0x54,
				BPSOut:     0x0,
			},
		},
		PersistenceEngine: "",
	}
	if !reflect.DeepEqual(got.Service, want) {
		t.Errorf("Got IPVS service %#v, want %#v", got.Service, want)
	}
}
