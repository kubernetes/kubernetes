/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package toolbox

import (
	"bytes"
	"fmt"
	"net"

	xdr "github.com/davecgh/go-xdr/xdr2"
)

// Defs from: open-vm-tools/lib/guestRpc/nicinfo.x

type TypedIPAddress struct {
	Type    int32
	Address []byte
}

type IPAddressEntry struct {
	Address      TypedIPAddress
	PrefixLength uint32
	Origin       *int32 `xdr:"optional"`
	Status       *int32 `xdr:"optional"`
}

type InetCidrRouteEntry struct {
	Dest         TypedIPAddress
	PrefixLength uint32
	NextHop      *TypedIPAddress `xdr:"optional"`
	IfIndex      uint32
	Type         int32
	Metric       uint32
}

type DNSConfigInfo struct {
	HostName   *string `xdr:"optional"`
	DomainName *string `xdr:"optional"`
	Servers    []TypedIPAddress
	Search     *string `xdr:"optional"`
}

type WinsConfigInfo struct {
	Primary   TypedIPAddress
	Secondary TypedIPAddress
}

type DhcpConfigInfo struct {
	Enabled  bool
	Settings string
}

type GuestNicV3 struct {
	MacAddress       string
	IPs              []IPAddressEntry
	DNSConfigInfo    *DNSConfigInfo  `xdr:"optional"`
	WinsConfigInfo   *WinsConfigInfo `xdr:"optional"`
	DhcpConfigInfov4 *DhcpConfigInfo `xdr:"optional"`
	DhcpConfigInfov6 *DhcpConfigInfo `xdr:"optional"`
}

type NicInfoV3 struct {
	Nics             []GuestNicV3
	Routes           []InetCidrRouteEntry
	DNSConfigInfo    *DNSConfigInfo  `xdr:"optional"`
	WinsConfigInfo   *WinsConfigInfo `xdr:"optional"`
	DhcpConfigInfov4 *DhcpConfigInfo `xdr:"optional"`
	DhcpConfigInfov6 *DhcpConfigInfo `xdr:"optional"`
}

type GuestNicInfo struct {
	Version int32
	V3      *NicInfoV3 `xdr:"optional"`
}

func EncodeXDR(val interface{}) ([]byte, error) {
	var buf bytes.Buffer

	_, err := xdr.Marshal(&buf, val)
	if err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

func DecodeXDR(buf []byte, val interface{}) error {
	r := bytes.NewReader(buf)
	_, err := xdr.Unmarshal(r, val)
	return err
}

func NewGuestNicInfo() *GuestNicInfo {
	return &GuestNicInfo{
		Version: 3,
		V3:      &NicInfoV3{},
	}
}

func (nic *GuestNicV3) AddIP(addr net.Addr) {
	ip, ok := addr.(*net.IPNet)
	if !ok {
		return
	}

	kind := int32(1) // IAT_IPV4
	if ip.IP.To4() == nil {
		kind = 2 // IAT_IPV6
	} else {
		ip.IP = ip.IP.To4() // convert to 4-byte representation
	}

	size, _ := ip.Mask.Size()

	// nicinfo.x defines enum IpAddressStatus, but vmtoolsd only uses IAS_PREFERRED
	var status int32 = 1 // IAS_PREFERRED

	e := IPAddressEntry{
		Address: TypedIPAddress{
			Type:    kind,
			Address: []byte(ip.IP),
		},
		PrefixLength: uint32(size),
		Status:       &status,
	}

	nic.IPs = append(nic.IPs, e)
}

func GuestInfoCommand(kind int, req []byte) []byte {
	request := fmt.Sprintf("SetGuestInfo  %d ", kind)
	return append([]byte(request), req...)
}

var (
	netInterfaces = net.Interfaces
	maxNics       = 16 // guestRpc/nicinfo.x:NICINFO_MAX_NICS
)

//
func DefaultGuestNicInfo() *GuestNicInfo {
	proto := NewGuestNicInfo()
	info := proto.V3
	// #nosec: Errors unhandled
	ifs, _ := netInterfaces()

	for _, i := range ifs {
		if i.Flags&net.FlagLoopback == net.FlagLoopback {
			continue
		}

		if len(i.HardwareAddr) == 0 {
			continue // Not useful from outside the guest without a MAC
		}

		// #nosec: Errors unhandled
		addrs, _ := i.Addrs()

		if len(addrs) == 0 {
			continue // Not useful from outside the guest without an IP
		}

		nic := GuestNicV3{
			MacAddress: i.HardwareAddr.String(),
		}

		for _, addr := range addrs {
			nic.AddIP(addr)
		}

		info.Nics = append(info.Nics, nic)

		if len(info.Nics) >= maxNics {
			break
		}
	}

	return proto
}

func GuestInfoNicInfoRequest() ([]byte, error) {
	r, err := EncodeXDR(DefaultGuestNicInfo())
	if err != nil {
		return nil, err
	}

	return GuestInfoCommand(9 /*INFO_IPADDRESS_V3*/, r), nil
}
