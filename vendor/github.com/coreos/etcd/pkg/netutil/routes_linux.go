// Copyright 2016 The etcd Authors
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

// +build linux
// +build 386 amd64

// TODO support native endian but without using "unsafe"

package netutil

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"net"
	"syscall"
)

var errNoDefaultRoute = fmt.Errorf("could not find default route")

func GetDefaultHost() (string, error) {
	rmsg, rerr := getDefaultRoute()
	if rerr != nil {
		return "", rerr
	}

	host, oif, err := parsePREFSRC(rmsg)
	if err != nil {
		return "", err
	}
	if host != "" {
		return host, nil
	}

	// prefsrc not detected, fall back to getting address from iface
	ifmsg, ierr := getIface(oif)
	if ierr != nil {
		return "", ierr
	}

	attrs, aerr := syscall.ParseNetlinkRouteAttr(ifmsg)
	if aerr != nil {
		return "", aerr
	}

	for _, attr := range attrs {
		if attr.Attr.Type == syscall.RTA_SRC {
			return net.IP(attr.Value).String(), nil
		}
	}

	return "", errNoDefaultRoute
}

func getDefaultRoute() (*syscall.NetlinkMessage, error) {
	dat, err := syscall.NetlinkRIB(syscall.RTM_GETROUTE, syscall.AF_UNSPEC)
	if err != nil {
		return nil, err
	}

	msgs, msgErr := syscall.ParseNetlinkMessage(dat)
	if msgErr != nil {
		return nil, msgErr
	}

	rtmsg := syscall.RtMsg{}
	for _, m := range msgs {
		if m.Header.Type != syscall.RTM_NEWROUTE {
			continue
		}
		buf := bytes.NewBuffer(m.Data[:syscall.SizeofRtMsg])
		if rerr := binary.Read(buf, binary.LittleEndian, &rtmsg); rerr != nil {
			continue
		}
		if rtmsg.Dst_len == 0 {
			// zero-length Dst_len implies default route
			return &m, nil
		}
	}

	return nil, errNoDefaultRoute
}

func getIface(idx uint32) (*syscall.NetlinkMessage, error) {
	dat, err := syscall.NetlinkRIB(syscall.RTM_GETADDR, syscall.AF_UNSPEC)
	if err != nil {
		return nil, err
	}

	msgs, msgErr := syscall.ParseNetlinkMessage(dat)
	if msgErr != nil {
		return nil, msgErr
	}

	ifaddrmsg := syscall.IfAddrmsg{}
	for _, m := range msgs {
		if m.Header.Type != syscall.RTM_NEWADDR {
			continue
		}
		buf := bytes.NewBuffer(m.Data[:syscall.SizeofIfAddrmsg])
		if rerr := binary.Read(buf, binary.LittleEndian, &ifaddrmsg); rerr != nil {
			continue
		}
		if ifaddrmsg.Index == idx {
			return &m, nil
		}
	}

	return nil, errNoDefaultRoute
}

var errNoDefaultInterface = fmt.Errorf("could not find default interface")

func GetDefaultInterface() (string, error) {
	rmsg, rerr := getDefaultRoute()
	if rerr != nil {
		return "", rerr
	}

	_, oif, err := parsePREFSRC(rmsg)
	if err != nil {
		return "", err
	}

	ifmsg, ierr := getIface(oif)
	if ierr != nil {
		return "", ierr
	}

	attrs, aerr := syscall.ParseNetlinkRouteAttr(ifmsg)
	if aerr != nil {
		return "", aerr
	}

	for _, attr := range attrs {
		if attr.Attr.Type == syscall.IFLA_IFNAME {
			return string(attr.Value[:len(attr.Value)-1]), nil
		}
	}
	return "", errNoDefaultInterface
}

// parsePREFSRC returns preferred source address and output interface index (RTA_OIF).
func parsePREFSRC(m *syscall.NetlinkMessage) (host string, oif uint32, err error) {
	var attrs []syscall.NetlinkRouteAttr
	attrs, err = syscall.ParseNetlinkRouteAttr(m)
	if err != nil {
		return "", 0, err
	}

	for _, attr := range attrs {
		if attr.Attr.Type == syscall.RTA_PREFSRC {
			host = net.IP(attr.Value).String()
		}
		if attr.Attr.Type == syscall.RTA_OIF {
			oif = binary.LittleEndian.Uint32(attr.Value)
		}
		if host != "" && oif != uint32(0) {
			break
		}
	}

	if oif == 0 {
		err = errNoDefaultRoute
	}
	return
}
