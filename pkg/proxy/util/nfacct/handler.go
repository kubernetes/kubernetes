//go:build linux

/*
Copyright 2024 The Kubernetes Authors.

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

package nfacct

import (
	"github.com/vishvananda/netlink/nl"
	"github.com/vishvananda/netns"
	"golang.org/x/sys/unix"
)

// handler is an injectable interface for creating netlink request.
type handler interface {
	newRequest(cmd int, flags uint16) request
}

// request is an injectable interface representing a netlink request.
type request interface {
	Serialize() []byte
	AddData(data nl.NetlinkRequestData)
	AddRawData(data []byte)
	Execute(sockType int, resType uint16) ([][]byte, error)
}

// netlinkHandler is an implementation of the handler interface. It maintains a netlink socket
// for communication with the NFAcct subsystem.
type netlinkHandler struct {
	socket *nl.NetlinkSocket
}

// newNetlinkHandler initializes a netlink socket in the current network namespace and returns
// an instance of netlinkHandler with the initialized socket.
func newNetlinkHandler() (handler, error) {
	socket, err := nl.GetNetlinkSocketAt(netns.None(), netns.None(), unix.NETLINK_NETFILTER)
	if err != nil {
		return nil, err
	}
	return &netlinkHandler{socket: socket}, nil
}

// newRequest creates a netlink request tailored for the NFAcct subsystem encapsulating the
// specified cmd and flags. It incorporates the netlink header and netfilter generic header
// into the resulting request.
func (n *netlinkHandler) newRequest(cmd int, flags uint16) request {
	req := &nl.NetlinkRequest{
		// netlink message header
		// (definition: https://github.com/torvalds/linux/blob/v6.7/include/uapi/linux/netlink.h#L44-L58)
		NlMsghdr: unix.NlMsghdr{
			Len:   uint32(unix.SizeofNlMsghdr),
			Type:  uint16(cmd | (unix.NFNL_SUBSYS_ACCT << 8)),
			Flags: flags,
		},
		Sockets: map[int]*nl.SocketHandle{
			unix.NETLINK_NETFILTER: {Socket: n.socket},
		},
		Data: []nl.NetlinkRequestData{
			// netfilter generic message
			// (definition: https://github.com/torvalds/linux/blob/v6.7/include/uapi/linux/netfilter/nfnetlink.h#L32-L38)
			&nl.Nfgenmsg{
				NfgenFamily: uint8(unix.AF_NETLINK),
				Version:     nl.NFNETLINK_V0,
				ResId:       0,
			},
		},
	}
	return req
}
