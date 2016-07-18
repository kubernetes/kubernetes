// Copyright 2015 flannel authors
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

package udp

//#include "proxy.h"
import "C"

import (
	"net"
	"os"
	"reflect"
	"unsafe"

	log "github.com/golang/glog"

	"github.com/coreos/flannel/pkg/ip"
)

func runCProxy(tun *os.File, conn *net.UDPConn, ctl *os.File, tunIP ip.IP4, tunMTU int) {
	var log_errors int
	if log.V(1) {
		log_errors = 1
	}

	c, err := conn.File()
	if err != nil {
		log.Error("Converting UDPConn to File failed: ", err)
		return
	}
	defer c.Close()

	C.run_proxy(
		C.int(tun.Fd()),
		C.int(c.Fd()),
		C.int(ctl.Fd()),
		C.in_addr_t(tunIP.NetworkOrder()),
		C.size_t(tunMTU),
		C.int(log_errors),
	)
}

func writeCommand(f *os.File, cmd *C.command) {
	hdr := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(cmd)),
		Len:  int(unsafe.Sizeof(*cmd)),
		Cap:  int(unsafe.Sizeof(*cmd)),
	}
	buf := *(*[]byte)(unsafe.Pointer(&hdr))

	f.Write(buf)
}

func setRoute(ctl *os.File, dst ip.IP4Net, nextHopIP ip.IP4, nextHopPort int) {
	cmd := C.command{
		cmd:           C.CMD_SET_ROUTE,
		dest_net:      C.in_addr_t(dst.IP.NetworkOrder()),
		dest_net_len:  C.int(dst.PrefixLen),
		next_hop_ip:   C.in_addr_t(nextHopIP.NetworkOrder()),
		next_hop_port: C.short(nextHopPort),
	}

	writeCommand(ctl, &cmd)
}

func removeRoute(ctl *os.File, dst ip.IP4Net) {
	cmd := C.command{
		cmd:          C.CMD_DEL_ROUTE,
		dest_net:     C.in_addr_t(dst.IP.NetworkOrder()),
		dest_net_len: C.int(dst.PrefixLen),
	}

	writeCommand(ctl, &cmd)
}

func stopProxy(ctl *os.File) {
	cmd := C.command{
		cmd: C.CMD_STOP,
	}

	writeCommand(ctl, &cmd)
}
