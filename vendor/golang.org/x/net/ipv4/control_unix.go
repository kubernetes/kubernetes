// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris
// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris

package ipv4

import (
	"unsafe"

	"golang.org/x/net/internal/iana"
	"golang.org/x/net/internal/socket"
)

func setControlMessage(c *socket.Conn, opt *rawOpt, cf ControlFlags, on bool) error {
	opt.Lock()
	defer opt.Unlock()
	if so, ok := sockOpts[ssoReceiveTTL]; ok && cf&FlagTTL != 0 {
		if err := so.SetInt(c, boolint(on)); err != nil {
			return err
		}
		if on {
			opt.set(FlagTTL)
		} else {
			opt.clear(FlagTTL)
		}
	}
	if so, ok := sockOpts[ssoPacketInfo]; ok {
		if cf&(FlagSrc|FlagDst|FlagInterface) != 0 {
			if err := so.SetInt(c, boolint(on)); err != nil {
				return err
			}
			if on {
				opt.set(cf & (FlagSrc | FlagDst | FlagInterface))
			} else {
				opt.clear(cf & (FlagSrc | FlagDst | FlagInterface))
			}
		}
	} else {
		if so, ok := sockOpts[ssoReceiveDst]; ok && cf&FlagDst != 0 {
			if err := so.SetInt(c, boolint(on)); err != nil {
				return err
			}
			if on {
				opt.set(FlagDst)
			} else {
				opt.clear(FlagDst)
			}
		}
		if so, ok := sockOpts[ssoReceiveInterface]; ok && cf&FlagInterface != 0 {
			if err := so.SetInt(c, boolint(on)); err != nil {
				return err
			}
			if on {
				opt.set(FlagInterface)
			} else {
				opt.clear(FlagInterface)
			}
		}
	}
	return nil
}

func marshalTTL(b []byte, cm *ControlMessage) []byte {
	m := socket.ControlMessage(b)
	m.MarshalHeader(iana.ProtocolIP, sysIP_RECVTTL, 1)
	return m.Next(1)
}

func parseTTL(cm *ControlMessage, b []byte) {
	cm.TTL = int(*(*byte)(unsafe.Pointer(&b[:1][0])))
}
