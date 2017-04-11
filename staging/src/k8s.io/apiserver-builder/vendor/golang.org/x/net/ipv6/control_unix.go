// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd

package ipv6

import (
	"os"
	"syscall"

	"golang.org/x/net/internal/iana"
)

func setControlMessage(fd int, opt *rawOpt, cf ControlFlags, on bool) error {
	opt.Lock()
	defer opt.Unlock()
	if cf&FlagTrafficClass != 0 && sockOpts[ssoReceiveTrafficClass].name > 0 {
		if err := setInt(fd, &sockOpts[ssoReceiveTrafficClass], boolint(on)); err != nil {
			return err
		}
		if on {
			opt.set(FlagTrafficClass)
		} else {
			opt.clear(FlagTrafficClass)
		}
	}
	if cf&FlagHopLimit != 0 && sockOpts[ssoReceiveHopLimit].name > 0 {
		if err := setInt(fd, &sockOpts[ssoReceiveHopLimit], boolint(on)); err != nil {
			return err
		}
		if on {
			opt.set(FlagHopLimit)
		} else {
			opt.clear(FlagHopLimit)
		}
	}
	if cf&flagPacketInfo != 0 && sockOpts[ssoReceivePacketInfo].name > 0 {
		if err := setInt(fd, &sockOpts[ssoReceivePacketInfo], boolint(on)); err != nil {
			return err
		}
		if on {
			opt.set(cf & flagPacketInfo)
		} else {
			opt.clear(cf & flagPacketInfo)
		}
	}
	if cf&FlagPathMTU != 0 && sockOpts[ssoReceivePathMTU].name > 0 {
		if err := setInt(fd, &sockOpts[ssoReceivePathMTU], boolint(on)); err != nil {
			return err
		}
		if on {
			opt.set(FlagPathMTU)
		} else {
			opt.clear(FlagPathMTU)
		}
	}
	return nil
}

func newControlMessage(opt *rawOpt) (oob []byte) {
	opt.RLock()
	var l int
	if opt.isset(FlagTrafficClass) && ctlOpts[ctlTrafficClass].name > 0 {
		l += syscall.CmsgSpace(ctlOpts[ctlTrafficClass].length)
	}
	if opt.isset(FlagHopLimit) && ctlOpts[ctlHopLimit].name > 0 {
		l += syscall.CmsgSpace(ctlOpts[ctlHopLimit].length)
	}
	if opt.isset(flagPacketInfo) && ctlOpts[ctlPacketInfo].name > 0 {
		l += syscall.CmsgSpace(ctlOpts[ctlPacketInfo].length)
	}
	if opt.isset(FlagPathMTU) && ctlOpts[ctlPathMTU].name > 0 {
		l += syscall.CmsgSpace(ctlOpts[ctlPathMTU].length)
	}
	if l > 0 {
		oob = make([]byte, l)
		b := oob
		if opt.isset(FlagTrafficClass) && ctlOpts[ctlTrafficClass].name > 0 {
			b = ctlOpts[ctlTrafficClass].marshal(b, nil)
		}
		if opt.isset(FlagHopLimit) && ctlOpts[ctlHopLimit].name > 0 {
			b = ctlOpts[ctlHopLimit].marshal(b, nil)
		}
		if opt.isset(flagPacketInfo) && ctlOpts[ctlPacketInfo].name > 0 {
			b = ctlOpts[ctlPacketInfo].marshal(b, nil)
		}
		if opt.isset(FlagPathMTU) && ctlOpts[ctlPathMTU].name > 0 {
			b = ctlOpts[ctlPathMTU].marshal(b, nil)
		}
	}
	opt.RUnlock()
	return
}

func parseControlMessage(b []byte) (*ControlMessage, error) {
	if len(b) == 0 {
		return nil, nil
	}
	cmsgs, err := syscall.ParseSocketControlMessage(b)
	if err != nil {
		return nil, os.NewSyscallError("parse socket control message", err)
	}
	cm := &ControlMessage{}
	for _, m := range cmsgs {
		if m.Header.Level != iana.ProtocolIPv6 {
			continue
		}
		switch int(m.Header.Type) {
		case ctlOpts[ctlTrafficClass].name:
			ctlOpts[ctlTrafficClass].parse(cm, m.Data[:])
		case ctlOpts[ctlHopLimit].name:
			ctlOpts[ctlHopLimit].parse(cm, m.Data[:])
		case ctlOpts[ctlPacketInfo].name:
			ctlOpts[ctlPacketInfo].parse(cm, m.Data[:])
		case ctlOpts[ctlPathMTU].name:
			ctlOpts[ctlPathMTU].parse(cm, m.Data[:])
		}
	}
	return cm, nil
}

func marshalControlMessage(cm *ControlMessage) (oob []byte) {
	if cm == nil {
		return
	}
	var l int
	tclass := false
	if ctlOpts[ctlTrafficClass].name > 0 && cm.TrafficClass > 0 {
		tclass = true
		l += syscall.CmsgSpace(ctlOpts[ctlTrafficClass].length)
	}
	hoplimit := false
	if ctlOpts[ctlHopLimit].name > 0 && cm.HopLimit > 0 {
		hoplimit = true
		l += syscall.CmsgSpace(ctlOpts[ctlHopLimit].length)
	}
	pktinfo := false
	if ctlOpts[ctlPacketInfo].name > 0 && (cm.Src.To16() != nil && cm.Src.To4() == nil || cm.IfIndex > 0) {
		pktinfo = true
		l += syscall.CmsgSpace(ctlOpts[ctlPacketInfo].length)
	}
	nexthop := false
	if ctlOpts[ctlNextHop].name > 0 && cm.NextHop.To16() != nil && cm.NextHop.To4() == nil {
		nexthop = true
		l += syscall.CmsgSpace(ctlOpts[ctlNextHop].length)
	}
	if l > 0 {
		oob = make([]byte, l)
		b := oob
		if tclass {
			b = ctlOpts[ctlTrafficClass].marshal(b, cm)
		}
		if hoplimit {
			b = ctlOpts[ctlHopLimit].marshal(b, cm)
		}
		if pktinfo {
			b = ctlOpts[ctlPacketInfo].marshal(b, cm)
		}
		if nexthop {
			b = ctlOpts[ctlNextHop].marshal(b, cm)
		}
	}
	return
}
