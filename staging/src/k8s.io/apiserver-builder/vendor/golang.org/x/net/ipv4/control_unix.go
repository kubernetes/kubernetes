// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd

package ipv4

import (
	"os"
	"syscall"
	"unsafe"

	"golang.org/x/net/internal/iana"
)

func setControlMessage(fd int, opt *rawOpt, cf ControlFlags, on bool) error {
	opt.Lock()
	defer opt.Unlock()
	if cf&FlagTTL != 0 && sockOpts[ssoReceiveTTL].name > 0 {
		if err := setInt(fd, &sockOpts[ssoReceiveTTL], boolint(on)); err != nil {
			return err
		}
		if on {
			opt.set(FlagTTL)
		} else {
			opt.clear(FlagTTL)
		}
	}
	if sockOpts[ssoPacketInfo].name > 0 {
		if cf&(FlagSrc|FlagDst|FlagInterface) != 0 {
			if err := setInt(fd, &sockOpts[ssoPacketInfo], boolint(on)); err != nil {
				return err
			}
			if on {
				opt.set(cf & (FlagSrc | FlagDst | FlagInterface))
			} else {
				opt.clear(cf & (FlagSrc | FlagDst | FlagInterface))
			}
		}
	} else {
		if cf&FlagDst != 0 && sockOpts[ssoReceiveDst].name > 0 {
			if err := setInt(fd, &sockOpts[ssoReceiveDst], boolint(on)); err != nil {
				return err
			}
			if on {
				opt.set(FlagDst)
			} else {
				opt.clear(FlagDst)
			}
		}
		if cf&FlagInterface != 0 && sockOpts[ssoReceiveInterface].name > 0 {
			if err := setInt(fd, &sockOpts[ssoReceiveInterface], boolint(on)); err != nil {
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

func newControlMessage(opt *rawOpt) (oob []byte) {
	opt.RLock()
	var l int
	if opt.isset(FlagTTL) && ctlOpts[ctlTTL].name > 0 {
		l += syscall.CmsgSpace(ctlOpts[ctlTTL].length)
	}
	if ctlOpts[ctlPacketInfo].name > 0 {
		if opt.isset(FlagSrc | FlagDst | FlagInterface) {
			l += syscall.CmsgSpace(ctlOpts[ctlPacketInfo].length)
		}
	} else {
		if opt.isset(FlagDst) && ctlOpts[ctlDst].name > 0 {
			l += syscall.CmsgSpace(ctlOpts[ctlDst].length)
		}
		if opt.isset(FlagInterface) && ctlOpts[ctlInterface].name > 0 {
			l += syscall.CmsgSpace(ctlOpts[ctlInterface].length)
		}
	}
	if l > 0 {
		oob = make([]byte, l)
		b := oob
		if opt.isset(FlagTTL) && ctlOpts[ctlTTL].name > 0 {
			b = ctlOpts[ctlTTL].marshal(b, nil)
		}
		if ctlOpts[ctlPacketInfo].name > 0 {
			if opt.isset(FlagSrc | FlagDst | FlagInterface) {
				b = ctlOpts[ctlPacketInfo].marshal(b, nil)
			}
		} else {
			if opt.isset(FlagDst) && ctlOpts[ctlDst].name > 0 {
				b = ctlOpts[ctlDst].marshal(b, nil)
			}
			if opt.isset(FlagInterface) && ctlOpts[ctlInterface].name > 0 {
				b = ctlOpts[ctlInterface].marshal(b, nil)
			}
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
		if m.Header.Level != iana.ProtocolIP {
			continue
		}
		switch int(m.Header.Type) {
		case ctlOpts[ctlTTL].name:
			ctlOpts[ctlTTL].parse(cm, m.Data[:])
		case ctlOpts[ctlDst].name:
			ctlOpts[ctlDst].parse(cm, m.Data[:])
		case ctlOpts[ctlInterface].name:
			ctlOpts[ctlInterface].parse(cm, m.Data[:])
		case ctlOpts[ctlPacketInfo].name:
			ctlOpts[ctlPacketInfo].parse(cm, m.Data[:])
		}
	}
	return cm, nil
}

func marshalControlMessage(cm *ControlMessage) (oob []byte) {
	if cm == nil {
		return nil
	}
	var l int
	pktinfo := false
	if ctlOpts[ctlPacketInfo].name > 0 && (cm.Src.To4() != nil || cm.IfIndex > 0) {
		pktinfo = true
		l += syscall.CmsgSpace(ctlOpts[ctlPacketInfo].length)
	}
	if l > 0 {
		oob = make([]byte, l)
		b := oob
		if pktinfo {
			b = ctlOpts[ctlPacketInfo].marshal(b, cm)
		}
	}
	return
}

func marshalTTL(b []byte, cm *ControlMessage) []byte {
	m := (*syscall.Cmsghdr)(unsafe.Pointer(&b[0]))
	m.Level = iana.ProtocolIP
	m.Type = sysIP_RECVTTL
	m.SetLen(syscall.CmsgLen(1))
	return b[syscall.CmsgSpace(1):]
}

func parseTTL(cm *ControlMessage, b []byte) {
	cm.TTL = int(*(*byte)(unsafe.Pointer(&b[:1][0])))
}
