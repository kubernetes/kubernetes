// +build linux

package nlgo

import (
	"fmt"
	"syscall"
	"unsafe"
)

type NlMsgerr syscall.NetlinkMessage

func (self NlMsgerr) Payload() syscall.NlMsgerr {
	msg := syscall.NetlinkMessage(self)
	return *(*syscall.NlMsgerr)(unsafe.Pointer(&msg.Data[0]))
}

func (self NlMsgerr) Error() string {
	p := self.Payload()
	return fmt.Sprintf("NlMsgerr %v", syscall.Errno(uintptr(-p.Error)))
}

type IfInfoMessage syscall.NetlinkMessage

func (self IfInfoMessage) IfInfo() syscall.IfInfomsg {
	msg := syscall.NetlinkMessage(self)
	return *(*syscall.IfInfomsg)(unsafe.Pointer(&msg.Data[0]))
}

func (self IfInfoMessage) Attrs() (NlaValue, error) {
	msg := syscall.NetlinkMessage(self)
	return RouteLinkPolicy.Parse(msg.Data[NLMSG_ALIGN(syscall.SizeofIfInfomsg):])
}

func (self *IfInfoMessage) Set(ifinfo syscall.IfInfomsg, attrs AttrList) {
	var data, ext []byte
	if attrs != nil {
		ext = attrs.Bytes()
	}
	if len(ext) > 0 {
		data = make([]byte, NLMSG_ALIGN(syscall.SizeofIfInfomsg)+len(ext))
	} else {
		data = make([]byte, syscall.SizeofIfInfomsg)
	}
	copy(data, (*[syscall.SizeofIfInfomsg]byte)(unsafe.Pointer(&ifinfo))[:])
	if len(ext) > 0 {
		copy(data[NLMSG_ALIGN(syscall.SizeofIfInfomsg):], ext)
	}
	(*syscall.NetlinkMessage)(self).Data = data
}

type RtMessage syscall.NetlinkMessage

func (self RtMessage) Rt() syscall.RtMsg {
	msg := syscall.NetlinkMessage(self)
	return *(*syscall.RtMsg)(unsafe.Pointer(&msg.Data[0]))
}

func (self RtMessage) Attrs() (NlaValue, error) {
	msg := syscall.NetlinkMessage(self)
	return RoutePolicy.Parse(msg.Data[NLMSG_ALIGN(syscall.SizeofRtMsg):])
}

func (self *RtMessage) Set(rt syscall.RtMsg, attrs AttrList) {
	var data []byte
	ext := attrs.Bytes()
	if len(ext) > 0 {
		data = make([]byte, NLMSG_ALIGN(syscall.SizeofRtMsg)+len(ext))
	} else {
		data = make([]byte, syscall.SizeofRtMsg)
	}
	copy(data, (*[syscall.SizeofRtMsg]byte)(unsafe.Pointer(&rt))[:])
	if len(ext) > 0 {
		copy(data[NLMSG_ALIGN(syscall.SizeofRtMsg):], ext)
	}
	(*syscall.NetlinkMessage)(self).Data = data
}
