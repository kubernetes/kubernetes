// +build linux

package ipvs

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"net"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"unsafe"

	"github.com/sirupsen/logrus"
	"github.com/vishvananda/netlink/nl"
	"github.com/vishvananda/netns"
)

// For Quick Reference IPVS related netlink message is described at the end of this file.
var (
	native     = nl.NativeEndian()
	ipvsFamily int
	ipvsOnce   sync.Once
)

type genlMsgHdr struct {
	cmd      uint8
	version  uint8
	reserved uint16
}

type ipvsFlags struct {
	flags uint32
	mask  uint32
}

func deserializeGenlMsg(b []byte) (hdr *genlMsgHdr) {
	return (*genlMsgHdr)(unsafe.Pointer(&b[0:unsafe.Sizeof(*hdr)][0]))
}

func (hdr *genlMsgHdr) Serialize() []byte {
	return (*(*[unsafe.Sizeof(*hdr)]byte)(unsafe.Pointer(hdr)))[:]
}

func (hdr *genlMsgHdr) Len() int {
	return int(unsafe.Sizeof(*hdr))
}

func (f *ipvsFlags) Serialize() []byte {
	return (*(*[unsafe.Sizeof(*f)]byte)(unsafe.Pointer(f)))[:]
}

func (f *ipvsFlags) Len() int {
	return int(unsafe.Sizeof(*f))
}

func setup() {
	ipvsOnce.Do(func() {
		var err error
		if out, err := exec.Command("modprobe", "-va", "ip_vs").CombinedOutput(); err != nil {
			logrus.Warnf("Running modprobe ip_vs failed with message: `%s`, error: %v", strings.TrimSpace(string(out)), err)
		}

		ipvsFamily, err = getIPVSFamily()
		if err != nil {
			logrus.Error("Could not get ipvs family information from the kernel. It is possible that ipvs is not enabled in your kernel. Native loadbalancing will not work until this is fixed.")
		}
	})
}

func fillService(s *Service) nl.NetlinkRequestData {
	cmdAttr := nl.NewRtAttr(ipvsCmdAttrService, nil)
	nl.NewRtAttrChild(cmdAttr, ipvsSvcAttrAddressFamily, nl.Uint16Attr(s.AddressFamily))
	if s.FWMark != 0 {
		nl.NewRtAttrChild(cmdAttr, ipvsSvcAttrFWMark, nl.Uint32Attr(s.FWMark))
	} else {
		nl.NewRtAttrChild(cmdAttr, ipvsSvcAttrProtocol, nl.Uint16Attr(s.Protocol))
		nl.NewRtAttrChild(cmdAttr, ipvsSvcAttrAddress, rawIPData(s.Address))

		// Port needs to be in network byte order.
		portBuf := new(bytes.Buffer)
		binary.Write(portBuf, binary.BigEndian, s.Port)
		nl.NewRtAttrChild(cmdAttr, ipvsSvcAttrPort, portBuf.Bytes())
	}

	nl.NewRtAttrChild(cmdAttr, ipvsSvcAttrSchedName, nl.ZeroTerminated(s.SchedName))
	if s.PEName != "" {
		nl.NewRtAttrChild(cmdAttr, ipvsSvcAttrPEName, nl.ZeroTerminated(s.PEName))
	}
	f := &ipvsFlags{
		flags: s.Flags,
		mask:  0xFFFFFFFF,
	}
	nl.NewRtAttrChild(cmdAttr, ipvsSvcAttrFlags, f.Serialize())
	nl.NewRtAttrChild(cmdAttr, ipvsSvcAttrTimeout, nl.Uint32Attr(s.Timeout))
	nl.NewRtAttrChild(cmdAttr, ipvsSvcAttrNetmask, nl.Uint32Attr(s.Netmask))
	return cmdAttr
}

func fillDestinaton(d *Destination) nl.NetlinkRequestData {
	cmdAttr := nl.NewRtAttr(ipvsCmdAttrDest, nil)

	nl.NewRtAttrChild(cmdAttr, ipvsDestAttrAddress, rawIPData(d.Address))
	// Port needs to be in network byte order.
	portBuf := new(bytes.Buffer)
	binary.Write(portBuf, binary.BigEndian, d.Port)
	nl.NewRtAttrChild(cmdAttr, ipvsDestAttrPort, portBuf.Bytes())

	nl.NewRtAttrChild(cmdAttr, ipvsDestAttrForwardingMethod, nl.Uint32Attr(d.ConnectionFlags&ConnectionFlagFwdMask))
	nl.NewRtAttrChild(cmdAttr, ipvsDestAttrWeight, nl.Uint32Attr(uint32(d.Weight)))
	nl.NewRtAttrChild(cmdAttr, ipvsDestAttrUpperThreshold, nl.Uint32Attr(d.UpperThreshold))
	nl.NewRtAttrChild(cmdAttr, ipvsDestAttrLowerThreshold, nl.Uint32Attr(d.LowerThreshold))

	return cmdAttr
}

func (i *Handle) doCmdwithResponse(s *Service, d *Destination, cmd uint8) ([][]byte, error) {
	req := newIPVSRequest(cmd)
	req.Seq = atomic.AddUint32(&i.seq, 1)

	if s == nil {
		req.Flags |= syscall.NLM_F_DUMP                    //Flag to dump all messages
		req.AddData(nl.NewRtAttr(ipvsCmdAttrService, nil)) //Add a dummy attribute
	} else {
		req.AddData(fillService(s))
	}

	if d == nil {
		if cmd == ipvsCmdGetDest {
			req.Flags |= syscall.NLM_F_DUMP
		}

	} else {
		req.AddData(fillDestinaton(d))
	}

	res, err := execute(i.sock, req, 0)
	if err != nil {
		return [][]byte{}, err
	}

	return res, nil
}

func (i *Handle) doCmd(s *Service, d *Destination, cmd uint8) error {
	_, err := i.doCmdwithResponse(s, d, cmd)

	return err
}

func getIPVSFamily() (int, error) {
	sock, err := nl.GetNetlinkSocketAt(netns.None(), netns.None(), syscall.NETLINK_GENERIC)
	if err != nil {
		return 0, err
	}
	defer sock.Close()

	req := newGenlRequest(genlCtrlID, genlCtrlCmdGetFamily)
	req.AddData(nl.NewRtAttr(genlCtrlAttrFamilyName, nl.ZeroTerminated("IPVS")))

	msgs, err := execute(sock, req, 0)
	if err != nil {
		return 0, err
	}

	for _, m := range msgs {
		hdr := deserializeGenlMsg(m)
		attrs, err := nl.ParseRouteAttr(m[hdr.Len():])
		if err != nil {
			return 0, err
		}

		for _, attr := range attrs {
			switch int(attr.Attr.Type) {
			case genlCtrlAttrFamilyID:
				return int(native.Uint16(attr.Value[0:2])), nil
			}
		}
	}

	return 0, fmt.Errorf("no family id in the netlink response")
}

func rawIPData(ip net.IP) []byte {
	family := nl.GetIPFamily(ip)
	if family == nl.FAMILY_V4 {
		return ip.To4()
	}
	return ip
}

func newIPVSRequest(cmd uint8) *nl.NetlinkRequest {
	return newGenlRequest(ipvsFamily, cmd)
}

func newGenlRequest(familyID int, cmd uint8) *nl.NetlinkRequest {
	req := nl.NewNetlinkRequest(familyID, syscall.NLM_F_ACK)
	req.AddData(&genlMsgHdr{cmd: cmd, version: 1})
	return req
}

func execute(s *nl.NetlinkSocket, req *nl.NetlinkRequest, resType uint16) ([][]byte, error) {
	var (
		err error
	)

	if err := s.Send(req); err != nil {
		return nil, err
	}

	pid, err := s.GetPid()
	if err != nil {
		return nil, err
	}

	var res [][]byte

done:
	for {
		msgs, err := s.Receive()
		if err != nil {
			return nil, err
		}
		for _, m := range msgs {
			if m.Header.Seq != req.Seq {
				continue
			}
			if m.Header.Pid != pid {
				return nil, fmt.Errorf("Wrong pid %d, expected %d", m.Header.Pid, pid)
			}
			if m.Header.Type == syscall.NLMSG_DONE {
				break done
			}
			if m.Header.Type == syscall.NLMSG_ERROR {
				error := int32(native.Uint32(m.Data[0:4]))
				if error == 0 {
					break done
				}
				return nil, syscall.Errno(-error)
			}
			if resType != 0 && m.Header.Type != resType {
				continue
			}
			res = append(res, m.Data)
			if m.Header.Flags&syscall.NLM_F_MULTI == 0 {
				break done
			}
		}
	}
	return res, nil
}

func parseIP(ip []byte, family uint16) (net.IP, error) {

	var resIP net.IP

	switch family {
	case syscall.AF_INET:
		resIP = (net.IP)(ip[:4])
	case syscall.AF_INET6:
		resIP = (net.IP)(ip[:16])
	default:
		return nil, fmt.Errorf("parseIP Error ip=%v", ip)

	}
	return resIP, nil
}

// parseStats
func assembleStats(msg []byte) (SvcStats, error) {

	var s SvcStats

	attrs, err := nl.ParseRouteAttr(msg)
	if err != nil {
		return s, err
	}

	for _, attr := range attrs {
		attrType := int(attr.Attr.Type)
		switch attrType {
		case ipvsSvcStatsConns:
			s.Connections = native.Uint32(attr.Value)
		case ipvsSvcStatsPktsIn:
			s.PacketsIn = native.Uint32(attr.Value)
		case ipvsSvcStatsPktsOut:
			s.PacketsOut = native.Uint32(attr.Value)
		case ipvsSvcStatsBytesIn:
			s.BytesIn = native.Uint64(attr.Value)
		case ipvsSvcStatsBytesOut:
			s.BytesOut = native.Uint64(attr.Value)
		case ipvsSvcStatsCPS:
			s.CPS = native.Uint32(attr.Value)
		case ipvsSvcStatsPPSIn:
			s.PPSIn = native.Uint32(attr.Value)
		case ipvsSvcStatsPPSOut:
			s.PPSOut = native.Uint32(attr.Value)
		case ipvsSvcStatsBPSIn:
			s.BPSIn = native.Uint32(attr.Value)
		case ipvsSvcStatsBPSOut:
			s.BPSOut = native.Uint32(attr.Value)
		}
	}
	return s, nil
}

// assembleService assembles a services back from a hain of netlink attributes
func assembleService(attrs []syscall.NetlinkRouteAttr) (*Service, error) {

	var s Service

	for _, attr := range attrs {

		attrType := int(attr.Attr.Type)

		switch attrType {

		case ipvsSvcAttrAddressFamily:
			s.AddressFamily = native.Uint16(attr.Value)
		case ipvsSvcAttrProtocol:
			s.Protocol = native.Uint16(attr.Value)
		case ipvsSvcAttrAddress:
			ip, err := parseIP(attr.Value, s.AddressFamily)
			if err != nil {
				return nil, err
			}
			s.Address = ip
		case ipvsSvcAttrPort:
			s.Port = binary.BigEndian.Uint16(attr.Value)
		case ipvsSvcAttrFWMark:
			s.FWMark = native.Uint32(attr.Value)
		case ipvsSvcAttrSchedName:
			s.SchedName = nl.BytesToString(attr.Value)
		case ipvsSvcAttrFlags:
			s.Flags = native.Uint32(attr.Value)
		case ipvsSvcAttrTimeout:
			s.Timeout = native.Uint32(attr.Value)
		case ipvsSvcAttrNetmask:
			s.Netmask = native.Uint32(attr.Value)
		case ipvsSvcAttrStats:
			stats, err := assembleStats(attr.Value)
			if err != nil {
				return nil, err
			}
			s.Stats = stats
		}

	}
	return &s, nil
}

// parseService given a ipvs netlink response this function will respond with a valid service entry, an error otherwise
func (i *Handle) parseService(msg []byte) (*Service, error) {

	var s *Service

	//Remove General header for this message and parse the NetLink message
	hdr := deserializeGenlMsg(msg)
	NetLinkAttrs, err := nl.ParseRouteAttr(msg[hdr.Len():])
	if err != nil {
		return nil, err
	}
	if len(NetLinkAttrs) == 0 {
		return nil, fmt.Errorf("error no valid netlink message found while parsing service record")
	}

	//Now Parse and get IPVS related attributes messages packed in this message.
	ipvsAttrs, err := nl.ParseRouteAttr(NetLinkAttrs[0].Value)
	if err != nil {
		return nil, err
	}

	//Assemble all the IPVS related attribute messages and create a service record
	s, err = assembleService(ipvsAttrs)
	if err != nil {
		return nil, err
	}

	return s, nil
}

// doGetServicesCmd a wrapper which could be used commonly for both GetServices() and GetService(*Service)
func (i *Handle) doGetServicesCmd(svc *Service) ([]*Service, error) {
	var res []*Service

	msgs, err := i.doCmdwithResponse(svc, nil, ipvsCmdGetService)
	if err != nil {
		return nil, err
	}

	for _, msg := range msgs {
		srv, err := i.parseService(msg)
		if err != nil {
			return nil, err
		}
		res = append(res, srv)
	}

	return res, nil
}

func assembleDestination(attrs []syscall.NetlinkRouteAttr) (*Destination, error) {

	var d Destination

	for _, attr := range attrs {

		attrType := int(attr.Attr.Type)

		switch attrType {
		case ipvsDestAttrAddress:
			ip, err := parseIP(attr.Value, syscall.AF_INET)
			if err != nil {
				return nil, err
			}
			d.Address = ip
		case ipvsDestAttrPort:
			d.Port = binary.BigEndian.Uint16(attr.Value)
		case ipvsDestAttrForwardingMethod:
			d.ConnectionFlags = native.Uint32(attr.Value)
		case ipvsDestAttrWeight:
			d.Weight = int(native.Uint16(attr.Value))
		case ipvsDestAttrUpperThreshold:
			d.UpperThreshold = native.Uint32(attr.Value)
		case ipvsDestAttrLowerThreshold:
			d.LowerThreshold = native.Uint32(attr.Value)
		case ipvsDestAttrAddressFamily:
			d.AddressFamily = native.Uint16(attr.Value)
		}
	}
	return &d, nil
}

// parseDestination given a ipvs netlink response this function will respond with a valid destination entry, an error otherwise
func (i *Handle) parseDestination(msg []byte) (*Destination, error) {
	var dst *Destination

	//Remove General header for this message
	hdr := deserializeGenlMsg(msg)
	NetLinkAttrs, err := nl.ParseRouteAttr(msg[hdr.Len():])
	if err != nil {
		return nil, err
	}
	if len(NetLinkAttrs) == 0 {
		return nil, fmt.Errorf("error no valid netlink message found while parsing destination record")
	}

	//Now Parse and get IPVS related attributes messages packed in this message.
	ipvsAttrs, err := nl.ParseRouteAttr(NetLinkAttrs[0].Value)
	if err != nil {
		return nil, err
	}

	//Assemble netlink attributes and create a Destination record
	dst, err = assembleDestination(ipvsAttrs)
	if err != nil {
		return nil, err
	}

	return dst, nil
}

// doGetDestinationsCmd a wrapper function to be used by GetDestinations and GetDestination(d) apis
func (i *Handle) doGetDestinationsCmd(s *Service, d *Destination) ([]*Destination, error) {

	var res []*Destination

	msgs, err := i.doCmdwithResponse(s, d, ipvsCmdGetDest)
	if err != nil {
		return nil, err
	}

	for _, msg := range msgs {
		dest, err := i.parseDestination(msg)
		if err != nil {
			return res, err
		}
		res = append(res, dest)
	}
	return res, nil
}

// IPVS related netlink message format explained

/* EACH NETLINK MSG is of the below format, this is what we will receive from execute() api.
   If we have multiple netlink objects to process like GetServices() etc., execute() will
   supply an array of this below object

            NETLINK MSG
|-----------------------------------|
    0        1        2        3
|--------|--------|--------|--------| -
| CMD ID |  VER   |    RESERVED     | |==> General Message Header represented by genlMsgHdr
|-----------------------------------| -
|    ATTR LEN     |   ATTR TYPE     | |
|-----------------------------------| |
|                                   | |
|              VALUE                | |
|     []byte Array of IPVS MSG      | |==> Attribute Message represented by syscall.NetlinkRouteAttr
|        PADDED BY 4 BYTES          | |
|                                   | |
|-----------------------------------| -


 Once We strip genlMsgHdr from above NETLINK MSG, we should parse the VALUE.
 VALUE will have an array of netlink attributes (syscall.NetlinkRouteAttr) such that each attribute will
 represent a "Service" or "Destination" object's field.  If we assemble these attributes we can construct
 Service or Destination.

            IPVS MSG
|-----------------------------------|
     0        1        2        3
|--------|--------|--------|--------|
|    ATTR LEN     |    ATTR TYPE    |
|-----------------------------------|
|                                   |
|                                   |
| []byte IPVS ATTRIBUTE  BY 4 BYTES |
|                                   |
|                                   |
|-----------------------------------|
           NEXT ATTRIBUTE
|-----------------------------------|
|    ATTR LEN     |    ATTR TYPE    |
|-----------------------------------|
|                                   |
|                                   |
| []byte IPVS ATTRIBUTE  BY 4 BYTES |
|                                   |
|                                   |
|-----------------------------------|
           NEXT ATTRIBUTE
|-----------------------------------|
|    ATTR LEN     |    ATTR TYPE    |
|-----------------------------------|
|                                   |
|                                   |
| []byte IPVS ATTRIBUTE  BY 4 BYTES |
|                                   |
|                                   |
|-----------------------------------|

*/
