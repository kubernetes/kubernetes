package netlink

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"net"

	"github.com/vishvananda/netlink/nl"
	"golang.org/x/sys/unix"
)

// ConntrackTableType Conntrack table for the netlink operation
type ConntrackTableType uint8

const (
	// ConntrackTable Conntrack table
	// https://github.com/torvalds/linux/blob/master/include/uapi/linux/netfilter/nfnetlink.h -> #define NFNL_SUBSYS_CTNETLINK		 1
	ConntrackTable = 1
	// ConntrackExpectTable Conntrack expect table
	// https://github.com/torvalds/linux/blob/master/include/uapi/linux/netfilter/nfnetlink.h -> #define NFNL_SUBSYS_CTNETLINK_EXP 2
	ConntrackExpectTable = 2
)

const (
	// backward compatibility with golang 1.6 which does not have io.SeekCurrent
	seekCurrent = 1
)

// InetFamily Family type
type InetFamily uint8

//  -L [table] [options]          List conntrack or expectation table
//  -G [table] parameters         Get conntrack or expectation

//  -I [table] parameters         Create a conntrack or expectation
//  -U [table] parameters         Update a conntrack
//  -E [table] [options]          Show events

//  -C [table]                    Show counter
//  -S                            Show statistics

// ConntrackTableList returns the flow list of a table of a specific family
// conntrack -L [table] [options]          List conntrack or expectation table
func ConntrackTableList(table ConntrackTableType, family InetFamily) ([]*ConntrackFlow, error) {
	return pkgHandle.ConntrackTableList(table, family)
}

// ConntrackTableFlush flushes all the flows of a specified table
// conntrack -F [table]            Flush table
// The flush operation applies to all the family types
func ConntrackTableFlush(table ConntrackTableType) error {
	return pkgHandle.ConntrackTableFlush(table)
}

// ConntrackDeleteFilter deletes entries on the specified table on the base of the filter
// conntrack -D [table] parameters         Delete conntrack or expectation
func ConntrackDeleteFilter(table ConntrackTableType, family InetFamily, filter CustomConntrackFilter) (uint, error) {
	return pkgHandle.ConntrackDeleteFilter(table, family, filter)
}

// ConntrackTableList returns the flow list of a table of a specific family using the netlink handle passed
// conntrack -L [table] [options]          List conntrack or expectation table
func (h *Handle) ConntrackTableList(table ConntrackTableType, family InetFamily) ([]*ConntrackFlow, error) {
	res, err := h.dumpConntrackTable(table, family)
	if err != nil {
		return nil, err
	}

	// Deserialize all the flows
	var result []*ConntrackFlow
	for _, dataRaw := range res {
		result = append(result, parseRawData(dataRaw))
	}

	return result, nil
}

// ConntrackTableFlush flushes all the flows of a specified table using the netlink handle passed
// conntrack -F [table]            Flush table
// The flush operation applies to all the family types
func (h *Handle) ConntrackTableFlush(table ConntrackTableType) error {
	req := h.newConntrackRequest(table, unix.AF_INET, nl.IPCTNL_MSG_CT_DELETE, unix.NLM_F_ACK)
	_, err := req.Execute(unix.NETLINK_NETFILTER, 0)
	return err
}

// ConntrackDeleteFilter deletes entries on the specified table on the base of the filter using the netlink handle passed
// conntrack -D [table] parameters         Delete conntrack or expectation
func (h *Handle) ConntrackDeleteFilter(table ConntrackTableType, family InetFamily, filter CustomConntrackFilter) (uint, error) {
	res, err := h.dumpConntrackTable(table, family)
	if err != nil {
		return 0, err
	}

	var matched uint
	for _, dataRaw := range res {
		flow := parseRawData(dataRaw)
		if match := filter.MatchConntrackFlow(flow); match {
			req2 := h.newConntrackRequest(table, family, nl.IPCTNL_MSG_CT_DELETE, unix.NLM_F_ACK)
			// skip the first 4 byte that are the netfilter header, the newConntrackRequest is adding it already
			req2.AddRawData(dataRaw[4:])
			req2.Execute(unix.NETLINK_NETFILTER, 0)
			matched++
		}
	}

	return matched, nil
}

func (h *Handle) newConntrackRequest(table ConntrackTableType, family InetFamily, operation, flags int) *nl.NetlinkRequest {
	// Create the Netlink request object
	req := h.newNetlinkRequest((int(table)<<8)|operation, flags)
	// Add the netfilter header
	msg := &nl.Nfgenmsg{
		NfgenFamily: uint8(family),
		Version:     nl.NFNETLINK_V0,
		ResId:       0,
	}
	req.AddData(msg)
	return req
}

func (h *Handle) dumpConntrackTable(table ConntrackTableType, family InetFamily) ([][]byte, error) {
	req := h.newConntrackRequest(table, family, nl.IPCTNL_MSG_CT_GET, unix.NLM_F_DUMP)
	return req.Execute(unix.NETLINK_NETFILTER, 0)
}

// The full conntrack flow structure is very complicated and can be found in the file:
// http://git.netfilter.org/libnetfilter_conntrack/tree/include/internal/object.h
// For the time being, the structure below allows to parse and extract the base information of a flow
type ipTuple struct {
	Bytes    uint64
	DstIP    net.IP
	DstPort  uint16
	Packets  uint64
	Protocol uint8
	SrcIP    net.IP
	SrcPort  uint16
}

type ConntrackFlow struct {
	FamilyType uint8
	Forward    ipTuple
	Reverse    ipTuple
	Mark       uint32
}

func (s *ConntrackFlow) String() string {
	// conntrack cmd output:
	// udp      17 src=127.0.0.1 dst=127.0.0.1 sport=4001 dport=1234 packets=5 bytes=532 [UNREPLIED] src=127.0.0.1 dst=127.0.0.1 sport=1234 dport=4001 packets=10 bytes=1078 mark=0
	return fmt.Sprintf("%s\t%d src=%s dst=%s sport=%d dport=%d packets=%d bytes=%d\tsrc=%s dst=%s sport=%d dport=%d packets=%d bytes=%d mark=%d",
		nl.L4ProtoMap[s.Forward.Protocol], s.Forward.Protocol,
		s.Forward.SrcIP.String(), s.Forward.DstIP.String(), s.Forward.SrcPort, s.Forward.DstPort, s.Forward.Packets, s.Forward.Bytes,
		s.Reverse.SrcIP.String(), s.Reverse.DstIP.String(), s.Reverse.SrcPort, s.Reverse.DstPort, s.Reverse.Packets, s.Reverse.Bytes,
		s.Mark)
}

// This method parse the ip tuple structure
// The message structure is the following:
// <len, [CTA_IP_V4_SRC|CTA_IP_V6_SRC], 16 bytes for the IP>
// <len, [CTA_IP_V4_DST|CTA_IP_V6_DST], 16 bytes for the IP>
// <len, NLA_F_NESTED|nl.CTA_TUPLE_PROTO, 1 byte for the protocol, 3 bytes of padding>
// <len, CTA_PROTO_SRC_PORT, 2 bytes for the source port, 2 bytes of padding>
// <len, CTA_PROTO_DST_PORT, 2 bytes for the source port, 2 bytes of padding>
func parseIpTuple(reader *bytes.Reader, tpl *ipTuple) uint8 {
	for i := 0; i < 2; i++ {
		_, t, _, v := parseNfAttrTLV(reader)
		switch t {
		case nl.CTA_IP_V4_SRC, nl.CTA_IP_V6_SRC:
			tpl.SrcIP = v
		case nl.CTA_IP_V4_DST, nl.CTA_IP_V6_DST:
			tpl.DstIP = v
		}
	}
	// Skip the next 4 bytes  nl.NLA_F_NESTED|nl.CTA_TUPLE_PROTO
	reader.Seek(4, seekCurrent)
	_, t, _, v := parseNfAttrTLV(reader)
	if t == nl.CTA_PROTO_NUM {
		tpl.Protocol = uint8(v[0])
	}
	// Skip some padding 3 bytes
	reader.Seek(3, seekCurrent)
	for i := 0; i < 2; i++ {
		_, t, _ := parseNfAttrTL(reader)
		switch t {
		case nl.CTA_PROTO_SRC_PORT:
			parseBERaw16(reader, &tpl.SrcPort)
		case nl.CTA_PROTO_DST_PORT:
			parseBERaw16(reader, &tpl.DstPort)
		}
		// Skip some padding 2 byte
		reader.Seek(2, seekCurrent)
	}
	return tpl.Protocol
}

func parseNfAttrTLV(r *bytes.Reader) (isNested bool, attrType, len uint16, value []byte) {
	isNested, attrType, len = parseNfAttrTL(r)

	value = make([]byte, len)
	binary.Read(r, binary.BigEndian, &value)
	return isNested, attrType, len, value
}

func parseNfAttrTL(r *bytes.Reader) (isNested bool, attrType, len uint16) {
	binary.Read(r, nl.NativeEndian(), &len)
	len -= nl.SizeofNfattr

	binary.Read(r, nl.NativeEndian(), &attrType)
	isNested = (attrType & nl.NLA_F_NESTED) == nl.NLA_F_NESTED
	attrType = attrType & (nl.NLA_F_NESTED - 1)

	return isNested, attrType, len
}

func parseBERaw16(r *bytes.Reader, v *uint16) {
	binary.Read(r, binary.BigEndian, v)
}

func parseBERaw32(r *bytes.Reader, v *uint32) {
	binary.Read(r, binary.BigEndian, v)
}

func parseBERaw64(r *bytes.Reader, v *uint64) {
	binary.Read(r, binary.BigEndian, v)
}

func parseByteAndPacketCounters(r *bytes.Reader) (bytes, packets uint64) {
	for i := 0; i < 2; i++ {
		switch _, t, _ := parseNfAttrTL(r); t {
		case nl.CTA_COUNTERS_BYTES:
			parseBERaw64(r, &bytes)
		case nl.CTA_COUNTERS_PACKETS:
			parseBERaw64(r, &packets)
		default:
			return
		}
	}
	return
}

func parseConnectionMark(r *bytes.Reader) (mark uint32) {
	parseBERaw32(r, &mark)
	return
}

func parseRawData(data []byte) *ConntrackFlow {
	s := &ConntrackFlow{}
	// First there is the Nfgenmsg header
	// consume only the family field
	reader := bytes.NewReader(data)
	binary.Read(reader, nl.NativeEndian(), &s.FamilyType)

	// skip rest of the Netfilter header
	reader.Seek(3, seekCurrent)
	// The message structure is the following:
	// <len, NLA_F_NESTED|CTA_TUPLE_ORIG> 4 bytes
	// <len, NLA_F_NESTED|CTA_TUPLE_IP> 4 bytes
	// flow information of the forward flow
	// <len, NLA_F_NESTED|CTA_TUPLE_REPLY> 4 bytes
	// <len, NLA_F_NESTED|CTA_TUPLE_IP> 4 bytes
	// flow information of the reverse flow
	for reader.Len() > 0 {
		if nested, t, l := parseNfAttrTL(reader); nested {
			switch t {
			case nl.CTA_TUPLE_ORIG:
				if nested, t, _ = parseNfAttrTL(reader); nested && t == nl.CTA_TUPLE_IP {
					parseIpTuple(reader, &s.Forward)
				}
			case nl.CTA_TUPLE_REPLY:
				if nested, t, _ = parseNfAttrTL(reader); nested && t == nl.CTA_TUPLE_IP {
					parseIpTuple(reader, &s.Reverse)
				} else {
					// Header not recognized skip it
					reader.Seek(int64(l), seekCurrent)
				}
			case nl.CTA_COUNTERS_ORIG:
				s.Forward.Bytes, s.Forward.Packets = parseByteAndPacketCounters(reader)
			case nl.CTA_COUNTERS_REPLY:
				s.Reverse.Bytes, s.Reverse.Packets = parseByteAndPacketCounters(reader)
			}
		} else {
			switch t {
			case nl.CTA_MARK:
				s.Mark = parseConnectionMark(reader)
			}
		}
	}
	return s
}

// Conntrack parameters and options:
//   -n, --src-nat ip                      source NAT ip
//   -g, --dst-nat ip                      destination NAT ip
//   -j, --any-nat ip                      source or destination NAT ip
//   -m, --mark mark                       Set mark
//   -c, --secmark secmark                 Set selinux secmark
//   -e, --event-mask eventmask            Event mask, eg. NEW,DESTROY
//   -z, --zero                            Zero counters while listing
//   -o, --output type[,...]               Output format, eg. xml
//   -l, --label label[,...]               conntrack labels

// Common parameters and options:
//   -s, --src, --orig-src ip              Source address from original direction
//   -d, --dst, --orig-dst ip              Destination address from original direction
//   -r, --reply-src ip            Source address from reply direction
//   -q, --reply-dst ip            Destination address from reply direction
//   -p, --protonum proto          Layer 4 Protocol, eg. 'tcp'
//   -f, --family proto            Layer 3 Protocol, eg. 'ipv6'
//   -t, --timeout timeout         Set timeout
//   -u, --status status           Set status, eg. ASSURED
//   -w, --zone value              Set conntrack zone
//   --orig-zone value             Set zone for original direction
//   --reply-zone value            Set zone for reply direction
//   -b, --buffer-size             Netlink socket buffer size
//   --mask-src ip                 Source mask address
//   --mask-dst ip                 Destination mask address

// Layer 4 Protocol common parameters and options:
// TCP, UDP, SCTP, UDPLite and DCCP
//    --sport, --orig-port-src port    Source port in original direction
//    --dport, --orig-port-dst port    Destination port in original direction

// Filter types
type ConntrackFilterType uint8

const (
	ConntrackOrigSrcIP   = iota                // -orig-src ip    Source address from original direction
	ConntrackOrigDstIP                         // -orig-dst ip    Destination address from original direction
	ConntrackReplySrcIP                        // --reply-src ip  Reply Source IP
	ConntrackReplyDstIP                        // --reply-dst ip  Reply Destination IP
	ConntrackReplyAnyIP                        // Match source or destination reply IP
	ConntrackOrigSrcPort                       // --orig-port-src port    Source port in original direction
	ConntrackOrigDstPort                       // --orig-port-dst port    Destination port in original direction
	ConntrackNatSrcIP    = ConntrackReplySrcIP // deprecated use instead ConntrackReplySrcIP
	ConntrackNatDstIP    = ConntrackReplyDstIP // deprecated use instead ConntrackReplyDstIP
	ConntrackNatAnyIP    = ConntrackReplyAnyIP // deprecated use instead ConntrackReplyAnyIP
)

type CustomConntrackFilter interface {
	// MatchConntrackFlow applies the filter to the flow and returns true if the flow matches
	// the filter or false otherwise
	MatchConntrackFlow(flow *ConntrackFlow) bool
}

type ConntrackFilter struct {
	ipFilter    map[ConntrackFilterType]net.IP
	portFilter  map[ConntrackFilterType]uint16
	protoFilter uint8
}

// AddIP adds an IP to the conntrack filter
func (f *ConntrackFilter) AddIP(tp ConntrackFilterType, ip net.IP) error {
	if f.ipFilter == nil {
		f.ipFilter = make(map[ConntrackFilterType]net.IP)
	}
	if _, ok := f.ipFilter[tp]; ok {
		return errors.New("Filter attribute already present")
	}
	f.ipFilter[tp] = ip
	return nil
}

// AddPort adds a Port to the conntrack filter if the Layer 4 protocol allows it
func (f *ConntrackFilter) AddPort(tp ConntrackFilterType, port uint16) error {
	switch f.protoFilter {
	// TCP, UDP, DCCP, SCTP, UDPLite
	case 6, 17, 33, 132, 136:
	default:
		return fmt.Errorf("Filter attribute not available without a valid Layer 4 protocol: %d", f.protoFilter)
	}

	if f.portFilter == nil {
		f.portFilter = make(map[ConntrackFilterType]uint16)
	}
	if _, ok := f.portFilter[tp]; ok {
		return errors.New("Filter attribute already present")
	}
	f.portFilter[tp] = port
	return nil
}

// AddProtocol adds the Layer 4 protocol to the conntrack filter
func (f *ConntrackFilter) AddProtocol(proto uint8) error {
	if f.protoFilter != 0 {
		return errors.New("Filter attribute already present")
	}
	f.protoFilter = proto
	return nil
}

// MatchConntrackFlow applies the filter to the flow and returns true if the flow matches the filter
// false otherwise
func (f *ConntrackFilter) MatchConntrackFlow(flow *ConntrackFlow) bool {
	if len(f.ipFilter) == 0 && len(f.portFilter) == 0 && f.protoFilter == 0 {
		// empty filter always not match
		return false
	}

	// -p, --protonum proto          Layer 4 Protocol, eg. 'tcp'
	if f.protoFilter != 0 && flow.Forward.Protocol != f.protoFilter {
		// different Layer 4 protocol always not match
		return false
	}

	match := true

	// IP conntrack filter
	if len(f.ipFilter) > 0 {
		// -orig-src ip   Source address from original direction
		if elem, found := f.ipFilter[ConntrackOrigSrcIP]; found {
			match = match && elem.Equal(flow.Forward.SrcIP)
		}

		// -orig-dst ip   Destination address from original direction
		if elem, found := f.ipFilter[ConntrackOrigDstIP]; match && found {
			match = match && elem.Equal(flow.Forward.DstIP)
		}

		// -src-nat ip    Source NAT ip
		if elem, found := f.ipFilter[ConntrackReplySrcIP]; match && found {
			match = match && elem.Equal(flow.Reverse.SrcIP)
		}

		// -dst-nat ip    Destination NAT ip
		if elem, found := f.ipFilter[ConntrackReplyDstIP]; match && found {
			match = match && elem.Equal(flow.Reverse.DstIP)
		}

		// Match source or destination reply IP
		if elem, found := f.ipFilter[ConntrackReplyAnyIP]; match && found {
			match = match && (elem.Equal(flow.Reverse.SrcIP) || elem.Equal(flow.Reverse.DstIP))
		}
	}

	// Layer 4 Port filter
	if len(f.portFilter) > 0 {
		// -orig-port-src port	Source port from original direction
		if elem, found := f.portFilter[ConntrackOrigSrcPort]; match && found {
			match = match && elem == flow.Forward.SrcPort
		}

		// -orig-port-dst port	Destination port from original direction
		if elem, found := f.portFilter[ConntrackOrigDstPort]; match && found {
			match = match && elem == flow.Forward.DstPort
		}
	}

	return match
}

var _ CustomConntrackFilter = (*ConntrackFilter)(nil)
