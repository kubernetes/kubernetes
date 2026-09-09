package netlink

import (
	"encoding/binary"
	"log"
	"net"
	"syscall"

	"github.com/vishvananda/netlink/nl"
	"golang.org/x/sys/unix"
)

// IPSetEntry is used for adding, updating, retreiving and deleting entries
type IPSetEntry struct {
	Comment  string
	MAC      net.HardwareAddr
	IP       net.IP
	CIDR     uint8
	Timeout  *uint32
	Packets  *uint64
	Bytes    *uint64
	Protocol *uint8
	Port     *uint16
	IP2      net.IP
	CIDR2    uint8
	IFace    string
	Mark     *uint32

	Replace bool // replace existing entry
}

// IPSetResult is the result of a dump request for a set
type IPSetResult struct {
	Nfgenmsg           *nl.Nfgenmsg
	Protocol           uint8
	ProtocolMinVersion uint8
	Revision           uint8
	Family             uint8
	Flags              uint8
	SetName            string
	TypeName           string
	Comment            string
	MarkMask           uint32

	IPFrom   net.IP
	IPTo     net.IP
	PortFrom uint16
	PortTo   uint16

	HashSize     uint32
	NumEntries   uint32
	MaxElements  uint32
	References   uint32
	SizeInMemory uint32
	CadtFlags    uint32
	Timeout      *uint32
	LineNo       uint32

	Entries []IPSetEntry
}

// IpsetCreateOptions is the options struct for creating a new ipset
type IpsetCreateOptions struct {
	Replace  bool // replace existing ipset
	Timeout  *uint32
	Counters bool
	Comments bool
	Skbinfo  bool

	Family      uint8
	Revision    uint8
	IPFrom      net.IP
	IPTo        net.IP
	PortFrom    uint16
	PortTo      uint16
	MaxElements uint32
}

// IpsetProtocol returns the ipset protocol version from the kernel
func IpsetProtocol() (uint8, uint8, error) {
	return pkgHandle.IpsetProtocol()
}

// IpsetCreate creates a new ipset
func IpsetCreate(setname, typename string, options IpsetCreateOptions) error {
	return pkgHandle.IpsetCreate(setname, typename, options)
}

// IpsetDestroy destroys an existing ipset
func IpsetDestroy(setname string) error {
	return pkgHandle.IpsetDestroy(setname)
}

// IpsetFlush flushes an existing ipset
func IpsetFlush(setname string) error {
	return pkgHandle.IpsetFlush(setname)
}

// IpsetSwap swaps two ipsets.
func IpsetSwap(setname, othersetname string) error {
	return pkgHandle.IpsetSwap(setname, othersetname)
}

// IpsetList dumps an specific ipset.
func IpsetList(setname string) (*IPSetResult, error) {
	return pkgHandle.IpsetList(setname)
}

// IpsetListAll dumps all ipsets.
func IpsetListAll() ([]IPSetResult, error) {
	return pkgHandle.IpsetListAll()
}

// IpsetAdd adds an entry to an existing ipset.
func IpsetAdd(setname string, entry *IPSetEntry) error {
	return pkgHandle.IpsetAdd(setname, entry)
}

// IpsetDel deletes an entry from an existing ipset.
func IpsetDel(setname string, entry *IPSetEntry) error {
	return pkgHandle.IpsetDel(setname, entry)
}

// IpsetTest tests whether an entry is in a set or not.
func IpsetTest(setname string, entry *IPSetEntry) (bool, error) {
	return pkgHandle.IpsetTest(setname, entry)
}

func (h *Handle) IpsetProtocol() (protocol uint8, minVersion uint8, err error) {
	req := h.newIpsetRequest(nl.IPSET_CMD_PROTOCOL)
	msgs, err := req.Execute(unix.NETLINK_NETFILTER, 0)

	if err != nil {
		return 0, 0, err
	}
	response := ipsetUnserialize(msgs)
	return response.Protocol, response.ProtocolMinVersion, nil
}

func (h *Handle) IpsetCreate(setname, typename string, options IpsetCreateOptions) error {
	req := h.newIpsetRequest(nl.IPSET_CMD_CREATE)

	if !options.Replace {
		req.Flags |= unix.NLM_F_EXCL
	}

	req.AddData(nl.NewRtAttr(nl.IPSET_ATTR_SETNAME, nl.ZeroTerminated(setname)))
	req.AddData(nl.NewRtAttr(nl.IPSET_ATTR_TYPENAME, nl.ZeroTerminated(typename)))

	cadtFlags := optionsToBitflag(options)

	revision := options.Revision
	if revision == 0 {
		revision = getIpsetDefaultRevision(typename, cadtFlags)
	}
	req.AddData(nl.NewRtAttr(nl.IPSET_ATTR_REVISION, nl.Uint8Attr(revision)))

	data := nl.NewRtAttr(nl.IPSET_ATTR_DATA|int(nl.NLA_F_NESTED), nil)

	var family uint8
	switch typename {
	case "hash:mac":
	case "bitmap:port":
		buf := make([]byte, 4)
		binary.BigEndian.PutUint16(buf, options.PortFrom)
		binary.BigEndian.PutUint16(buf[2:], options.PortTo)
		data.AddChild(nl.NewRtAttr(nl.IPSET_ATTR_PORT_FROM|int(nl.NLA_F_NET_BYTEORDER), buf[:2]))
		data.AddChild(nl.NewRtAttr(nl.IPSET_ATTR_PORT_TO|int(nl.NLA_F_NET_BYTEORDER), buf[2:]))
	default:
		family = options.Family
		if family == 0 {
			family = unix.AF_INET
		}
	}

	req.AddData(nl.NewRtAttr(nl.IPSET_ATTR_FAMILY, nl.Uint8Attr(family)))

	if options.MaxElements != 0 {
		data.AddChild(&nl.Uint32Attribute{Type: nl.IPSET_ATTR_MAXELEM | nl.NLA_F_NET_BYTEORDER, Value: options.MaxElements})
	}

	if timeout := options.Timeout; timeout != nil {
		data.AddChild(&nl.Uint32Attribute{Type: nl.IPSET_ATTR_TIMEOUT | nl.NLA_F_NET_BYTEORDER, Value: *timeout})
	}

	if cadtFlags != 0 {
		data.AddChild(&nl.Uint32Attribute{Type: nl.IPSET_ATTR_CADT_FLAGS | nl.NLA_F_NET_BYTEORDER, Value: cadtFlags})
	}

	req.AddData(data)
	_, err := ipsetExecute(req)
	return err
}

func (h *Handle) IpsetDestroy(setname string) error {
	req := h.newIpsetRequest(nl.IPSET_CMD_DESTROY)
	req.AddData(nl.NewRtAttr(nl.IPSET_ATTR_SETNAME, nl.ZeroTerminated(setname)))
	_, err := ipsetExecute(req)
	return err
}

func (h *Handle) IpsetFlush(setname string) error {
	req := h.newIpsetRequest(nl.IPSET_CMD_FLUSH)
	req.AddData(nl.NewRtAttr(nl.IPSET_ATTR_SETNAME, nl.ZeroTerminated(setname)))
	_, err := ipsetExecute(req)
	return err
}

func (h *Handle) IpsetSwap(setname, othersetname string) error {
	req := h.newIpsetRequest(nl.IPSET_CMD_SWAP)
	req.AddData(nl.NewRtAttr(nl.IPSET_ATTR_SETNAME, nl.ZeroTerminated(setname)))
	req.AddData(nl.NewRtAttr(nl.IPSET_ATTR_TYPENAME, nl.ZeroTerminated(othersetname)))
	_, err := ipsetExecute(req)
	return err
}

func (h *Handle) IpsetList(name string) (*IPSetResult, error) {
	req := h.newIpsetRequest(nl.IPSET_CMD_LIST)
	req.AddData(nl.NewRtAttr(nl.IPSET_ATTR_SETNAME, nl.ZeroTerminated(name)))

	msgs, err := ipsetExecute(req)
	if err != nil {
		return nil, err
	}

	result := ipsetUnserialize(msgs)
	return &result, nil
}

func (h *Handle) IpsetListAll() ([]IPSetResult, error) {
	req := h.newIpsetRequest(nl.IPSET_CMD_LIST)

	msgs, err := ipsetExecute(req)
	if err != nil {
		return nil, err
	}

	result := make([]IPSetResult, len(msgs))
	for i, msg := range msgs {
		result[i].unserialize(msg)
	}

	return result, nil
}

// IpsetAdd adds an entry to an existing ipset.
func (h *Handle) IpsetAdd(setname string, entry *IPSetEntry) error {
	return h.ipsetAddDel(nl.IPSET_CMD_ADD, setname, entry)
}

// IpsetDel deletes an entry from an existing ipset.
func (h *Handle) IpsetDel(setname string, entry *IPSetEntry) error {
	return h.ipsetAddDel(nl.IPSET_CMD_DEL, setname, entry)
}

func encodeIP(ip net.IP) (*nl.RtAttr, error) {
	typ := int(nl.NLA_F_NET_BYTEORDER)
	if ip4 := ip.To4(); ip4 != nil {
		typ |= nl.IPSET_ATTR_IPADDR_IPV4
		ip = ip4
	} else {
		typ |= nl.IPSET_ATTR_IPADDR_IPV6
	}

	return nl.NewRtAttr(typ, ip), nil
}

func buildEntryData(entry *IPSetEntry) (*nl.RtAttr, error) {
	data := nl.NewRtAttr(nl.IPSET_ATTR_DATA|int(nl.NLA_F_NESTED), nil)

	if entry.Comment != "" {
		data.AddChild(nl.NewRtAttr(nl.IPSET_ATTR_COMMENT, nl.ZeroTerminated(entry.Comment)))
	}

	if entry.Timeout != nil {
		data.AddChild(&nl.Uint32Attribute{Type: nl.IPSET_ATTR_TIMEOUT | nl.NLA_F_NET_BYTEORDER, Value: *entry.Timeout})
	}

	if entry.IP != nil {
		nestedData, err := encodeIP(entry.IP)
		if err != nil {
			return nil, err
		}
		data.AddChild(nl.NewRtAttr(nl.IPSET_ATTR_IP|int(nl.NLA_F_NESTED), nestedData.Serialize()))
	}

	if entry.MAC != nil {
		data.AddChild(nl.NewRtAttr(nl.IPSET_ATTR_ETHER, entry.MAC))
	}

	if entry.CIDR != 0 {
		data.AddChild(nl.NewRtAttr(nl.IPSET_ATTR_CIDR, nl.Uint8Attr(entry.CIDR)))
	}

	if entry.IP2 != nil {
		nestedData, err := encodeIP(entry.IP2)
		if err != nil {
			return nil, err
		}
		data.AddChild(nl.NewRtAttr(nl.IPSET_ATTR_IP2|int(nl.NLA_F_NESTED), nestedData.Serialize()))
	}

	if entry.CIDR2 != 0 {
		data.AddChild(nl.NewRtAttr(nl.IPSET_ATTR_CIDR2, nl.Uint8Attr(entry.CIDR2)))
	}

	if entry.Port != nil {
		if entry.Protocol == nil {
			// use tcp protocol as default
			val := uint8(unix.IPPROTO_TCP)
			entry.Protocol = &val
		}
		data.AddChild(nl.NewRtAttr(nl.IPSET_ATTR_PROTO, nl.Uint8Attr(*entry.Protocol)))
		buf := make([]byte, 2)
		binary.BigEndian.PutUint16(buf, *entry.Port)
		data.AddChild(nl.NewRtAttr(int(nl.IPSET_ATTR_PORT|nl.NLA_F_NET_BYTEORDER), buf))
	}

	if entry.IFace != "" {
		data.AddChild(nl.NewRtAttr(nl.IPSET_ATTR_IFACE, nl.ZeroTerminated(entry.IFace)))
	}

	if entry.Mark != nil {
		data.AddChild(&nl.Uint32Attribute{Type: nl.IPSET_ATTR_MARK | nl.NLA_F_NET_BYTEORDER, Value: *entry.Mark})
	}
	return data, nil
}

func (h *Handle) ipsetAddDel(nlCmd int, setname string, entry *IPSetEntry) error {
	req := h.newIpsetRequest(nlCmd)
	req.AddData(nl.NewRtAttr(nl.IPSET_ATTR_SETNAME, nl.ZeroTerminated(setname)))

	if !entry.Replace {
		req.Flags |= unix.NLM_F_EXCL
	}

	data, err := buildEntryData(entry)
	if err != nil {
		return err
	}
	data.AddChild(&nl.Uint32Attribute{Type: nl.IPSET_ATTR_LINENO | nl.NLA_F_NET_BYTEORDER, Value: 0})
	req.AddData(data)

	_, err = ipsetExecute(req)
	return err
}

func (h *Handle) IpsetTest(setname string, entry *IPSetEntry) (bool, error) {
	req := h.newIpsetRequest(nl.IPSET_CMD_TEST)
	req.AddData(nl.NewRtAttr(nl.IPSET_ATTR_SETNAME, nl.ZeroTerminated(setname)))

	if !entry.Replace {
		req.Flags |= unix.NLM_F_EXCL
	}

	data, err := buildEntryData(entry)
	if err != nil {
		return false, err
	}
	req.AddData(data)

	_, err = ipsetExecute(req)
	if err != nil {
		if err == nl.IPSetError(nl.IPSET_ERR_EXIST) {
			// not exist
			return false, nil
		}
		return false, err
	}
	return true, nil
}

func (h *Handle) newIpsetRequest(cmd int) *nl.NetlinkRequest {
	req := h.newNetlinkRequest(cmd|(unix.NFNL_SUBSYS_IPSET<<8), nl.GetIpsetFlags(cmd))

	// Add the netfilter header
	msg := &nl.Nfgenmsg{
		NfgenFamily: uint8(unix.AF_NETLINK),
		Version:     nl.NFNETLINK_V0,
		ResId:       0,
	}
	req.AddData(msg)
	req.AddData(nl.NewRtAttr(nl.IPSET_ATTR_PROTOCOL, nl.Uint8Attr(nl.IPSET_PROTOCOL)))

	return req
}

// NOTE: This can't just take typename into account, it also has to take desired
// feature support into account, on a per-set-type basis, to return the correct revision, see e.g.
// https://github.com/Olipro/ipset/blob/9f145b49100104d6570fe5c31a5236816ebb4f8f/kernel/net/netfilter/ipset/ip_set_hash_ipport.c#L30
//
// This means that whenever a new "type" of ipset is added, returning the "correct" default revision
// requires adding a new case here for that type, and consulting the ipset C code to figure out the correct
// combination of type name, feature bit flags, and revision ranges.
//
// Care should be taken as some types share the same revision ranges for the same features, and others do not.
// When in doubt, mimic the C code.
func getIpsetDefaultRevision(typename string, featureFlags uint32) uint8 {
	switch typename {
	case "hash:ip,port",
		"hash:ip,port,ip":
		// Taken from
		// - ipset/kernel/net/netfilter/ipset/ip_set_hash_ipport.c
		// - ipset/kernel/net/netfilter/ipset/ip_set_hash_ipportip.c
		if (featureFlags & nl.IPSET_FLAG_WITH_SKBINFO) != 0 {
			return 5
		}

		if (featureFlags & nl.IPSET_FLAG_WITH_FORCEADD) != 0 {
			return 4
		}

		if (featureFlags & nl.IPSET_FLAG_WITH_COMMENT) != 0 {
			return 3
		}

		if (featureFlags & nl.IPSET_FLAG_WITH_COUNTERS) != 0 {
			return 2
		}

		// the min revision this library supports for this type
		return 1

	case "hash:ip,port,net",
		"hash:net,port":
		// Taken from
		// - ipset/kernel/net/netfilter/ipset/ip_set_hash_ipportnet.c
		// - ipset/kernel/net/netfilter/ipset/ip_set_hash_netport.c
		if (featureFlags & nl.IPSET_FLAG_WITH_SKBINFO) != 0 {
			return 7
		}

		if (featureFlags & nl.IPSET_FLAG_WITH_FORCEADD) != 0 {
			return 6
		}

		if (featureFlags & nl.IPSET_FLAG_WITH_COMMENT) != 0 {
			return 5
		}

		if (featureFlags & nl.IPSET_FLAG_WITH_COUNTERS) != 0 {
			return 4
		}

		if (featureFlags & nl.IPSET_FLAG_NOMATCH) != 0 {
			return 3
		}
		// the min revision this library supports for this type
		return 2

	case "hash:ip":
		// Taken from
		// - ipset/kernel/net/netfilter/ipset/ip_set_hash_ip.c
		if (featureFlags & nl.IPSET_FLAG_WITH_SKBINFO) != 0 {
			return 4
		}

		if (featureFlags & nl.IPSET_FLAG_WITH_FORCEADD) != 0 {
			return 3
		}

		if (featureFlags & nl.IPSET_FLAG_WITH_COMMENT) != 0 {
			return 2
		}

		// the min revision this library supports for this type
		return 1
	}

	// can't map the correct revision for this type.
	return 0
}

func ipsetExecute(req *nl.NetlinkRequest) (msgs [][]byte, err error) {
	msgs, err = req.Execute(unix.NETLINK_NETFILTER, 0)

	if err != nil {
		if errno := int(err.(syscall.Errno)); errno >= nl.IPSET_ERR_PRIVATE {
			err = nl.IPSetError(uintptr(errno))
		}
	}
	return
}

func ipsetUnserialize(msgs [][]byte) (result IPSetResult) {
	for _, msg := range msgs {
		result.unserialize(msg)
	}
	return result
}

func (result *IPSetResult) unserialize(msg []byte) {
	result.Nfgenmsg = nl.DeserializeNfgenmsg(msg)

	for attr := range nl.ParseAttributes(msg[4:]) {
		switch attr.Type {
		case nl.IPSET_ATTR_PROTOCOL:
			result.Protocol = attr.Value[0]
		case nl.IPSET_ATTR_SETNAME:
			result.SetName = nl.BytesToString(attr.Value)
		case nl.IPSET_ATTR_COMMENT:
			result.Comment = nl.BytesToString(attr.Value)
		case nl.IPSET_ATTR_TYPENAME:
			result.TypeName = nl.BytesToString(attr.Value)
		case nl.IPSET_ATTR_REVISION:
			result.Revision = attr.Value[0]
		case nl.IPSET_ATTR_FAMILY:
			result.Family = attr.Value[0]
		case nl.IPSET_ATTR_FLAGS:
			result.Flags = attr.Value[0]
		case nl.IPSET_ATTR_DATA | nl.NLA_F_NESTED:
			result.parseAttrData(attr.Value)
		case nl.IPSET_ATTR_ADT | nl.NLA_F_NESTED:
			result.parseAttrADT(attr.Value)
		case nl.IPSET_ATTR_PROTOCOL_MIN:
			result.ProtocolMinVersion = attr.Value[0]
		case nl.IPSET_ATTR_MARKMASK:
			result.MarkMask = attr.Uint32()
		default:
			log.Printf("unknown ipset attribute from kernel: %+v %v", attr, attr.Type&nl.NLA_TYPE_MASK)
		}
	}
}

func (result *IPSetResult) parseAttrData(data []byte) {
	for attr := range nl.ParseAttributes(data) {
		switch attr.Type {
		case nl.IPSET_ATTR_HASHSIZE | nl.NLA_F_NET_BYTEORDER:
			result.HashSize = attr.Uint32()
		case nl.IPSET_ATTR_MAXELEM | nl.NLA_F_NET_BYTEORDER:
			result.MaxElements = attr.Uint32()
		case nl.IPSET_ATTR_TIMEOUT | nl.NLA_F_NET_BYTEORDER:
			val := attr.Uint32()
			result.Timeout = &val
		case nl.IPSET_ATTR_ELEMENTS | nl.NLA_F_NET_BYTEORDER:
			result.NumEntries = attr.Uint32()
		case nl.IPSET_ATTR_REFERENCES | nl.NLA_F_NET_BYTEORDER:
			result.References = attr.Uint32()
		case nl.IPSET_ATTR_MEMSIZE | nl.NLA_F_NET_BYTEORDER:
			result.SizeInMemory = attr.Uint32()
		case nl.IPSET_ATTR_CADT_FLAGS | nl.NLA_F_NET_BYTEORDER:
			result.CadtFlags = attr.Uint32()
		case nl.IPSET_ATTR_IP | nl.NLA_F_NESTED:
			for nested := range nl.ParseAttributes(attr.Value) {
				switch nested.Type {
				case nl.IPSET_ATTR_IP | nl.NLA_F_NET_BYTEORDER:
					result.Entries = append(result.Entries, IPSetEntry{IP: nested.Value})
				case nl.IPSET_ATTR_IP:
					result.IPFrom = nested.Value
				default:
					log.Printf("unknown nested ipset data attribute from kernel: %+v %v", nested, nested.Type&nl.NLA_TYPE_MASK)
				}
			}
		case nl.IPSET_ATTR_IP_TO | nl.NLA_F_NESTED:
			for nested := range nl.ParseAttributes(attr.Value) {
				switch nested.Type {
				case nl.IPSET_ATTR_IP:
					result.IPTo = nested.Value
				default:
					log.Printf("unknown nested ipset data attribute from kernel: %+v %v", nested, nested.Type&nl.NLA_TYPE_MASK)
				}
			}
		case nl.IPSET_ATTR_PORT_FROM | nl.NLA_F_NET_BYTEORDER:
			result.PortFrom = networkOrder.Uint16(attr.Value)
		case nl.IPSET_ATTR_PORT_TO | nl.NLA_F_NET_BYTEORDER:
			result.PortTo = networkOrder.Uint16(attr.Value)
		case nl.IPSET_ATTR_CADT_LINENO | nl.NLA_F_NET_BYTEORDER:
			result.LineNo = attr.Uint32()
		case nl.IPSET_ATTR_COMMENT:
			result.Comment = nl.BytesToString(attr.Value)
		case nl.IPSET_ATTR_MARKMASK:
			result.MarkMask = attr.Uint32()
		default:
			log.Printf("unknown ipset data attribute from kernel: %+v %v", attr, attr.Type&nl.NLA_TYPE_MASK)
		}
	}
}

func (result *IPSetResult) parseAttrADT(data []byte) {
	for attr := range nl.ParseAttributes(data) {
		switch attr.Type {
		case nl.IPSET_ATTR_DATA | nl.NLA_F_NESTED:
			result.Entries = append(result.Entries, parseIPSetEntry(attr.Value))
		default:
			log.Printf("unknown ADT attribute from kernel: %+v %v", attr, attr.Type&nl.NLA_TYPE_MASK)
		}
	}
}

func parseIPSetEntry(data []byte) (entry IPSetEntry) {
	for attr := range nl.ParseAttributes(data) {
		switch attr.Type {
		case nl.IPSET_ATTR_TIMEOUT | nl.NLA_F_NET_BYTEORDER:
			val := attr.Uint32()
			entry.Timeout = &val
		case nl.IPSET_ATTR_BYTES | nl.NLA_F_NET_BYTEORDER:
			val := attr.Uint64()
			entry.Bytes = &val
		case nl.IPSET_ATTR_PACKETS | nl.NLA_F_NET_BYTEORDER:
			val := attr.Uint64()
			entry.Packets = &val
		case nl.IPSET_ATTR_ETHER:
			entry.MAC = net.HardwareAddr(attr.Value)
		case nl.IPSET_ATTR_IP:
			entry.IP = net.IP(attr.Value)
		case nl.IPSET_ATTR_COMMENT:
			entry.Comment = nl.BytesToString(attr.Value)
		case nl.IPSET_ATTR_IP | nl.NLA_F_NESTED:
			for attr := range nl.ParseAttributes(attr.Value) {
				switch attr.Type {
				case nl.IPSET_ATTR_IPADDR_IPV4, nl.IPSET_ATTR_IPADDR_IPV6:
					entry.IP = net.IP(attr.Value)
				default:
					log.Printf("unknown nested ADT attribute from kernel: %+v", attr)
				}
			}
		case nl.IPSET_ATTR_IP2 | nl.NLA_F_NESTED:
			for attr := range nl.ParseAttributes(attr.Value) {
				switch attr.Type {
				case nl.IPSET_ATTR_IPADDR_IPV4, nl.IPSET_ATTR_IPADDR_IPV6:
					entry.IP2 = net.IP(attr.Value)
				default:
					log.Printf("unknown nested ADT attribute from kernel: %+v", attr)
				}
			}
		case nl.IPSET_ATTR_CIDR:
			entry.CIDR = attr.Value[0]
		case nl.IPSET_ATTR_CIDR2:
			entry.CIDR2 = attr.Value[0]
		case nl.IPSET_ATTR_PORT | nl.NLA_F_NET_BYTEORDER:
			val := networkOrder.Uint16(attr.Value)
			entry.Port = &val
		case nl.IPSET_ATTR_PROTO:
			val := attr.Value[0]
			entry.Protocol = &val
		case nl.IPSET_ATTR_IFACE:
			entry.IFace = nl.BytesToString(attr.Value)
		case nl.IPSET_ATTR_MARK | nl.NLA_F_NET_BYTEORDER:
			val := attr.Uint32()
			entry.Mark = &val
		default:
			log.Printf("unknown ADT attribute from kernel: %+v", attr)
		}
	}
	return
}

func optionsToBitflag(options IpsetCreateOptions) uint32 {
	var cadtFlags uint32

	if options.Comments {
		cadtFlags |= nl.IPSET_FLAG_WITH_COMMENT
	}
	if options.Counters {
		cadtFlags |= nl.IPSET_FLAG_WITH_COUNTERS
	}
	if options.Skbinfo {
		cadtFlags |= nl.IPSET_FLAG_WITH_SKBINFO
	}

	return cadtFlags
}
