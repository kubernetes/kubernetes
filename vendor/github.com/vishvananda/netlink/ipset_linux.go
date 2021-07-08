package netlink

import (
	"log"
	"net"
	"syscall"

	"github.com/vishvananda/netlink/nl"
	"golang.org/x/sys/unix"
)

// IPSetEntry is used for adding, updating, retreiving and deleting entries
type IPSetEntry struct {
	Comment string
	MAC     net.HardwareAddr
	IP      net.IP
	Timeout *uint32
	Packets *uint64
	Bytes   *uint64

	Replace bool // replace existing entry
}

// IPSetResult is the result of a dump request for a set
type IPSetResult struct {
	Nfgenmsg *nl.Nfgenmsg
	Protocol uint8
	Revision uint8
	Family   uint8
	Flags    uint8
	SetName  string
	TypeName string

	HashSize     uint32
	NumEntries   uint32
	MaxElements  uint32
	References   uint32
	SizeInMemory uint32
	CadtFlags    uint32
	Timeout      *uint32

	Entries []IPSetEntry
}

// IpsetCreateOptions is the options struct for creating a new ipset
type IpsetCreateOptions struct {
	Replace  bool // replace existing ipset
	Timeout  *uint32
	Counters bool
	Comments bool
	Skbinfo  bool
}

// IpsetProtocol returns the ipset protocol version from the kernel
func IpsetProtocol() (uint8, error) {
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
	return pkgHandle.ipsetAddDel(nl.IPSET_CMD_ADD, setname, entry)
}

// IpsetDele deletes an entry from an existing ipset.
func IpsetDel(setname string, entry *IPSetEntry) error {
	return pkgHandle.ipsetAddDel(nl.IPSET_CMD_DEL, setname, entry)
}

func (h *Handle) IpsetProtocol() (uint8, error) {
	req := h.newIpsetRequest(nl.IPSET_CMD_PROTOCOL)
	msgs, err := req.Execute(unix.NETLINK_NETFILTER, 0)

	if err != nil {
		return 0, err
	}

	return ipsetUnserialize(msgs).Protocol, nil
}

func (h *Handle) IpsetCreate(setname, typename string, options IpsetCreateOptions) error {
	req := h.newIpsetRequest(nl.IPSET_CMD_CREATE)

	if !options.Replace {
		req.Flags |= unix.NLM_F_EXCL
	}

	req.AddData(nl.NewRtAttr(nl.IPSET_ATTR_SETNAME, nl.ZeroTerminated(setname)))
	req.AddData(nl.NewRtAttr(nl.IPSET_ATTR_TYPENAME, nl.ZeroTerminated(typename)))
	req.AddData(nl.NewRtAttr(nl.IPSET_ATTR_REVISION, nl.Uint8Attr(0)))
	req.AddData(nl.NewRtAttr(nl.IPSET_ATTR_FAMILY, nl.Uint8Attr(0)))

	data := nl.NewRtAttr(nl.IPSET_ATTR_DATA|int(nl.NLA_F_NESTED), nil)

	if timeout := options.Timeout; timeout != nil {
		data.AddChild(&nl.Uint32Attribute{Type: nl.IPSET_ATTR_TIMEOUT | nl.NLA_F_NET_BYTEORDER, Value: *timeout})
	}

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

func (h *Handle) ipsetAddDel(nlCmd int, setname string, entry *IPSetEntry) error {
	req := h.newIpsetRequest(nlCmd)
	req.AddData(nl.NewRtAttr(nl.IPSET_ATTR_SETNAME, nl.ZeroTerminated(setname)))
	data := nl.NewRtAttr(nl.IPSET_ATTR_DATA|int(nl.NLA_F_NESTED), nil)

	if !entry.Replace {
		req.Flags |= unix.NLM_F_EXCL
	}

	if entry.Timeout != nil {
		data.AddChild(&nl.Uint32Attribute{Type: nl.IPSET_ATTR_TIMEOUT | nl.NLA_F_NET_BYTEORDER, Value: *entry.Timeout})
	}
	if entry.MAC != nil {
		data.AddChild(nl.NewRtAttr(nl.IPSET_ATTR_ETHER, entry.MAC))
	}

	data.AddChild(&nl.Uint32Attribute{Type: nl.IPSET_ATTR_LINENO | nl.NLA_F_NET_BYTEORDER, Value: 0})
	req.AddData(data)

	_, err := ipsetExecute(req)
	return err
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
		case nl.IPSET_ATTR_COMMENT:
			entry.Comment = nl.BytesToString(attr.Value)
		case nl.IPSET_ATTR_IP | nl.NLA_F_NESTED:
			for attr := range nl.ParseAttributes(attr.Value) {
				switch attr.Type {
				case nl.IPSET_ATTR_IP:
					entry.IP = net.IP(attr.Value)
				default:
					log.Printf("unknown nested ADT attribute from kernel: %+v", attr)
				}
			}
		default:
			log.Printf("unknown ADT attribute from kernel: %+v", attr)
		}
	}
	return
}
