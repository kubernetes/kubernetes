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

// LinkAttrs represents data shared by most link types
type RdmaLinkAttrs struct {
	Index           uint32
	Name            string
	FirmwareVersion string
	NodeGuid        string
	SysImageGuid    string
	NumPorts        uint32
}

// Link represents a rdma device from netlink.
type RdmaLink struct {
	Attrs RdmaLinkAttrs
}

func getProtoField(clientType int, op int) int {
	return ((clientType << nl.RDMA_NL_GET_CLIENT_SHIFT) | op)
}

func uint64ToGuidString(guid uint64) string {
	//Convert to byte array
	sysGuidBytes := new(bytes.Buffer)
	binary.Write(sysGuidBytes, binary.LittleEndian, guid)

	//Convert to HardwareAddr
	sysGuidNet := net.HardwareAddr(sysGuidBytes.Bytes())

	//Get the String
	return sysGuidNet.String()
}

func executeOneGetRdmaLink(data []byte) (*RdmaLink, error) {

	link := RdmaLink{}

	reader := bytes.NewReader(data)
	for reader.Len() >= 4 {
		_, attrType, len, value := parseNfAttrTLV(reader)

		switch attrType {
		case nl.RDMA_NLDEV_ATTR_DEV_INDEX:
			var Index uint32
			r := bytes.NewReader(value)
			binary.Read(r, nl.NativeEndian(), &Index)
			link.Attrs.Index = Index
		case nl.RDMA_NLDEV_ATTR_DEV_NAME:
			link.Attrs.Name = string(value[0 : len-1])
		case nl.RDMA_NLDEV_ATTR_FW_VERSION:
			link.Attrs.FirmwareVersion = string(value[0 : len-1])
		case nl.RDMA_NLDEV_ATTR_NODE_GUID:
			var guid uint64
			r := bytes.NewReader(value)
			binary.Read(r, nl.NativeEndian(), &guid)
			link.Attrs.NodeGuid = uint64ToGuidString(guid)
		case nl.RDMA_NLDEV_ATTR_SYS_IMAGE_GUID:
			var sysGuid uint64
			r := bytes.NewReader(value)
			binary.Read(r, nl.NativeEndian(), &sysGuid)
			link.Attrs.SysImageGuid = uint64ToGuidString(sysGuid)
		case nl.RDMA_NLDEV_ATTR_PORT_INDEX:
			var availablePort uint32
			r := bytes.NewReader(value)
			binary.Read(r, nl.NativeEndian(), &availablePort)
			link.Attrs.NumPorts = availablePort
		}
		if (len % 4) != 0 {
			// Skip pad bytes
			reader.Seek(int64(4-(len%4)), seekCurrent)
		}
	}
	return &link, nil
}

func execRdmaSetLink(req *nl.NetlinkRequest) error {

	_, err := req.Execute(unix.NETLINK_RDMA, 0)
	return err
}

// RdmaLinkList gets a list of RDMA link devices.
// Equivalent to: `rdma dev show`
//
// If the returned error is [ErrDumpInterrupted], results may be inconsistent
// or incomplete.
func RdmaLinkList() ([]*RdmaLink, error) {
	return pkgHandle.RdmaLinkList()
}

// RdmaLinkList gets a list of RDMA link devices.
// Equivalent to: `rdma dev show`
//
// If the returned error is [ErrDumpInterrupted], results may be inconsistent
// or incomplete.
func (h *Handle) RdmaLinkList() ([]*RdmaLink, error) {
	proto := getProtoField(nl.RDMA_NL_NLDEV, nl.RDMA_NLDEV_CMD_GET)
	req := h.newNetlinkRequest(proto, unix.NLM_F_ACK|unix.NLM_F_DUMP)

	msgs, executeErr := req.Execute(unix.NETLINK_RDMA, 0)
	if executeErr != nil && !errors.Is(executeErr, ErrDumpInterrupted) {
		return nil, executeErr
	}

	var res []*RdmaLink
	for _, m := range msgs {
		link, err := executeOneGetRdmaLink(m)
		if err != nil {
			return nil, err
		}
		res = append(res, link)
	}

	return res, executeErr
}

// RdmaLinkByName finds a link by name and returns a pointer to the object if
// found and nil error, otherwise returns error code.
//
// If the returned error is [ErrDumpInterrupted], the result may be missing or
// outdated and the caller should retry.
func RdmaLinkByName(name string) (*RdmaLink, error) {
	return pkgHandle.RdmaLinkByName(name)
}

// RdmaLinkByName finds a link by name and returns a pointer to the object if
// found and nil error, otherwise returns error code.
//
// If the returned error is [ErrDumpInterrupted], the result may be missing or
// outdated and the caller should retry.
func (h *Handle) RdmaLinkByName(name string) (*RdmaLink, error) {
	links, err := h.RdmaLinkList()
	if err != nil {
		return nil, err
	}
	for _, link := range links {
		if link.Attrs.Name == name {
			return link, nil
		}
	}
	return nil, fmt.Errorf("Rdma device %v not found", name)
}

// RdmaLinkSetName sets the name of the rdma link device. Return nil on success
// or error otherwise.
// Equivalent to: `rdma dev set $old_devname name $name`
func RdmaLinkSetName(link *RdmaLink, name string) error {
	return pkgHandle.RdmaLinkSetName(link, name)
}

// RdmaLinkSetName sets the name of the rdma link device. Return nil on success
// or error otherwise.
// Equivalent to: `rdma dev set $old_devname name $name`
func (h *Handle) RdmaLinkSetName(link *RdmaLink, name string) error {
	proto := getProtoField(nl.RDMA_NL_NLDEV, nl.RDMA_NLDEV_CMD_SET)
	req := h.newNetlinkRequest(proto, unix.NLM_F_ACK)

	b := make([]byte, 4)
	native.PutUint32(b, uint32(link.Attrs.Index))
	data := nl.NewRtAttr(nl.RDMA_NLDEV_ATTR_DEV_INDEX, b)
	req.AddData(data)

	b = make([]byte, len(name)+1)
	copy(b, name)
	data = nl.NewRtAttr(nl.RDMA_NLDEV_ATTR_DEV_NAME, b)
	req.AddData(data)

	return execRdmaSetLink(req)
}

func netnsModeToString(mode uint8) string {
	switch mode {
	case 0:
		return "exclusive"
	case 1:
		return "shared"
	default:
		return "unknown"
	}
}

func executeOneGetRdmaNetnsMode(data []byte) (string, error) {
	reader := bytes.NewReader(data)
	for reader.Len() >= 4 {
		_, attrType, len, value := parseNfAttrTLV(reader)

		switch attrType {
		case nl.RDMA_NLDEV_SYS_ATTR_NETNS_MODE:
			var mode uint8
			r := bytes.NewReader(value)
			binary.Read(r, nl.NativeEndian(), &mode)
			return netnsModeToString(mode), nil
		}
		if (len % 4) != 0 {
			// Skip pad bytes
			reader.Seek(int64(4-(len%4)), seekCurrent)
		}
	}
	return "", fmt.Errorf("Invalid netns mode")
}

// RdmaSystemGetNetnsMode gets the net namespace mode for RDMA subsystem
// Returns mode string and error status as nil on success or returns error
// otherwise.
// Equivalent to: `rdma system show netns'
func RdmaSystemGetNetnsMode() (string, error) {
	return pkgHandle.RdmaSystemGetNetnsMode()
}

// RdmaSystemGetNetnsMode gets the net namespace mode for RDMA subsystem
// Returns mode string and error status as nil on success or returns error
// otherwise.
// Equivalent to: `rdma system show netns'
func (h *Handle) RdmaSystemGetNetnsMode() (string, error) {

	proto := getProtoField(nl.RDMA_NL_NLDEV, nl.RDMA_NLDEV_CMD_SYS_GET)
	req := h.newNetlinkRequest(proto, unix.NLM_F_ACK)

	msgs, err := req.Execute(unix.NETLINK_RDMA, 0)
	if err != nil {
		return "", err
	}
	if len(msgs) == 0 {
		return "", fmt.Errorf("No valid response from kernel")
	}
	return executeOneGetRdmaNetnsMode(msgs[0])
}

func netnsModeStringToUint8(mode string) (uint8, error) {
	switch mode {
	case "exclusive":
		return 0, nil
	case "shared":
		return 1, nil
	default:
		return 0, fmt.Errorf("Invalid mode; %q", mode)
	}
}

// RdmaSystemSetNetnsMode sets the net namespace mode for RDMA subsystem
// Returns nil on success or appropriate error code.
// Equivalent to: `rdma system set netns { shared | exclusive }'
func RdmaSystemSetNetnsMode(NewMode string) error {
	return pkgHandle.RdmaSystemSetNetnsMode(NewMode)
}

// RdmaSystemSetNetnsMode sets the net namespace mode for RDMA subsystem
// Returns nil on success or appropriate error code.
// Equivalent to: `rdma system set netns { shared | exclusive }'
func (h *Handle) RdmaSystemSetNetnsMode(NewMode string) error {
	value, err := netnsModeStringToUint8(NewMode)
	if err != nil {
		return err
	}

	proto := getProtoField(nl.RDMA_NL_NLDEV, nl.RDMA_NLDEV_CMD_SYS_SET)
	req := h.newNetlinkRequest(proto, unix.NLM_F_ACK)

	data := nl.NewRtAttr(nl.RDMA_NLDEV_SYS_ATTR_NETNS_MODE, []byte{value})
	req.AddData(data)

	_, err = req.Execute(unix.NETLINK_RDMA, 0)
	return err
}

// RdmaLinkSetNsFd puts the RDMA device into a new network namespace. The
// fd must be an open file descriptor to a network namespace.
// Similar to: `rdma dev set $dev netns $ns`
func RdmaLinkSetNsFd(link *RdmaLink, fd uint32) error {
	return pkgHandle.RdmaLinkSetNsFd(link, fd)
}

// RdmaLinkSetNsFd puts the RDMA device into a new network namespace. The
// fd must be an open file descriptor to a network namespace.
// Similar to: `rdma dev set $dev netns $ns`
func (h *Handle) RdmaLinkSetNsFd(link *RdmaLink, fd uint32) error {
	proto := getProtoField(nl.RDMA_NL_NLDEV, nl.RDMA_NLDEV_CMD_SET)
	req := h.newNetlinkRequest(proto, unix.NLM_F_ACK)

	data := nl.NewRtAttr(nl.RDMA_NLDEV_ATTR_DEV_INDEX,
		nl.Uint32Attr(link.Attrs.Index))
	req.AddData(data)

	data = nl.NewRtAttr(nl.RDMA_NLDEV_NET_NS_FD, nl.Uint32Attr(fd))
	req.AddData(data)

	return execRdmaSetLink(req)
}

// RdmaLinkDel deletes an rdma link
//
// Similar to: rdma link delete NAME
// REF: https://man7.org/linux/man-pages/man8/rdma-link.8.html
func RdmaLinkDel(name string) error {
	return pkgHandle.RdmaLinkDel(name)
}

// RdmaLinkDel deletes an rdma link.
//
// If the returned error is [ErrDumpInterrupted], the caller should retry.
func (h *Handle) RdmaLinkDel(name string) error {
	link, err := h.RdmaLinkByName(name)
	if err != nil {
		return err
	}

	proto := getProtoField(nl.RDMA_NL_NLDEV, nl.RDMA_NLDEV_CMD_DELLINK)
	req := h.newNetlinkRequest(proto, unix.NLM_F_ACK)

	b := make([]byte, 4)
	native.PutUint32(b, link.Attrs.Index)
	req.AddData(nl.NewRtAttr(nl.RDMA_NLDEV_ATTR_DEV_INDEX, b))

	_, err = req.Execute(unix.NETLINK_RDMA, 0)
	return err
}

// RdmaLinkAdd adds an rdma link for the specified type to the network device.
// Similar to: rdma link add NAME type TYPE netdev NETDEV
//
//	NAME - specifies the new name of the rdma link to add
//	TYPE - specifies which rdma type to use.  Link types:
//		rxe - Soft RoCE driver
//		siw - Soft iWARP driver
//	NETDEV - specifies the network device to which the link is bound
//
// REF: https://man7.org/linux/man-pages/man8/rdma-link.8.html
func RdmaLinkAdd(linkName, linkType, netdev string) error {
	return pkgHandle.RdmaLinkAdd(linkName, linkType, netdev)
}

// RdmaLinkAdd adds an rdma link for the specified type to the network device.
func (h *Handle) RdmaLinkAdd(linkName string, linkType string, netdev string) error {
	proto := getProtoField(nl.RDMA_NL_NLDEV, nl.RDMA_NLDEV_CMD_NEWLINK)
	req := h.newNetlinkRequest(proto, unix.NLM_F_ACK)

	req.AddData(nl.NewRtAttr(nl.RDMA_NLDEV_ATTR_DEV_NAME, nl.ZeroTerminated(linkName)))
	req.AddData(nl.NewRtAttr(nl.RDMA_NLDEV_ATTR_LINK_TYPE, nl.ZeroTerminated(linkType)))
	req.AddData(nl.NewRtAttr(nl.RDMA_NLDEV_ATTR_NDEV_NAME, nl.ZeroTerminated(netdev)))
	_, err := req.Execute(unix.NETLINK_RDMA, 0)
	return err
}

// RdmaResource represents a rdma device resource tracking summaries
type RdmaResource struct {
	Index                      uint32
	Name                       string
	RdmaResourceSummaryEntries map[string]uint64
}

// RdmaResourceList list rdma resource tracking information
// Returns all rdma devices resource tracking summary on success or returns error
// otherwise.
// Equivalent to: `rdma resource'
func RdmaResourceList() ([]*RdmaResource, error) {
	return pkgHandle.RdmaResourceList()
}

// RdmaResourceList list rdma resource tracking information
// Returns all rdma devices resource tracking summary on success or returns error
// otherwise.
// Equivalent to: `rdma resource'
func (h *Handle) RdmaResourceList() ([]*RdmaResource, error) {
	proto := getProtoField(nl.RDMA_NL_NLDEV, nl.RDMA_NLDEV_CMD_RES_GET)
	req := h.newNetlinkRequest(proto, unix.NLM_F_ACK|unix.NLM_F_DUMP)

	msgs, err := req.Execute(unix.NETLINK_RDMA, 0)
	if err != nil {
		return nil, err
	}
	if len(msgs) == 0 {
		return nil, fmt.Errorf("No valid response from kernel")
	}
	var rdmaResources []*RdmaResource
	for _, msg := range msgs {
		res, err := executeOneGetRdmaResourceList(msg)
		if err != nil {
			return nil, err
		}
		rdmaResources = append(rdmaResources, res)
	}
	return rdmaResources, nil
}

func parseRdmaCounters(counterType uint16, data []byte) (map[string]uint64, error) {
	var counterKeyType, counterValueType uint16
	switch counterType {
	case nl.RDMA_NLDEV_ATTR_RES_SUMMARY_ENTRY:
		counterKeyType = nl.RDMA_NLDEV_ATTR_RES_SUMMARY_ENTRY_NAME
		counterValueType = nl.RDMA_NLDEV_ATTR_RES_SUMMARY_ENTRY_CURR
	case nl.RDMA_NLDEV_ATTR_STAT_HWCOUNTER_ENTRY:
		counterKeyType = nl.RDMA_NLDEV_ATTR_STAT_HWCOUNTER_ENTRY_NAME
		counterValueType = nl.RDMA_NLDEV_ATTR_STAT_HWCOUNTER_ENTRY_VALUE
	default:
		return nil, fmt.Errorf("Invalid counter type: %d", counterType)
	}
	counters := make(map[string]uint64)
	reader := bytes.NewReader(data)

	for reader.Len() >= 4 {
		_, attrType, _, value := parseNfAttrTLV(reader)
		if attrType != counterType {
			return nil, fmt.Errorf("Invalid resource summary entry type; %d", attrType)
		}

		summaryReader := bytes.NewReader(value)
		for summaryReader.Len() >= 4 {
			_, attrType, len, value := parseNfAttrTLV(summaryReader)
			if attrType != counterKeyType {
				return nil, fmt.Errorf("Invalid resource summary entry name type; %d", attrType)
			}
			name := string(value[0 : len-1])
			// Skip pad bytes
			if (len % 4) != 0 {
				summaryReader.Seek(int64(4-(len%4)), seekCurrent)
			}
			_, attrType, len, value = parseNfAttrTLV(summaryReader)
			if attrType != counterValueType {
				return nil, fmt.Errorf("Invalid resource summary entry value type; %d", attrType)
			}
			counters[name] = native.Uint64(value)
		}
	}
	return counters, nil
}

func executeOneGetRdmaResourceList(data []byte) (*RdmaResource, error) {
	var res RdmaResource
	reader := bytes.NewReader(data)
	for reader.Len() >= 4 {
		_, attrType, len, value := parseNfAttrTLV(reader)

		switch attrType {
		case nl.RDMA_NLDEV_ATTR_DEV_INDEX:
			var Index uint32
			r := bytes.NewReader(value)
			binary.Read(r, nl.NativeEndian(), &Index)
			res.Index = Index
		case nl.RDMA_NLDEV_ATTR_DEV_NAME:
			res.Name = string(value[0 : len-1])
		case nl.RDMA_NLDEV_ATTR_RES_SUMMARY:
			var err error
			res.RdmaResourceSummaryEntries, err = parseRdmaCounters(nl.RDMA_NLDEV_ATTR_RES_SUMMARY_ENTRY, value)
			if err != nil {
				return nil, err
			}
		}
		if (len % 4) != 0 {
			// Skip pad bytes
			reader.Seek(int64(4-(len%4)), seekCurrent)
		}
	}
	return &res, nil
}

// RdmaPortStatistic represents a rdma port statistic counter
type RdmaPortStatistic struct {
	PortIndex  uint32
	Statistics map[string]uint64
}

// RdmaDeviceStatistic represents a rdma device statistic counter
type RdmaDeviceStatistic struct {
	RdmaPortStatistics []*RdmaPortStatistic
}

// RdmaStatistic get rdma device statistic counters
// Returns rdma device statistic counters on success or returns error
// otherwise.
// Equivalent to: `rdma statistic show link [DEV]'
func RdmaStatistic(link *RdmaLink) (*RdmaDeviceStatistic, error) {
	return pkgHandle.RdmaStatistic(link)
}

// RdmaStatistic get rdma device statistic counters
// Returns rdma device statistic counters on success or returns error
// otherwise.
// Equivalent to: `rdma statistic show link [DEV]'
func (h *Handle) RdmaStatistic(link *RdmaLink) (*RdmaDeviceStatistic, error) {
	rdmaLinkStatistic := make([]*RdmaPortStatistic, 0)
	for portIndex := uint32(1); portIndex <= link.Attrs.NumPorts; portIndex++ {
		portStatistic, err := h.RdmaPortStatisticList(link, portIndex)
		if err != nil {
			return nil, err
		}
		rdmaLinkStatistic = append(rdmaLinkStatistic, portStatistic)
	}
	return &RdmaDeviceStatistic{RdmaPortStatistics: rdmaLinkStatistic}, nil
}

// RdmaPortStatisticList get rdma device port statistic counters
// Returns rdma device port statistic counters on success or returns error
// otherwise.
// Equivalent to: `rdma statistic show link [DEV/PORT]'
func RdmaPortStatisticList(link *RdmaLink, port uint32) (*RdmaPortStatistic, error) {
	return pkgHandle.RdmaPortStatisticList(link, port)
}

// RdmaPortStatisticList get rdma device port statistic counters
// Returns rdma device port statistic counters on success or returns error
// otherwise.
// Equivalent to: `rdma statistic show link [DEV/PORT]'
func (h *Handle) RdmaPortStatisticList(link *RdmaLink, port uint32) (*RdmaPortStatistic, error) {
	proto := getProtoField(nl.RDMA_NL_NLDEV, nl.RDMA_NLDEV_CMD_STAT_GET)
	req := h.newNetlinkRequest(proto, unix.NLM_F_ACK|unix.NLM_F_REQUEST)
	b := make([]byte, 4)
	native.PutUint32(b, link.Attrs.Index)
	data := nl.NewRtAttr(nl.RDMA_NLDEV_ATTR_DEV_INDEX, b)
	req.AddData(data)

	b = make([]byte, 4)
	native.PutUint32(b, port)
	data = nl.NewRtAttr(nl.RDMA_NLDEV_ATTR_PORT_INDEX, b)
	req.AddData(data)

	msgs, err := req.Execute(unix.NETLINK_RDMA, 0)
	if err != nil {
		return nil, err
	}
	if len(msgs) != 1 {
		return nil, fmt.Errorf("No valid response from kernel")
	}
	return executeOneGetRdmaPortStatistics(msgs[0])
}

func executeOneGetRdmaPortStatistics(data []byte) (*RdmaPortStatistic, error) {
	var stat RdmaPortStatistic
	reader := bytes.NewReader(data)
	for reader.Len() >= 4 {
		_, attrType, len, value := parseNfAttrTLV(reader)

		switch attrType {
		case nl.RDMA_NLDEV_ATTR_PORT_INDEX:
			var Index uint32
			r := bytes.NewReader(value)
			binary.Read(r, nl.NativeEndian(), &Index)
			stat.PortIndex = Index
		case nl.RDMA_NLDEV_ATTR_STAT_HWCOUNTERS:
			var err error
			stat.Statistics, err = parseRdmaCounters(nl.RDMA_NLDEV_ATTR_STAT_HWCOUNTER_ENTRY, value)
			if err != nil {
				return nil, err
			}
		}
		if (len % 4) != 0 {
			// Skip pad bytes
			reader.Seek(int64(4-(len%4)), seekCurrent)
		}
	}
	return &stat, nil
}
